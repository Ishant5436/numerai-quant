#!/usr/bin/env python3
"""
Numerai 60-Day Multi-Target Alpha Training Engine (v5.0 Standard).
Trains 5 specialized LightGBM regressors on the 60-day target quintet.
Computes out-of-sample Spearman Correlation, Raw Era Sharpe, and Drawdown.
Enforces Power of 10 safety invariants and the 95th+ percentile quality gate (Sharpe >= 1.15).
"""

import json
import os
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    DATA_DIR,
    ENSEMBLE_TARGETS_60D,
    FEATURES_JSON,
    LGB_PARAMS,
    MODEL_60D_DIR,
    NEUTRALIZATION_PROPORTION,
)
from neutralize import neutralize


def get_feature_metadata() -> tuple[list, list]:
    """Extract medium and canonical fncv3 feature lists from features.json."""
    assert os.path.exists(FEATURES_JSON), f"Missing features file: {FEATURES_JSON}"
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    medium = meta["feature_sets"]["medium"]
    fncv3 = meta["feature_sets"]["fncv3_features"]
    assert len(medium) == 705, f"Unexpected medium features count: {len(medium)}"
    assert len(fncv3) == 400, f"Unexpected fncv3 features count: {len(fncv3)}"
    return medium, fncv3


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target_cyrusd_60") -> pd.Series:
    """Compute per-era Spearman rank correlation with non-null defensive check."""
    assert pred_col in df.columns, f"Prediction column missing: {pred_col}"
    assert target_col in df.columns, f"Target column missing: {target_col}"

    def era_corr(era_df):
        valid = era_df.dropna(subset=[pred_col, target_col])
        if len(valid) < 10:
            return 0.0
        return float(spearmanr(valid[pred_col], valid[target_col]).statistic)

    return df.groupby("era", observed=True).apply(era_corr)


def train_single_target_model(train_df: pd.DataFrame, target: str, features: list, model_dir: str) -> str:
    """Train and persist a single LightGBM model for a given 60-day target."""
    assert target in train_df.columns, f"Target missing in train dataframe: {target}"
    assert len(features) > 0, "Feature list cannot be empty"

    model_path = os.path.join(model_dir, f"lgb_{target}.pkl")
    print(f"[RUN] Training LightGBM on target: '{target}' across {len(features)} features...")
    t0 = time.time()

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train_df[features], train_df[target])
    joblib.dump(model, model_path)

    elapsed = time.time() - t0
    print(f"[OK] Saved model to {model_path} ({elapsed:.1f}s)")
    assert os.path.exists(model_path), f"Failed to persist model: {model_path}"
    return model_path


def load_training_data(train_path: str, targets: list, features: list, sample_stride: int = 1) -> pd.DataFrame:
    """Load train.parquet with bounded columns and validation assertions."""
    assert os.path.exists(train_path), f"Train dataset missing: {train_path}"
    assert len(targets) > 0, "Targets list cannot be empty"

    cols_to_load = ["era"] + targets + features
    print(f"[LOAD] Loading training dataset ({len(cols_to_load)} columns)...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    if sample_stride > 1:
        eras = sorted(train_df["era"].unique())
        selected_eras = eras[::sample_stride]
        train_df = train_df[train_df["era"].isin(selected_eras)].copy()
        print(f"[INFO] Sampled {len(selected_eras)}/{len(eras)} training eras (stride={sample_stride})")

    assert len(train_df) > 0, "Loaded empty training set"
    return train_df


def evaluate_60d_ensemble(val_path: str, models: dict, features: list, fncv3_feats: list) -> dict:
    """Evaluate individual models, blended ensemble, and FNCv3 neutralization out-of-sample."""
    assert os.path.exists(val_path), f"Validation dataset missing: {val_path}"
    assert len(models) == 5, f"Expected 5 models, found: {len(models)}"

    val_cols = ["era", "target_cyrusd_60"] + features
    print(f"[LOAD] Loading validation dataset ({len(val_cols)} columns)...")
    val_df = pd.read_parquet(val_path, columns=val_cols)
    val_df = val_df.dropna(subset=["target_cyrusd_60"]).copy()
    assert len(val_df) > 0, "No valid validation eras found with target_cyrusd_60"

    pred_cols = []
    print("\n[EVAL] Evaluating individual 60-day target models out-of-sample:")
    for target, model in models.items():
        pred_col = f"pred_{target}"
        pred_cols.append(pred_col)
        val_df[f"raw_{target}"] = model.predict(val_df[features])
        val_df[pred_col] = val_df.groupby("era", observed=True)[f"raw_{target}"].rank(pct=True)
        val_df.drop(columns=[f"raw_{target}"], inplace=True)
        corr = calculate_era_correlation(val_df, pred_col)
        m_corr = float(corr.mean())
        s_corr = float(corr.std())
        sharpe = m_corr / (s_corr + 1e-8)
        print(f"  • Target [{target:17s}] -> Mean Corr: {m_corr:+.4f} | Raw Sharpe: {sharpe:+.3f}")

    # Build Rank Blend
    val_df["ensemble_blend"] = val_df[pred_cols].mean(axis=1)
    val_df["ensemble_blend"] = val_df.groupby("era", observed=True)["ensemble_blend"].rank(pct=True)

    corr_raw = calculate_era_correlation(val_df, "ensemble_blend")
    sharpe_raw = float(corr_raw.mean() / (corr_raw.std() + 1e-8))
    print(f"\n[BENCHMARK] Blended 5-Target Ensemble (Raw) -> Mean Corr: {corr_raw.mean():+.4f} | Raw Sharpe: {sharpe_raw:+.3f}")

    # Apply Canonical 400 FNCv3 Neutralization
    print(f"[NEUTRALIZE] Applying {int(NEUTRALIZATION_PROPORTION*100)}% QR feature neutralization across {len(fncv3_feats)} factors...")
    val_df["ensemble_neutral"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era_df: pd.Series(
            neutralize(era_df, ["ensemble_blend"], extra_neutralizers=fncv3_feats, proportion=NEUTRALIZATION_PROPORTION)["ensemble_blend"].values,
            index=era_df.index
        )
    )

    corr_neut = calculate_era_correlation(val_df, "ensemble_neutral")
    mean_neut = float(corr_neut.mean())
    std_neut = float(corr_neut.std())
    raw_sharpe = float(mean_neut / (std_neut + 1e-8))
    ann_sharpe = float(raw_sharpe * np.sqrt(12))
    cumsum = corr_neut.cumsum()
    max_dd = float((cumsum.cummax() - cumsum).max())

    metrics = {
        "mean_corr": mean_neut,
        "std_corr": std_neut,
        "raw_era_sharpe": raw_sharpe,
        "ann_sharpe": ann_sharpe,
        "max_drawdown": max_dd,
    }
    return metrics


def main():
    """Main pipeline execution entry point."""
    print("=========================================================================")
    print("[INIT] NUMERAI 60-DAY MULTI-TARGET ALPHA TRAINING ENGINE (95%ILE GATE)")
    print("=========================================================================")
    os.makedirs(MODEL_60D_DIR, exist_ok=True)
    features, fncv3_features = get_feature_metadata()
    print(f"[INFO] Features Universe: {len(features)} medium features | {len(fncv3_features)} FNCv3 risk factors")
    print(f"[INFO] Target Quintet: {ENSEMBLE_TARGETS_60D}")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    missing = [t for t in ENSEMBLE_TARGETS_60D if not os.path.exists(os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl"))]
    if missing:
        train_df = load_training_data(train_path, ENSEMBLE_TARGETS_60D, features, sample_stride=2)
        for target in missing:
            train_single_target_model(train_df, target, features, MODEL_60D_DIR)
        del train_df
    else:
        print(f"[INFO] All 5 60-day models already trained in {MODEL_60D_DIR}.")

    models = {}
    for target in ENSEMBLE_TARGETS_60D:
        model_file = os.path.join(MODEL_60D_DIR, f"lgb_{target}.pkl")
        assert os.path.exists(model_file), f"Missing model weight: {model_file}"
        models[target] = joblib.load(model_file)

    metrics = evaluate_60d_ensemble(val_path, models, features, fncv3_features)

    print("\n" + "=" * 60)
    print("[REPORT] 60-DAY MULTI-TARGET FNCV3 ENSEMBLE VALIDATION AUDIT")
    print("=" * 60)
    print(f"• Mean Era Correlation (corr60)   : {metrics['mean_corr']:+.4f}")
    print(f"• Raw Per-Era Sharpe (mu/sigma)   : {metrics['raw_era_sharpe']:+.3f} (Leaderboard Standard)")
    print(f"• Annualized Sharpe (Monthly sq12): {metrics['ann_sharpe']:+.2f}")
    print(f"• Peak-to-Trough Max Drawdown     : {metrics['max_drawdown']*100:.2f}%")
    print("=" * 60)

    # 95th Percentile Quality Gate & Risk Gate Assertions
    assert metrics["raw_era_sharpe"] >= 1.15, f"Quality Gate Failure: Sharpe {metrics['raw_era_sharpe']} < 1.15"
    assert metrics["max_drawdown"] <= 0.40, f"Risk Gate Failure: 647-era Max Drawdown {metrics['max_drawdown']} > 0.40"
    print("[PASS] 95th+ Percentile Quality Gate & Risk Invariants Satisfied!")


if __name__ == "__main__":
    main()
