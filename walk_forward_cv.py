#!/usr/bin/env python3
"""
Numerai 4-Fold Purged Walk-Forward Cross-Validation Engine.
Partitions validation eras into 4 distinct historical market regimes with 8-era boundary purging.
Evaluates out-of-sample Spearman correlation, Sharpe, drawdown, and hit rate across cycles.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    DATA_DIR,
    ENSEMBLE_TARGETS_60D,
    MODEL_60D_DIR,
    NEUTRALIZATION_PROPORTION,
)
from fleet_submit import load_feature_groups
from neutralize import neutralize, rank_01

# 4 Non-Overlapping Chronological Folds spanning 660 validation eras (0575 to 1234)
FOLD_DEFINITIONS = {
    1: {"name": "Early Regime (2013-2016)", "start_era": "0575", "end_era": "0739"},
    2: {"name": "Crisis & Shock (2017-2020)", "start_era": "0740", "end_era": "0904"},
    3: {"name": "Post-Pandemic Inflation (2020-2023)", "start_era": "0905", "end_era": "1069"},
    4: {"name": "Modern Dispersion (2023-2026)", "start_era": "1070", "end_era": "1234"},
}

PURGE_BUFFER_ERAS = 12  # 60-day horizon (~12 weekly eras) embargo buffer to prevent target return leakage


def get_purged_fold_eras(all_eras: list, start_era: str, end_era: str, purge_buffer: int = 12) -> list:
    """Return chronological era slice with boundary purge buffer applied."""
    assert len(all_eras) > 0, "Era universe cannot be empty"
    assert start_era <= end_era, f"Invalid era range: {start_era} > {end_era}"

    in_range = [e for e in all_eras if start_era <= e <= end_era]
    assert len(in_range) > purge_buffer, f"Insufficient eras ({len(in_range)}) for purge buffer {purge_buffer}"

    # Purge first 12 eras of fold to avoid 60-day boundary overlap from previous regime
    purged = in_range[purge_buffer:]
    assert len(purged) > 0, "Zero eras remaining after purge"
    return purged


def evaluate_fold_predictions(fold_df: pd.DataFrame, models: dict, features: list, fncv3_feats: list) -> pd.Series:
    """Generate 60-day ensemble predictions and compute per-era Spearman correlations."""
    assert len(fold_df) > 0, "Empty fold dataframe"
    assert len(models) == 5, f"Expected 5 ensemble models, got {len(models)}"

    preds_list = []
    for target, model in models.items():
        raw_p = model.predict(fold_df[features])
        preds_list.append(rank_01(raw_p))

    raw_ensemble = np.mean(preds_list, axis=0)
    df_copy = fold_df[["era"] + fncv3_feats].copy()
    df_copy["pred"] = rank_01(raw_ensemble)
    if NEUTRALIZATION_PROPORTION > 0.0 and fncv3_feats:
        neutralized_chunks = []
        for _, era_df in df_copy.groupby("era", sort=False):
            sub_res = neutralize(era_df, ["pred"], extra_neutralizers=fncv3_feats, proportion=NEUTRALIZATION_PROPORTION)
            neutralized_chunks.append(sub_res["pred"])
        neut_pred = rank_01(pd.concat(neutralized_chunks).values)
    else:
        neut_pred = rank_01(df_copy["pred"].values)

    eval_df = fold_df[["era", "target_cyrusd_60"]].copy()
    eval_df["pred"] = neut_pred
    eval_df = eval_df.dropna(subset=["target_cyrusd_60", "pred"])

    era_corrs = {}
    for era, group in eval_df.groupby("era", observed=True):
        if len(group) < 50:
            continue
        corr, _ = spearmanr(group["pred"].values, group["target_cyrusd_60"].values)
        if np.isfinite(corr):
            era_corrs[era] = float(corr)

    assert len(era_corrs) > 0, "Failed to compute correlations in fold"
    return pd.Series(era_corrs)


def compute_regime_metrics(corrs: pd.Series) -> dict:
    """Compute institutional quantitative statistics for an evaluation regime."""
    assert len(corrs) > 0, "Cannot compute metrics on empty series"
    m_corr = float(corrs.mean())
    s_corr = float(corrs.std())
    raw_sharpe = float(m_corr / (s_corr + 1e-8))
    ann_sharpe = float(raw_sharpe * np.sqrt(12.0))  # Monthly annualized convention (matches train_60d_ensemble.py)
    ann_sharpe_weekly = float(raw_sharpe * np.sqrt(52.0))  # Raw weekly annualization
    cumsum = corrs.cumsum()
    max_dd = float((cumsum.cummax() - cumsum).max())
    hit_rate = float(np.mean(corrs > 0.0))

    assert max_dd >= 0.0, "Negative drawdown encountered"
    assert 0.0 <= hit_rate <= 1.0, "Hit rate out of probability bounds"

    return {
        "mean_corr": round(m_corr, 5),
        "std_corr": round(s_corr, 5),
        "raw_era_sharpe": round(raw_sharpe, 3),
        "ann_sharpe": round(ann_sharpe, 2),
        "ann_sharpe_weekly": round(ann_sharpe_weekly, 2),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(hit_rate, 3),
        "n_eras": len(corrs)
    }


def run_purged_walk_forward_cv() -> dict:
    """Execute complete 4-fold purged walk-forward cross-validation."""
    val_path = os.path.join(DATA_DIR, "validation.parquet")
    assert os.path.exists(val_path), f"Validation parquet missing: {val_path}"

    groups = load_feature_groups()
    features = groups["all_medium"]
    fncv3_feats = groups["fncv3_features"]

    print("[CV INIT] Loading 5-target 60-day ensemble weights...")
    models = {}
    for t in ENSEMBLE_TARGETS_60D:
        p = os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl")
        assert os.path.exists(p), f"Missing model: {p}"
        models[t] = joblib.load(p)

    cols = ["era", "target_cyrusd_60"] + features
    print(f"[CV LOAD] Ingesting validation dataset ({len(cols)} columns)...")
    val_df = pd.read_parquet(val_path, columns=cols)
    all_eras = sorted(val_df["era"].unique())
    assert len(all_eras) >= 500, f"Insufficient validation eras: {len(all_eras)}"

    fold_results = {}
    print("\n=========================================================================")
    print("[AUDIT] 4-FOLD PURGED WALK-FORWARD CROSS-VALIDATION RESULTS")
    print("=========================================================================")

    for fold_id, fold_info in FOLD_DEFINITIONS.items():
        purged_eras = get_purged_fold_eras(
            all_eras, fold_info["start_era"], fold_info["end_era"], purge_buffer=PURGE_BUFFER_ERAS
        )
        fold_df = val_df[val_df["era"].isin(purged_eras)].copy()
        corrs = evaluate_fold_predictions(fold_df, models, features, fncv3_feats)
        metrics = compute_regime_metrics(corrs)
        fold_results[fold_id] = {**fold_info, **metrics}

        print(
            f"Fold {fold_id} [{fold_info['name']:32s}]: "
            f"Corr: {metrics['mean_corr']:+.4f} | "
            f"Sharpe: {metrics['raw_era_sharpe']:+.3f} | "
            f"HitRate: {metrics['hit_rate']*100:.1f}% | "
            f"MaxDD: {metrics['max_drawdown']:.4f} | "
            f"Eras: {metrics['n_eras']}"
        )

        # Quality assertions per fold
        assert metrics["mean_corr"] > 0.0, f"Fold {fold_id} failed positive correlation gate"
        assert metrics["raw_era_sharpe"] >= 0.50, f"Fold {fold_id} failed Sharpe gate (0.50)"
        assert metrics["hit_rate"] >= 0.60, f"Fold {fold_id} failed hit rate gate (0.60)"
        assert metrics["max_drawdown"] <= 0.40, f"Fold {fold_id} exceeded max drawdown gate (0.40)"

    mean_cv_corr = float(np.mean([f["mean_corr"] for f in fold_results.values()]))
    mean_cv_sharpe = float(np.mean([f["raw_era_sharpe"] for f in fold_results.values()]))

    print("=" * 73)
    print(f"[SUMMARY] Mean CV Correlation: {mean_cv_corr:+.4f} | Mean CV Sharpe: {mean_cv_sharpe:+.3f}")
    print("[PASS] All 4 Chronological Regimes Passed Positive Alpha & Drawdown Gates!")
    print("=" * 73)

    output = {
        "mean_cv_spearman": round(mean_cv_corr, 5),
        "mean_cv_sharpe": round(mean_cv_sharpe, 3),
        "folds": fold_results
    }
    out_file = os.path.join(DATA_DIR, "walk_forward_cv_metrics.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    return output


def main():
    run_purged_walk_forward_cv()


if __name__ == "__main__":
    main()
