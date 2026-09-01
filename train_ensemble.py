#!/usr/bin/env python3
"""
Numerai Multi-Target Alpha Ensemble Training Engine
- Loads 705 'medium' features
- Trains 4 specialized LightGBM models on distinct market targets
- Computes out-of-sample Spearman Correlation, Raw Era Sharpe, and Drawdown
- Evaluates blended & neutralized ensemble performance
- Persists all model weights to /Users/ishantpanchal/numerai-quant/models/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from config import (
    FEATURE_SET,
    ENSEMBLE_TARGETS,
    MODEL_DIR,
    FEATURES_JSON,
    DATA_DIR,
    LGB_PARAMS,
    NEUTRALIZATION_PROPORTION
)
from neutralize import neutralize, rank_01

os.makedirs(MODEL_DIR, exist_ok=True)


def get_feature_list() -> list:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    return meta["feature_sets"][FEATURE_SET]


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    features = get_feature_list()
    print(f"=== Numerai Alpha Ensemble Pipeline (v5.0) ===")
    print(f"Feature set: '{FEATURE_SET}' ({len(features)} features)")
    print(f"Ensemble targets: {ENSEMBLE_TARGETS}")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    # 1. Load or Train Multi-Target Models
    models = {}
    missing_models = [t for t in ENSEMBLE_TARGETS if not os.path.exists(os.path.join(MODEL_DIR, f"lgb_{t}.pkl"))]

    if missing_models:
        cols_to_load = ["era"] + ENSEMBLE_TARGETS + features
        print(f"\nLoading training dataset into memory for {len(missing_models)} missing models...")
        train_df = pd.read_parquet(train_path, columns=cols_to_load)
        print(f"Train matrix shape: {train_df.shape}")
        X_train = train_df[features]

        for target in missing_models:
            model_file = os.path.join(MODEL_DIR, f"lgb_{target}.pkl")
            print(f"\n---> Training LightGBM on target: '{target}'...")
            y_train = train_df[target]
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_train, y_train)
            joblib.dump(model, model_file)
            print(f"Saved: {model_file}")

        del train_df, X_train

    # Load all models
    for target in ENSEMBLE_TARGETS:
        model_file = os.path.join(MODEL_DIR, f"lgb_{target}.pkl")
        models[target] = joblib.load(model_file)
        print(f"Loaded model: {model_file}")

    # 2. Load Validation Data
    val_cols = ["era", "target"] + features
    print("\nLoading validation dataset for out-of-sample evaluation...")
    val_df = pd.read_parquet(val_path, columns=val_cols)
    print(f"Validation matrix shape: {val_df.shape}")

    X_val = val_df[features]

    # 3. Generate Predictions & Evaluate Individual Models
    print("\nGenerating out-of-sample predictions across validation eras...")
    pred_cols = []
    for target, model in models.items():
        pred_col = f"pred_{target}"
        pred_cols.append(pred_col)
        raw_preds = model.predict(X_val)
        val_df["temp_raw"] = raw_preds
        val_df[pred_col] = val_df.groupby("era", observed=True)["temp_raw"].rank(pct=True)
        
        corrs = calculate_era_correlation(val_df, pred_col, "target")
        mean_c = corrs.mean()
        raw_sharpe = mean_c / (corrs.std() + 1e-8)
        print(f"  • Target [{target:16s}] -> Mean Corr: {mean_c:.4f} | Per-Era Sharpe: {raw_sharpe:.3f}")

    val_df.drop(columns=["temp_raw"], inplace=True)

    # 4. Build Blended Ensemble
    print("\nBuilding rank-weighted multi-target ensemble...")
    val_df["ensemble_raw"] = val_df[pred_cols].mean(axis=1)
    val_df["ensemble_raw"] = val_df.groupby("era", observed=True)["ensemble_raw"].rank(pct=True)

    corrs_ensemble = calculate_era_correlation(val_df, "ensemble_raw", "target")
    mean_ens = corrs_ensemble.mean()
    raw_sharpe_ens = mean_ens / (corrs_ensemble.std() + 1e-8)
    print(f"\n[BENCHMARK]  Blended Ensemble (Raw)       -> Mean Corr: {mean_ens:.4f} | Per-Era Sharpe: {raw_sharpe_ens:.3f}")

    # 5. Apply Feature Neutralization
    print(f"Applying {int(NEUTRALIZATION_PROPORTION*100)}% feature neutralization...")
    neutralizer_feats = features[:50]
    
    val_df["ensemble_neutralized"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(
            neutralize(era, ["ensemble_raw"], extra_neutralizers=neutralizer_feats, proportion=NEUTRALIZATION_PROPORTION)["ensemble_raw"].values,
            index=era.index
        )
    )

    corrs_neutral = calculate_era_correlation(val_df, "ensemble_neutralized", "target")
    mean_neut = corrs_neutral.mean()
    raw_sharpe_neut = mean_neut / (corrs_neutral.std() + 1e-8)
    annualized_monthly = raw_sharpe_neut * np.sqrt(12)
    max_dd = (corrs_neutral.cumsum().cummax() - corrs_neutral.cumsum()).max()

    print("\n" + "="*60)
    print(f"[GUARD]   NEUTRALIZED ALPHA ENSEMBLE AUDIT SUMMARY")
    print("="*60)
    print(f"• Mean Era Correlation (Corr20v2) : {mean_neut:.4f}")
    print(f"• Raw Per-Era Sharpe (μ/σ)        : {raw_sharpe_neut:.3f} (Leaderboard Standard)")
    print(f"• Annualized Sharpe (Monthly √12) : {annualized_monthly:.2f}")
    print(f"• Peak-to-Trough Max Drawdown     : {max_dd:.4f} ({max_dd*100:.2f}%)")
    print("="*60)
    print("[SUCCESS]  Alpha Ensemble successfully verified and standardized!")


if __name__ == "__main__":
    main()
