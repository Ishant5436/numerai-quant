#!/usr/bin/env python3
"""
Strategy 2: High-MMC Volatility & Orthogonal Alpha Specialist
- Trains on pure volatility-adjusted and residual targets:
  'target_victor_20', 'target_xerxes_20', 'target_delta_20'
- Uses 50% aggressive feature neutralization for maximum Meta-Model Contribution (MMC)
- Saved to /Users/ishantpanchal/numerai-quant/models/vol_specialist/
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
    FEATURES_JSON,
    DATA_DIR,
    LGB_PARAMS
)
from neutralize import neutralize, rank_01

VOL_MODEL_DIR = "/Users/ishantpanchal/numerai-quant/models/vol_specialist"
os.makedirs(VOL_MODEL_DIR, exist_ok=True)

VOL_TARGETS = [
    "target_victor_20",  # Volatility-adjusted returns
    "target_xerxes_20",  # Tail risk return target
    "target_delta_20"    # Uncorrelated residual target
]
VOL_NEUTRALIZATION = 0.50  # 50% heavy neutralization to strip market beta


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
    print(f"=== Numerai Volatility & MMC Specialist Pipeline ===")
    print(f"Features: {len(features)} | Targets: {VOL_TARGETS}")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    cols_to_load = ["era"] + VOL_TARGETS + features
    print("\nLoading training dataset...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    X_train = train_df[features]

    models = {}
    for target in VOL_TARGETS:
        model_file = os.path.join(VOL_MODEL_DIR, f"lgb_{target}.pkl")
        print(f"\n---> Training Vol-Specialist on '{target}'...")
        y_train = train_df[target]
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)
        models[target] = model
        print(f"Saved: {model_file}")

    del train_df, X_train

    # Validation
    val_cols = ["era", "target"] + features
    print("\nEvaluating out-of-sample on validation eras...")
    val_df = pd.read_parquet(val_path, columns=val_cols)
    X_val = val_df[features]

    pred_cols = []
    for target, model in models.items():
        pred_col = f"pred_{target}"
        pred_cols.append(pred_col)
        raw_preds = model.predict(X_val)
        val_df["temp"] = raw_preds
        val_df[pred_col] = val_df.groupby("era", observed=True)["temp"].rank(pct=True)

    val_df.drop(columns=["temp"], inplace=True)
    val_df["vol_raw"] = val_df[pred_cols].mean(axis=1)
    val_df["vol_raw"] = val_df.groupby("era", observed=True)["vol_raw"].rank(pct=True)

    # 50% Heavy Feature Neutralization
    print(f"Applying {int(VOL_NEUTRALIZATION*100)}% heavy feature neutralization for high MMC...")
    neutralizer_feats = features[:60]
    val_df["vol_neutralized"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(
            neutralize(era, ["vol_raw"], extra_neutralizers=neutralizer_feats, proportion=VOL_NEUTRALIZATION)["vol_raw"].values,
            index=era.index
        )
    )

    corrs_neutral = calculate_era_correlation(val_df, "vol_neutralized", "target")
    mean_neut = corrs_neutral.mean()
    sharpe_neut = mean_neut / (corrs_neutral.std() + 1e-8)
    max_dd = (corrs_neutral.cumsum().cummax() - corrs_neutral.cumsum()).max()

    print("\n" + "="*60)
    print(f"[GUARD]   VOLATILITY / MMC SPECIALIST AUDIT SUMMARY")
    print("="*60)
    print(f"• Mean Era Correlation (Corr20v2) : {mean_neut:.4f}")
    print(f"• Raw Per-Era Sharpe (μ/σ)        : {sharpe_neut:.3f}")
    print(f"• Peak-to-Trough Max Drawdown     : {max_dd:.4f} ({max_dd*100:.2f}%)")
    print("="*60)
    print("[SUCCESS]  Volatility Specialist successfully trained!")


if __name__ == "__main__":
    main()
