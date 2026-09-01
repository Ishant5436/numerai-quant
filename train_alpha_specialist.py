#!/usr/bin/env python3
"""
Strategy 3: Quality Momentum & Residual Alpha Specialist
- Trains on orthogonal factor horizons:
  'target_jeremy_20' (Quality Momentum) and 'target_agnes_20' (Residual factor returns)
- Applies 35% linear feature neutralization to isolate pure idiosyncratic alpha
- Saved to /Users/ishantpanchal/numerai-quant/models/alpha_specialist/
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

ALPHA_MODEL_DIR = "/Users/ishantpanchal/numerai-quant/models/alpha_specialist"
os.makedirs(ALPHA_MODEL_DIR, exist_ok=True)

ALPHA_TARGETS = [
    "target_jeremy_20",  # Value / Quality momentum returns
    "target_agnes_20"    # Orthogonal residual factor returns
]
ALPHA_NEUTRALIZATION = 0.35  # 35% balanced feature neutralization


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
    print("=" * 65)
    print("[EXECUTE]  TRAINING NUMERAI STRATEGY 3: QUALITY MOMENTUM & RESIDUAL ALPHA")
    print("=" * 65)
    print(f"Features: {len(features)} medium features | Targets: {ALPHA_TARGETS}")
    print(f"Neutralization Proportion: {ALPHA_NEUTRALIZATION * 100:.0f}%")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    cols_to_load = ["era"] + ALPHA_TARGETS + features
    print("\n[LOAD]  Loading training dataset...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    X_train = train_df[features]

    models = {}
    for target in ALPHA_TARGETS:
        model_file = os.path.join(ALPHA_MODEL_DIR, f"lgb_{target}.pkl")
        print(f"\n[TRAIN]  Training LightGBM Booster on '{target}'...")
        y_train = train_df[target]
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)
        models[target] = model
        print(f"[SAVE]  Saved: {model_file}")

    del train_df, X_train

    # Validation Phase
    val_cols = ["era", "target"] + features
    print("\n[METRICS]  Evaluating out-of-sample on validation eras...")
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
    val_df["alpha_raw"] = val_df[pred_cols].mean(axis=1)
    val_df["alpha_raw"] = val_df.groupby("era", observed=True)["alpha_raw"].rank(pct=True)

    # 35% Feature Neutralization
    print(f"\n[GUARD]  Applying {int(ALPHA_NEUTRALIZATION*100)}% feature neutralization for idiosyncratic alpha...")
    neutralizer_feats = features[:60]
    val_df["alpha_neutralized"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(
            neutralize(era, ["alpha_raw"], extra_neutralizers=neutralizer_feats, proportion=ALPHA_NEUTRALIZATION)["alpha_raw"].values,
            index=era.index
        )
    )

    corrs_neutral = calculate_era_correlation(val_df, "alpha_neutralized", "target")
    mean_neut = corrs_neutral.mean()
    sharpe_neut = mean_neut / (corrs_neutral.std() + 1e-8)
    max_dd = (corrs_neutral.cumsum().cummax() - corrs_neutral.cumsum()).max()

    print("\n" + "=" * 65)
    print("[AUDIT]  STRATEGY 3 OUT-OF-SAMPLE AUDIT SUMMARY")
    print("=" * 65)
    print(f"• Mean Era Correlation (Corr20v2) : +{mean_neut:.4f}")
    print(f"• Raw Per-Era Sharpe (μ/σ)        : {sharpe_neut:.3f}")
    print(f"• Annualized Sharpe (x √12)       : {sharpe_neut * np.sqrt(12):.3f}")
    print(f"• Peak-to-Trough Max Drawdown     : {max_dd:.4f} ({max_dd * 100:.2f}%)")
    print("=" * 65)
    print("[SUCCESS]  Quality Momentum Strategy successfully trained and validated!\n")


if __name__ == "__main__":
    main()
