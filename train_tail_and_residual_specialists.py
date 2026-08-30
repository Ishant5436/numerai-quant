#!/usr/bin/env python3
"""
Train Strategies 4 & 5:
- Strategy 4: Pure Tail-Risk Specialist (target_xerxes_20) with 40% Neutralization
- Strategy 5: Pure Residual Horizon Specialist (target_delta_20) with 40% Neutralization
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

TAIL_DIR = "/Users/ishantpanchal/numerai-quant/models/tail_specialist"
RESIDUAL_DIR = "/Users/ishantpanchal/numerai-quant/models/residual_specialist"
os.makedirs(TAIL_DIR, exist_ok=True)
os.makedirs(RESIDUAL_DIR, exist_ok=True)


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
    print("=" * 70)
    print("🚀 TRAINING STRATEGIES 4 & 5: TAIL-RISK & RESIDUAL ORTHOGONAL SPECIALISTS")
    print("=" * 70)

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    cols_to_load = ["era", "target_xerxes_20", "target_delta_20"] + features
    print("\n📥 Loading training dataset (2.74M rows)...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    X_train = train_df[features]

    # 1. Strategy 4: target_xerxes_20
    xerxes_file = os.path.join(TAIL_DIR, "lgb_target_xerxes_20.pkl")
    print("\n🧠 Training Strategy 4: Tail-Risk Specialist ('target_xerxes_20')...")
    model_xerxes = lgb.LGBMRegressor(**LGB_PARAMS)
    model_xerxes.fit(X_train, train_df["target_xerxes_20"])
    joblib.dump(model_xerxes, xerxes_file)
    print(f"💾 Saved: {xerxes_file}")

    # 2. Strategy 5: target_delta_20
    delta_file = os.path.join(RESIDUAL_DIR, "lgb_target_delta_20.pkl")
    print("\n🧠 Training Strategy 5: Residual Specialist ('target_delta_20')...")
    model_delta = lgb.LGBMRegressor(**LGB_PARAMS)
    model_delta.fit(X_train, train_df["target_delta_20"])
    joblib.dump(model_delta, delta_file)
    print(f"💾 Saved: {delta_file}")

    del train_df, X_train

    # Validation Phase
    print("\n📈 Evaluating both strategies out-of-sample on validation eras...")
    val_cols = ["era", "target"] + features
    val_df = pd.read_parquet(val_path, columns=val_cols)
    X_val = val_df[features]

    # Validate Strategy 4
    val_df["temp_xerxes"] = model_xerxes.predict(X_val)
    val_df["pred_xerxes"] = val_df.groupby("era", observed=True)["temp_xerxes"].rank(pct=True)
    neutralizer_feats = features[:60]
    val_df["neut_xerxes"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(
            neutralize(era, ["pred_xerxes"], extra_neutralizers=neutralizer_feats, proportion=0.40)["pred_xerxes"].values,
            index=era.index
        )
    )
    corr_xerxes = calculate_era_correlation(val_df, "neut_xerxes", "target")
    mean_xerxes = corr_xerxes.mean()
    sharpe_xerxes = mean_xerxes / (corr_xerxes.std() + 1e-8)

    # Validate Strategy 5
    val_df["temp_delta"] = model_delta.predict(X_val)
    val_df["pred_delta"] = val_df.groupby("era", observed=True)["temp_delta"].rank(pct=True)
    val_df["neut_delta"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(
            neutralize(era, ["pred_delta"], extra_neutralizers=neutralizer_feats, proportion=0.40)["pred_delta"].values,
            index=era.index
        )
    )
    corr_delta = calculate_era_correlation(val_df, "neut_delta", "target")
    mean_delta = corr_delta.mean()
    sharpe_delta = mean_delta / (corr_delta.std() + 1e-8)

    print("\n" + "=" * 70)
    print("📊 STRATEGIES 4 & 5 AUDIT SUMMARY")
    print("=" * 70)
    print(f"• Strategy 4 (Tail-Risk xerxes_20) : Mean Corr = +{mean_xerxes:.4f} | Per-Era Sharpe = {sharpe_xerxes:.3f}")
    print(f"• Strategy 5 (Residual delta_20)   : Mean Corr = +{mean_delta:.4f} | Per-Era Sharpe = {sharpe_delta:.3f}")
    print("=" * 70)
    print("✅ All 5 Fleet Strategy Archetypes Trained Successfully!\n")


if __name__ == "__main__":
    main()
