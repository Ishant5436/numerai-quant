#!/usr/bin/env python3
"""
Numerai ML Training Engine
- Loads feature metadata
- Downloads train & validation parquet partitions
- Trains LightGBM ranker/regressor
- Calculates out-of-sample era correlation & Sharpe ratio
- Serializes trained model
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from numerapi import NumerAPI
from config import FEATURE_SET, TARGET_COL, MODEL_PATH, FEATURES_JSON, DATA_DIR, LGB_PARAMS

os.makedirs(DATA_DIR, exist_ok=True)
napi = NumerAPI()


def get_feature_list() -> list:
    if not os.path.exists(FEATURES_JSON):
        print("Downloading features.json...")
        napi.download_dataset("v5.0/features.json", FEATURES_JSON)
    
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    return meta["feature_sets"][FEATURE_SET]


def download_if_missing(dataset_name: str, local_path: str):
    if not os.path.exists(local_path):
        print(f"Downloading {dataset_name} to {local_path}...")
        napi.download_dataset(dataset_name, local_path)
    else:
        print(f"Using cached {local_path}")


def calculate_per_era_corr(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    """Calculates Spearman correlation per era."""
    def era_corr(sub_df):
        r, _ = spearmanr(sub_df[pred_col], sub_df[target_col])
        return r
    return df.groupby("era").apply(era_corr)


def main():
    features = get_feature_list()
    print(f"Loaded {len(features)} features for set '{FEATURE_SET}'")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    download_if_missing("v5.0/train.parquet", train_path)
    download_if_missing("v5.0/validation.parquet", val_path)

    # Read selected columns
    cols_to_load = ["era", TARGET_COL] + features
    print("Loading training dataset...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    
    # Filter rows with non-null target
    train_df = train_df.dropna(subset=[TARGET_COL])
    print(f"Train shape: {train_df.shape}")

    X_train = train_df[features]
    y_train = train_df[TARGET_COL]

    print(f"Training LightGBM model on {len(features)} features...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Out-of-sample Validation
    print("Loading validation dataset...")
    val_df = pd.read_parquet(val_path, columns=cols_to_load)
    val_df = val_df.dropna(subset=[TARGET_COL])
    print(f"Validation shape: {val_df.shape}")

    print("Evaluating out-of-sample era correlation...")
    val_df["prediction"] = model.predict(val_df[features])
    
    # Rank predictions per era (0 to 1)
    val_df["prediction"] = val_df.groupby("era")["prediction"].rank(pct=True)

    era_corrs = calculate_per_era_corr(val_df, "prediction", TARGET_COL)
    mean_corr = era_corrs.mean()
    std_corr = era_corrs.std()
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0
    max_drawdown = (era_corrs.cumsum().cummax() - era_corrs.cumsum()).max()

    print("\n--- Validation Metrics ---")
    print(f"Mean Era Correlation (Corr20v2): {mean_corr:.4f}")
    print(f"Sharpe Ratio:                   {sharpe:.4f}")
    print(f"Std Dev:                        {std_corr:.4f}")
    print(f"Max Drawdown:                   {max_drawdown:.4f}")


if __name__ == "__main__":
    main()
