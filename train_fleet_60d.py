#!/usr/bin/env python3
"""
Numerai 25-Model Fleet 60-Day Alpha Training Engine.
Trains LightGBM models for strategies 2 through 25 on designated 60-day targets.
Adheres strictly to Power of 10 invariants and Numerai v5.0 standards.
"""

import os
import time
import joblib
import lightgbm as lgb
import pandas as pd

from config import (
    DATA_DIR,
    FLEET_STRATEGY_MAP_60D,
    LGB_PARAMS,
    ORTHO_60D_DIR,
)
from fleet_submit import load_feature_groups


def load_train_data(train_path: str, targets: list, features: list, stride: int = 2) -> pd.DataFrame:
    """Load train.parquet with bounded columns and era stride sampling."""
    assert os.path.exists(train_path), f"Train parquet missing: {train_path}"
    assert len(targets) > 0, "Target list cannot be empty"
    assert len(features) > 0, "Feature list cannot be empty"

    cols = ["era"] + targets + features
    print(f"[LOAD] Ingesting training dataset ({len(cols)} columns, stride={stride})...")
    df = pd.read_parquet(train_path, columns=cols)
    assert len(df) > 0, "Read empty train dataset"

    if stride > 1:
        eras = sorted(df["era"].unique())
        selected = eras[::stride]
        df = df[df["era"].isin(selected)].copy()
        print(f"[INFO] Sampled {len(selected)}/{len(eras)} training eras")

    assert len(df) > 0, "Empty dataset after stride sampling"
    return df


def train_and_save_strategy_model(train_df: pd.DataFrame, strat_id: int, target: str, feats: list) -> str:
    """Fit LightGBM on specific target and feature subset, then persist weights."""
    assert target in train_df.columns, f"Target '{target}' missing in training data"
    assert len(feats) > 0, f"Empty features list for strategy {strat_id}"

    out_file = os.path.join(ORTHO_60D_DIR, f"lgb_strat_{strat_id}.pkl")
    t0 = time.time()
    
    valid_df = train_df.dropna(subset=[target])
    assert len(valid_df) > 100, f"Insufficient rows for strategy {strat_id}"

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(valid_df[feats], valid_df[target])
    joblib.dump(model, out_file)
    
    elapsed = time.time() - t0
    assert os.path.exists(out_file), f"Failed to persist {out_file}"
    assert os.path.getsize(out_file) > 10_000, f"Saved model too small: {out_file}"
    print(f"[OK] Strat {strat_id:2d} -> Target: {target:18s} | Feats: {len(feats):3d} | Saved: {elapsed:.1f}s")
    return out_file


def main():
    """Execute complete 25-strategy 60-day training pipeline."""
    print("=========================================================================")
    print("[INIT] NUMERAI 25-MODEL FLEET 60-DAY ARCHITECTURE TRAINING PIPELINE")
    print("=========================================================================")
    os.makedirs(ORTHO_60D_DIR, exist_ok=True)
    groups = load_feature_groups()
    all_medium = groups["all_medium"]
    assert len(all_medium) == 705, f"Unexpected medium features count: {len(all_medium)}"

    # Identify all needed 60d targets for strategies 2..25
    targets_needed = sorted(list(set(
        FLEET_STRATEGY_MAP_60D[s][0] for s in range(2, 26)
    )))
    print(f"[INFO] Unique 60-day targets required: {len(targets_needed)}")
    print("[INFO] Strategies to train: 2 through 25 (Strategy 1 uses 60d quintet)")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    train_df = load_train_data(train_path, targets_needed, all_medium, stride=2)

    for strat_id in range(2, 26):
        target, feat_key, _ = FLEET_STRATEGY_MAP_60D[strat_id]
        feats = groups[feat_key]
        train_and_save_strategy_model(train_df, strat_id, target, feats)

    print("\n=========================================================================")
    print("[SUCCESS] All 24 60-day orthogonal models trained and verified in ORTHO_60D_DIR!")
    print("=========================================================================")


if __name__ == "__main__":
    main()
