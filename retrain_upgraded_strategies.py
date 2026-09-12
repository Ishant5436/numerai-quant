#!/usr/bin/env python3
"""
Retrain Upgraded 60-Day Specialist Strategies (5, 11, 13, 18, 21)
Upgrades weak baseline targets to high-Sharpe v5 benchmark targets with optimal 35% neutralization.
Adheres strictly to Power of 10 safety invariants.
"""

import os
import sys
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


def load_train_slice(train_path: str, targets: list, features: list, stride: int = 2) -> pd.DataFrame:
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


def train_single_strategy(train_df: pd.DataFrame, strat_id: int, target: str, feats: list) -> str:
    """Train LightGBM on specific target and feature subset, then persist weights."""
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
    strats_to_upgrade = [5, 11, 13, 18, 21]
    print("=========================================================================")
    print(f"[INIT] Retraining {len(strats_to_upgrade)} Upgraded Specialist Strategies for 95th+ Percentile")
    print("=========================================================================")

    groups = load_feature_groups()
    targets_needed = sorted(list(set(FLEET_STRATEGY_MAP_60D[s][0] for s in strats_to_upgrade)))
    features_needed = set()
    for s in strats_to_upgrade:
        _, feat_key, _ = FLEET_STRATEGY_MAP_60D[s]
        features_needed.update(groups[feat_key])

    features_needed = sorted(list(features_needed))
    print(f"[INFO] Targets: {targets_needed}")
    print(f"[INFO] Unique features needed: {len(features_needed)}")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    train_df = load_train_slice(train_path, targets_needed, features_needed, stride=2)

    for strat_id in strats_to_upgrade:
        target, feat_key, _ = FLEET_STRATEGY_MAP_60D[strat_id]
        feats = groups[feat_key]
        train_single_strategy(train_df, strat_id, target, feats)

    print("\n=========================================================================")
    print(f"[SUCCESS] All {len(strats_to_upgrade)} upgraded models trained and saved to {ORTHO_60D_DIR}!")
    print("=========================================================================")


if __name__ == "__main__":
    main()
