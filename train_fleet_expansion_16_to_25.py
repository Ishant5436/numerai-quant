#!/usr/bin/env python3
"""
Numerai Fleet Scaling Research Engine: Strategies 16 through 25
Expands tournament model coverage from 15 to 25 orthogonal strategies (Full Account Capacity):
• Strategy 16: Fundamental Value Spread (Charisma + Wisdom, 35% Neutralized)
• Strategy 17: Low-Beta Defensive (Serenity + Constitution, 40% Neutralized)
• Strategy 18: Pure Residual Alpha (Constitution + Dexterity, 50% Neutralized)
• Strategy 19: Mean Reversion Dynamic (Dexterity + Agility, 30% Neutralized)
• Strategy 20: Factor Momentum Velocity (Strength + Agility, 35% Neutralized)
• Strategy 21: High-Sharpe Quality (Intelligence + Wisdom, 30% Neutralized)
• Strategy 22: Macro Tail Liquidity (Midnight + Rain, 45% Neutralized)
• Strategy 23: Earnings Quality Momentum (Charisma + Sunshine, 35% Neutralized)
• Strategy 24: Sentiment Divergence (Midnight + Serenity, 40% Neutralized)
• Strategy 25: Volatility-Adjusted Alpha (Serenity + Strength, 35% Neutralized)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_JSON, DATA_DIR
from neutralize import neutralize, rank_01

ORTHO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "orthogonal_fleet")
os.makedirs(ORTHO_DIR, exist_ok=True)


def load_expansion_feature_groups() -> dict:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    medium_set = set(meta["feature_sets"]["medium"])

    def get_subset(keys):
        c = set()
        for k in keys:
            c.update(meta["feature_sets"].get(k, []))
        return sorted(list(c.intersection(medium_set)))

    groups = {
        "all_medium": meta["feature_sets"]["medium"],
        "fundamental_value": get_subset(["charisma", "wisdom"]),
        "low_beta_defensive": get_subset(["serenity", "constitution"]),
        "residual_alpha": get_subset(["constitution", "dexterity"]),
        "mean_reversion": get_subset(["dexterity", "agility"]),
        "factor_momentum": get_subset(["strength", "agility"]),
        "high_sharpe_quality": get_subset(["intelligence", "wisdom"]),
        "macro_tail_liquidity": get_subset(["midnight", "rain"]),
        "earnings_quality": get_subset(["charisma", "sunshine"]),
        "sentiment_divergence": get_subset(["midnight", "serenity"]),
        "vol_adjusted_alpha": get_subset(["serenity", "strength"]),
    }
    return groups


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    groups = load_expansion_feature_groups()
    print("=" * 80)
    print("[RESEARCH] NUMERAI FLEET EXPANSION ENGINE: STRATEGIES 16 - 25")
    print("=" * 80)
    for g_name, g_feats in groups.items():
        if g_name != "all_medium":
            print(f"  • {g_name:24s}: {len(g_feats)} features")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    targets = [
        "target_bravo_20",
        "target_charlie_20",
        "target_delta_20",
        "target_echo_20",
        "target_ralph_20",
        "target_rowan_20",
        "target_sam_20",
        "target_tyler_20",
        "target_waldo_20",
        "target_victor_20",
    ]
    cols_to_load = ["era"] + targets + groups["all_medium"]

    print("\n[LOAD] Loading training dataset (selected columns)...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    print(f"[LOAD] Training dataset loaded successfully ({len(train_df):,} rows).")

    configs = [
        {
            "id": 16,
            "name": "Fundamental Value Spread",
            "target": "target_bravo_20",
            "features": groups["fundamental_value"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1616},
            "neut": 0.35,
            "badge": "Value Spread"
        },
        {
            "id": 17,
            "name": "Low-Beta Defensive",
            "target": "target_charlie_20",
            "features": groups["low_beta_defensive"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1717},
            "neut": 0.40,
            "badge": "Low-Beta Defense"
        },
        {
            "id": 18,
            "name": "Pure Residual Alpha",
            "target": "target_delta_20",
            "features": groups["residual_alpha"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1818},
            "neut": 0.50,
            "badge": "Residual Alpha"
        },
        {
            "id": 19,
            "name": "Mean Reversion Dynamic",
            "target": "target_echo_20",
            "features": groups["mean_reversion"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1919},
            "neut": 0.30,
            "badge": "Mean Reversion"
        },
        {
            "id": 20,
            "name": "Factor Momentum Velocity",
            "target": "target_ralph_20",
            "features": groups["factor_momentum"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 2020},
            "neut": 0.35,
            "badge": "Factor Momentum"
        },
        {
            "id": 21,
            "name": "High-Sharpe Quality",
            "target": "target_rowan_20",
            "features": groups["high_sharpe_quality"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 2121},
            "neut": 0.30,
            "badge": "Sharpe Quality"
        },
        {
            "id": 22,
            "name": "Macro Tail Liquidity",
            "target": "target_sam_20",
            "features": groups["macro_tail_liquidity"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 32, "max_depth": 5, "colsample_bytree": 0.15, "n_jobs": -1, "random_state": 2222},
            "neut": 0.45,
            "badge": "Tail Liquidity"
        },
        {
            "id": 23,
            "name": "Earnings Quality Momentum",
            "target": "target_tyler_20",
            "features": groups["earnings_quality"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 2323},
            "neut": 0.35,
            "badge": "Earnings Quality"
        },
        {
            "id": 24,
            "name": "Sentiment Divergence",
            "target": "target_waldo_20",
            "features": groups["sentiment_divergence"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 2424},
            "neut": 0.40,
            "badge": "Sentiment Divergence"
        },
        {
            "id": 25,
            "name": "Volatility-Adjusted Alpha",
            "target": "target_victor_20",
            "features": groups["vol_adjusted_alpha"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 2525},
            "neut": 0.35,
            "badge": "Vol-Adjusted Alpha"
        },
    ]

    trained_models = {}
    for cfg in configs:
        sid = cfg["id"]
        model_file = os.path.join(ORTHO_DIR, f"lgb_strat_{sid}.pkl")
        print(f"\n[TRAIN] Strategy {sid}: {cfg['name']} ({cfg['badge']})...")
        print(f"        Target: {cfg['target']} | Features: {len(cfg['features'])} | Neut: {cfg['neut']*100:.0f}%")

        X_train = train_df[cfg["features"]]
        y_train = train_df[cfg["target"]]

        model = lgb.LGBMRegressor(**cfg["params"])
        model.fit(X_train, y_train)

        joblib.dump(model, model_file)
        trained_models[sid] = model
        print(f"[SAVE]  Saved model artifact -> {model_file}")

    del train_df
    print("\n[COMPLETE] All 10 model weights (Strategies 16-25) trained and serialized successfully.")


if __name__ == "__main__":
    main()
