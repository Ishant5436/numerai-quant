#!/usr/bin/env python3
"""
Numerai Fleet Scaling Research Engine: Strategies 6 through 10
Expands tournament model coverage from 5 to 10 orthogonal strategies:
• Strategy 6: Cyrusd-20 Deep Non-Linear Horizon (705 features, 30% Neutralized)
• Strategy 7: Low-Volatility Quality Defensive (104 features, 35% Neutralized)
• Strategy 8: High-Beta Trend Velocity (242 features, 40% Neutralized)
• Strategy 9: Capital Efficiency & Value (304 features, 35% Neutralized)
• Strategy 10: Macro Regime Tail Shield (156 features, 45% Neutralized)
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
        "quality_defensive": get_subset(["serenity", "wisdom", "intelligence"]),   # 104 features
        "trend_velocity": get_subset(["agility", "strength", "sunshine"]),          # 242 features
        "value_capital": get_subset(["charisma", "wisdom", "constitution"]),        # 304 features
        "macro_tail": get_subset(["midnight", "rain", "serenity"])                  # 156 features
    }
    return groups


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    groups = load_expansion_feature_groups()
    print("=" * 80)
    print("[RESEARCH]  NUMERAI FLEET EXPANSION ENGINE: STRATEGIES 6 - 10")
    print("=" * 80)
    print(f"Feature Partitions:")
    print(f"  • Strategy 6  (Deep Horizon)     : {len(groups['all_medium'])} features (All Medium)")
    print(f"  • Strategy 7  (Defensive Quality): {len(groups['quality_defensive'])} features (serenity, wisdom, intelligence)")
    print(f"  • Strategy 8  (Trend Velocity)   : {len(groups['trend_velocity'])} features (agility, strength, sunshine)")
    print(f"  • Strategy 9  (Capital Value)    : {len(groups['value_capital'])} features (charisma, wisdom, constitution)")
    print(f"  • Strategy 10 (Macro Tail)       : {len(groups['macro_tail'])} features (midnight, rain, serenity)")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    targets = [
        "target_cyrusd_20",
        "target_agnes_20",
        "target_victor_20",
        "target_caroline_20",
        "target_sam_20"
    ]
    cols_to_load = ["era"] + targets + groups["all_medium"]

    print("\n[LOAD]  Loading training dataset (2.74M rows, selected columns)...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    print("[LOAD]  Training dataset loaded into memory successfully.")

    configs = [
        {
            "id": 6,
            "name": "Cyrusd-20 Deep Horizon",
            "target": "target_cyrusd_20",
            "features": groups["all_medium"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 48, "max_depth": 6, "colsample_bytree": 0.1, "n_jobs": -1, "random_state": 606},
            "neut": 0.30,
            "badge": "v5.0 Deep Horizon"
        },
        {
            "id": 7,
            "name": "Low-Volatility Quality Specialist",
            "target": "target_agnes_20",
            "features": groups["quality_defensive"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 20, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 707},
            "neut": 0.35,
            "badge": "Defensive Quality"
        },
        {
            "id": 8,
            "name": "High-Beta Trend Velocity Specialist",
            "target": "target_victor_20",
            "features": groups["trend_velocity"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 808},
            "neut": 0.40,
            "badge": "Trend Velocity"
        },
        {
            "id": 9,
            "name": "Capital Efficiency & Value Specialist",
            "target": "target_caroline_20",
            "features": groups["value_capital"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 909},
            "neut": 0.35,
            "badge": "Capital Efficiency"
        },
        {
            "id": 10,
            "name": "Macro Regime Tail Shield Specialist",
            "target": "target_sam_20",
            "features": groups["macro_tail"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 32, "max_depth": 5, "colsample_bytree": 0.15, "n_jobs": -1, "random_state": 1010},
            "neut": 0.45,
            "badge": "Macro Regime Shield"
        }
    ]

    trained_models = {}
    for cfg in configs:
        sid = cfg["id"]
        model_file = os.path.join(ORTHO_DIR, f"lgb_strat_{sid}.pkl")
        print(f"\n[TRAIN]  Strategy {sid}: {cfg['name']} ({cfg['badge']})...")
        print(f"         Target: {cfg['target']} | Features: {len(cfg['features'])} | Neut: {cfg['neut']*100:.0f}%")

        X_train = train_df[cfg["features"]]
        y_train = train_df[cfg["target"]]

        model = lgb.LGBMRegressor(**cfg["params"])
        model.fit(X_train, y_train)

        joblib.dump(model, model_file)
        trained_models[sid] = model
        print(f"[SAVE]   Saved model artifact -> {model_file}")

    del train_df

    # Out-of-sample validation evaluation
    print("\n" + "=" * 80)
    print("[VALIDATE]  EVALUATING OUT-OF-SAMPLE BENCHMARKS ON VALIDATION ERAS...")
    print("=" * 80)
    val_cols = ["era", "target"] + groups["all_medium"]
    val_df = pd.read_parquet(val_path, columns=val_cols)
    print(f"Validation dataset loaded: {len(val_df):,} rows across {val_df['era'].nunique()} eras.")

    neutralizer_feats = groups["all_medium"][:60]
    results = []

    for cfg in configs:
        sid = cfg["id"]
        model = trained_models[sid]
        print(f"[EVAL]   Scoring Strategy {sid} ({cfg['name']})...")

        raw_pred = model.predict(val_df[cfg["features"]])
        val_df[f"pred_raw_{sid}"] = val_df.groupby("era", observed=True)[f"temp_{sid}"].rank(pct=True) if f"temp_{sid}" in val_df else pd.Series(raw_pred, index=val_df.index).groupby(val_df["era"], observed=True).rank(pct=True)

        val_df[f"neut_{sid}"] = val_df.groupby("era", observed=True, group_keys=False).apply(
            lambda era: pd.Series(
                neutralize(era, [f"pred_raw_{sid}"], extra_neutralizers=neutralizer_feats, proportion=cfg["neut"])[f"pred_raw_{sid}"].values,
                index=era.index
            )
        )

        corr_series = calculate_era_correlation(val_df, f"neut_{sid}", "target")
        mean_corr = corr_series.mean()
        sharpe = mean_corr / (corr_series.std() + 1e-8)
        ann_sharpe = sharpe * np.sqrt(52)

        # Drawdown calculation
        cum_returns = corr_series.cumsum()
        peak = cum_returns.cummax()
        drawdown = peak - cum_returns
        max_dd = drawdown.max()

        results.append({
            "Strategy": f"Strategy {sid}",
            "Name": cfg["name"],
            "Target": cfg["target"],
            "Mean Corr": f"{mean_corr:+.4f}",
            "Per-Era Sharpe": f"{sharpe:.3f}",
            "Annualized Sharpe": f"{ann_sharpe:.3f}",
            "Max Drawdown": f"{max_dd:.3f}"
        })

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("[SUMMARY]  STRATEGIES 6 - 10 OUT-OF-SAMPLE BENCHMARK RESULTS")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)
    print("\n[SUCCESS]  All 5 Expansion Models Trained & Saved in models/orthogonal_fleet/!")


if __name__ == "__main__":
    main()
