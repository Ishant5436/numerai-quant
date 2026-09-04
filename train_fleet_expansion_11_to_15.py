#!/usr/bin/env python3
"""
Numerai Fleet Scaling Research Engine: Strategies 11 through 15
Expands tournament model coverage from 10 to 15 orthogonal strategies:
• Strategy 11: High-Conviction Alpha Specialist (124 features, 30% Neutralized)
• Strategy 12: Volatility-Defensive Specialist (168 features, 40% Neutralized)
• Strategy 13: Risk-Parity Claudia Specialist (163 features, 35% Neutralized)
• Strategy 14: Deep Multi-Horizon Specialist (705 features, 25% Neutralized)
• Strategy 15: Macro-Hedged Xerxes Specialist (180 features, 50% Neutralized)
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
        "alpha_conviction": get_subset(["intelligence", "strength", "wisdom"]),      # 124 features
        "volatility_defensive": get_subset(["serenity", "constitution", "rain"]),     # 168 features
        "risk_parity": get_subset(["sunshine", "intelligence", "dexterity"]),         # 163 features
        "macro_hedged": get_subset(["midnight", "agility", "rain"])                   # 180 features
    }
    return groups


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    groups = load_expansion_feature_groups()
    print("=" * 80)
    print("[RESEARCH]  NUMERAI FLEET EXPANSION ENGINE: STRATEGIES 11 - 15")
    print("=" * 80)
    print("Feature Partitions:")
    print(f"  • Strategy 11 (Alpha Conviction)    : {len(groups['alpha_conviction'])} features (intelligence, strength, wisdom)")
    print(f"  • Strategy 12 (Volatility Defense)  : {len(groups['volatility_defensive'])} features (serenity, constitution, rain)")
    print(f"  • Strategy 13 (Risk Parity)         : {len(groups['risk_parity'])} features (sunshine, intelligence, dexterity)")
    print(f"  • Strategy 14 (Deep Horizon)        : {len(groups['all_medium'])} features (All Medium)")
    print(f"  • Strategy 15 (Macro Hedged)        : {len(groups['macro_hedged'])} features (midnight, agility, rain)")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    targets = [
        "target_alpha_20",
        "target_teager2b_20",
        "target_claudia_20",
        "target_cyrusd_20",
        "target_xerxes_20"
    ]
    cols_to_load = ["era"] + targets + groups["all_medium"]

    print("\n[LOAD]  Loading training dataset (2.74M rows, selected columns)...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    print("[LOAD]  Training dataset loaded into memory successfully.")

    configs = [
        {
            "id": 11,
            "name": "High-Conviction Alpha Specialist",
            "target": "target_alpha_20",
            "features": groups["alpha_conviction"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1111},
            "neut": 0.30,
            "badge": "Alpha Conviction"
        },
        {
            "id": 12,
            "name": "Volatility-Defensive Specialist",
            "target": "target_teager2b_20",
            "features": groups["volatility_defensive"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1212},
            "neut": 0.40,
            "badge": "Defensive Volatility"
        },
        {
            "id": 13,
            "name": "Risk-Parity Claudia Specialist",
            "target": "target_claudia_20",
            "features": groups["risk_parity"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 28, "max_depth": 5, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 1313},
            "neut": 0.35,
            "badge": "Risk Parity"
        },
        {
            "id": 14,
            "name": "Deep Multi-Horizon Specialist",
            "target": "target_cyrusd_20",
            "features": groups["all_medium"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 40, "max_depth": 6, "colsample_bytree": 0.15, "n_jobs": -1, "random_state": 1414},
            "neut": 0.25,
            "badge": "Multi-Horizon Deep"
        },
        {
            "id": 15,
            "name": "Macro-Hedged Xerxes Specialist",
            "target": "target_xerxes_20",
            "features": groups["macro_hedged"],
            "params": {"n_estimators": 400, "learning_rate": 0.02, "num_leaves": 32, "max_depth": 5, "colsample_bytree": 0.15, "n_jobs": -1, "random_state": 1515},
            "neut": 0.50,
            "badge": "Macro Hedged Shield"
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
        val_df[f"pred_raw_{sid}"] = pd.Series(raw_pred, index=val_df.index).groupby(val_df["era"], observed=True).rank(pct=True)

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
    print("[SUMMARY]  STRATEGIES 11 - 15 OUT-OF-SAMPLE BENCHMARK RESULTS")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)
    print("\n[SUCCESS]  All 5 Expansion Models Trained & Saved in models/orthogonal_fleet/!")


if __name__ == "__main__":
    main()
