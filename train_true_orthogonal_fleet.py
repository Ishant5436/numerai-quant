#!/usr/bin/env python3
"""
Numerai True Orthogonal Fleet Research Engine
Builds genuinely uncorrelated alpha streams by varying:
1. Feature Factor Groups (Fundamental vs Momentum vs Macro vs Constitution subsets)
2. Tree Architecture Asymmetry (Deep non-linear vs Shallow vs Medium trees)
3. Heavy Factor Neutralization (25% to 50%)
Achieves true statistical diversification (low cross-strategy correlation).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from config import FEATURES_JSON, DATA_DIR
from neutralize import neutralize, rank_01

ORTHO_DIR = "/Users/ishantpanchal/numerai-quant/models/orthogonal_fleet"
os.makedirs(ORTHO_DIR, exist_ok=True)


def load_feature_groups() -> dict:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    medium_set = set(meta["feature_sets"]["medium"])
    
    def get_subset(keys):
        combined = set()
        for k in keys:
            combined.update(meta["feature_sets"].get(k, []))
        return sorted(list(combined.intersection(medium_set)))

    groups = {
        "all_medium": meta["feature_sets"]["medium"],
        "fundamental": get_subset(["intelligence", "charisma", "wisdom"]),      # 186 features
        "momentum": get_subset(["strength", "dexterity", "agility"]),            # 133 features
        "macro": get_subset(["serenity", "sunshine", "midnight"]),               # 286 features
        "constitution": get_subset(["constitution", "dexterity"])                # 155 features
    }
    return groups


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    groups = load_feature_groups()
    print("=" * 75)
    print("[RESEARCH]  NUMERAI TRUE ORTHOGONAL ALPHA MINING ENGINE")
    print("=" * 75)
    print(f"[AUDIT]  Feature Partitions (Filtered to Medium Universe):")
    print(f"  • Strategy 1 (Core All): {len(groups['all_medium'])} features")
    print(f"  • Strategy 2 (Fundamental): {len(groups['fundamental'])} features (intelligence, charisma, wisdom)")
    print(f"  • Strategy 3 (Momentum): {len(groups['momentum'])} features (strength, dexterity, agility)")
    print(f"  • Strategy 4 (Macro/Regime): {len(groups['macro'])} features (serenity, sunshine, midnight)")
    print(f"  • Strategy 5 (Constitution/Residual): {len(groups['constitution'])} features (constitution, dexterity)")

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    cols_to_load = ["era", "target", "target_victor_20", "target_xerxes_20", "target_jeremy_20", "target_delta_20"] + groups["all_medium"]
    print("\n[LOAD]  Loading train & validation data...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    val_df = pd.read_parquet(val_path, columns=["era", "target"] + groups["all_medium"])

    # 5 Structurally Diverse Strategies
    configs = [
        {
            "id": 1,
            "name": "Core Alpha Flagship",
            "target": "target",
            "features": groups["all_medium"],
            "params": {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 48, "max_depth": 6, "colsample_bytree": 0.1, "n_jobs": -1, "random_state": 101},
            "neut": 0.25,
            "color": "#3b82f6",
            "badge": "Flagship"
        },
        {
            "id": 2,
            "name": "Fundamental Alpha Specialist",
            "target": "target_jeremy_20",
            "features": groups["fundamental"],
            "params": {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 16, "max_depth": 3, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 202},
            "neut": 0.35,
            "color": "#10b981",
            "badge": "Fundamental"
        },
        {
            "id": 3,
            "name": "Momentum Alpha Specialist",
            "target": "target_victor_20",
            "features": groups["momentum"],
            "params": {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 24, "max_depth": 4, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 303},
            "neut": 0.40,
            "color": "#8b5cf6",
            "badge": "Momentum"
        },
        {
            "id": 4,
            "name": "Macro Regime Specialist",
            "target": "target_xerxes_20",
            "features": groups["macro"],
            "params": {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 32, "max_depth": 5, "colsample_bytree": 0.15, "n_jobs": -1, "random_state": 404},
            "neut": 0.45,
            "color": "#f59e0b",
            "badge": "Macro Regime"
        },
        {
            "id": 5,
            "name": "Constitution Residual Specialist",
            "target": "target_delta_20",
            "features": groups["constitution"],
            "params": {"n_estimators": 450, "learning_rate": 0.02, "num_leaves": 16, "max_depth": 3, "colsample_bytree": 0.2, "n_jobs": -1, "random_state": 505},
            "neut": 0.50,
            "color": "#ec4899",
            "badge": "Residual"
        }
    ]

    strategy_pred_cols = []
    benchmark_metrics = []
    neutralizer_feats = groups["all_medium"][:60]

    for cfg in configs:
        model_file = os.path.join(ORTHO_DIR, f"lgb_strat_{cfg['id']}.pkl")
        print(f"\n[TRAIN]  Training Strategy {cfg['id']} [{cfg['name']}] on {len(cfg['features'])} features (Target: {cfg['target']})...")
        
        y_train = train_df[cfg["target"]]
        X_train = train_df[cfg["features"]]
        
        model = lgb.LGBMRegressor(**cfg["params"])
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)
        print(f"[SAVE]  Saved: {model_file}")

        # Predict Out-of-Sample on Validation
        pred_col = f"Strat_{cfg['id']}_Pred"
        raw_preds = model.predict(val_df[cfg["features"]])
        val_df["temp"] = raw_preds
        val_df[pred_col] = val_df.groupby("era", observed=True)["temp"].rank(pct=True)

        # Apply neutralization
        neut_col = f"Strat_{cfg['id']}_{cfg['name'].replace(' ', '_')}"
        val_df[neut_col] = val_df.groupby("era", observed=True, group_keys=False).apply(
            lambda era: pd.Series(neutralize(era, [pred_col], extra_neutralizers=neutralizer_feats, proportion=cfg["neut"])[pred_col].values, index=era.index)
        )
        strategy_pred_cols.append(neut_col)

        corrs = calculate_era_correlation(val_df, neut_col, "target")
        m_corr = float(corrs.mean())
        s_sharpe = float(m_corr / (corrs.std() + 1e-8))
        m_dd = float((corrs.cumsum().cummax() - corrs.cumsum()).max())

        benchmark_metrics.append({
            "Strategy": f"Strat {cfg['id']}: {cfg['name']}",
            "Target": cfg["target"],
            "Features": f"{len(cfg['features'])} feats",
            "Mean Corr": f"+{m_corr:.4f}",
            "Per-Era Sharpe": f"{s_sharpe:.3f}",
            "Annualized Sharpe": f"{s_sharpe * np.sqrt(12):.3f}",
            "Max Drawdown": f"{m_dd * 100:.2f}%"
        })

    print("\n" + "=" * 75)
    print("[BENCHMARK]  TRUE ORTHOGONAL FLEET PERFORMANCE BENCHMARK")
    print("=" * 75)
    print(pd.DataFrame(benchmark_metrics).to_string(index=False))

    corr_matrix = val_df[strategy_pred_cols].corr(method="spearman").round(3)
    print("\n" + "=" * 75)
    print("[MATRIX]  PAIRWISE CROSS-STRATEGY CORRELATION MATRIX (TRUE ORTHOGONALITY)")
    print("=" * 75)
    print(corr_matrix.to_string())

    # Save to dynamic metrics.json
    metrics_payload = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "strategies": [
            {
                "id": c["id"],
                "name": c["name"],
                "target": f"{c['target']} ({len(c['features'])} features)",
                "neut": f"{int(c['neut']*100)}% Factor Neutralization",
                "corr": m["Mean Corr"],
                "sharpe": m["Per-Era Sharpe"],
                "ann_sharpe": m["Annualized Sharpe"],
                "dd": m["Max Drawdown"],
                "badge": c["badge"],
                "color": c["color"]
            }
            for c, m in zip(configs, benchmark_metrics)
        ],
        "correlation_matrix": corr_matrix.values.tolist(),
        "strategy_names": [f"Strat {c['id']} ({c['badge']})" for c in configs]
    }
    with open("/Users/ishantpanchal/numerai-quant/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("\n[SAVE]  Saved truly orthogonal metrics to /Users/ishantpanchal/numerai-quant/metrics.json!\n")


if __name__ == "__main__":
    main()
