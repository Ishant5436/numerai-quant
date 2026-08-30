#!/usr/bin/env python3
"""
Numerai Tri-Algorithm Gradient Boosting Fleet Research Engine
Blends LightGBM + XGBoost + CatBoost across orthogonal feature factor groups:
1. LightGBM (Histogram leaf-wise splitting)
2. XGBoost (Exact depth-wise regularization)
3. CatBoost (Symmetric oblivious decision trees)
Reduces single-model variance, boosts out-of-sample Sharpe, and maximizes Meta Model Contribution (MMC).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from config import FEATURES_JSON, DATA_DIR
from neutralize import neutralize, rank_01

TRI_DIR = "/Users/ishantpanchal/numerai-quant/models/tri_ensemble_fleet"
os.makedirs(TRI_DIR, exist_ok=True)


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
        "fundamental": get_subset(["intelligence", "charisma", "wisdom"]),
        "momentum": get_subset(["strength", "dexterity", "agility"]),
        "macro": get_subset(["serenity", "sunshine", "midnight"]),
        "constitution": get_subset(["constitution", "dexterity"])
    }
    return groups


def calculate_era_correlation(df: pd.DataFrame, pred_col: str, target_col: str = "target") -> pd.Series:
    def era_corr(era_df):
        return spearmanr(era_df[pred_col], era_df[target_col]).statistic
    return df.groupby("era", observed=True).apply(era_corr)


def main():
    groups = load_feature_groups()
    print("=" * 80)
    print("🧠 NUMERAI TRI-ALGORITHM (LGBM + XGBOOST + CATBOOST) STACKING ENGINE")
    print("=" * 80)

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    cols_to_load = ["era", "target", "target_victor_20", "target_xerxes_20", "target_jeremy_20", "target_delta_20"] + groups["all_medium"]
    print("\n📥 Loading train & validation datasets...")
    train_df = pd.read_parquet(train_path, columns=cols_to_load)
    val_df = pd.read_parquet(val_path, columns=["era", "target"] + groups["all_medium"])

    configs = [
        {
            "id": 1,
            "name": "Core Tri-Ensemble Flagship",
            "target": "target",
            "features": groups["all_medium"],
            "neut": 0.25,
            "badge": "Flagship"
        },
        {
            "id": 2,
            "name": "Fundamental Tri-Ensemble Specialist",
            "target": "target_jeremy_20",
            "features": groups["fundamental"],
            "neut": 0.35,
            "badge": "Fundamental"
        },
        {
            "id": 3,
            "name": "Momentum Tri-Ensemble Specialist",
            "target": "target_victor_20",
            "features": groups["momentum"],
            "neut": 0.40,
            "badge": "Momentum"
        },
        {
            "id": 4,
            "name": "Macro Regime Tri-Ensemble Specialist",
            "target": "target_xerxes_20",
            "features": groups["macro"],
            "neut": 0.45,
            "badge": "Macro Regime"
        },
        {
            "id": 5,
            "name": "Constitution Residual Tri-Ensemble",
            "target": "target_delta_20",
            "features": groups["constitution"],
            "neut": 0.50,
            "badge": "Residual"
        }
    ]

    strategy_pred_cols = []
    benchmark_metrics = []
    neutralizer_feats = groups["all_medium"][:60]

    for cfg in configs:
        print(f"\n⚡ Training Strategy {cfg['id']} [{cfg['name']}] on {len(cfg['features'])} features (Target: {cfg['target']})...")
        valid_mask = ~train_df[cfg["target"]].isna()
        X_train = train_df.loc[valid_mask, cfg["features"]]
        y_train = train_df.loc[valid_mask, cfg["target"]]
        X_val = val_df[cfg["features"]]

        # 1. LightGBM
        print("  [1/3] Fitting LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=32, max_depth=5, colsample_bytree=0.2, n_jobs=-1, random_state=100 + cfg["id"])
        lgb_model.fit(X_train, y_train)
        joblib.dump(lgb_model, os.path.join(TRI_DIR, f"lgb_strat_{cfg['id']}.pkl"))
        p_lgb = lgb_model.predict(X_val)

        # 2. XGBoost
        print("  [2/3] Fitting XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=4, colsample_bytree=0.2, subsample=0.8, n_jobs=-1, random_state=200 + cfg["id"], tree_method="hist")
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, os.path.join(TRI_DIR, f"xgb_strat_{cfg['id']}.pkl"))
        p_xgb = xgb_model.predict(X_val)

        # 3. CatBoost
        print("  [3/3] Fitting CatBoost...")
        cb_model = CatBoostRegressor(iterations=250, learning_rate=0.03, depth=5, thread_count=-1, random_seed=300 + cfg["id"], verbose=0)
        cb_model.fit(X_train, y_train)
        joblib.dump(cb_model, os.path.join(TRI_DIR, f"cb_strat_{cfg['id']}.pkl"))
        p_cb = cb_model.predict(X_val)

        # 4. Tri-Algorithm Blending
        val_df["temp_lgb"] = p_lgb
        val_df["temp_xgb"] = p_xgb
        val_df["temp_cb"] = p_cb

        r_lgb = val_df.groupby("era", observed=True)["temp_lgb"].rank(pct=True)
        r_xgb = val_df.groupby("era", observed=True)["temp_xgb"].rank(pct=True)
        r_cb = val_df.groupby("era", observed=True)["temp_cb"].rank(pct=True)

        blend = 0.40 * r_lgb + 0.30 * r_xgb + 0.30 * r_cb
        raw_col = f"Strat_{cfg['id']}_TriBlend"
        val_df[raw_col] = blend

        # 5. Factor Neutralization
        neut_col = f"Strat_{cfg['id']}_{cfg['name'].replace(' ', '_')}"
        val_df[neut_col] = val_df.groupby("era", observed=True, group_keys=False).apply(
            lambda era: pd.Series(neutralize(era, [raw_col], extra_neutralizers=neutralizer_feats, proportion=cfg["neut"])[raw_col].values, index=era.index)
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

    print("\n" + "=" * 80)
    print("🏆 TRI-ALGORITHM ENSEMBLE BENCHMARK RESULTS")
    print("=" * 80)
    print(pd.DataFrame(benchmark_metrics).to_string(index=False))

    corr_matrix = val_df[strategy_pred_cols].corr(method="spearman").round(3)
    print("\n" + "=" * 80)
    print("🔗 PAIRWISE CROSS-STRATEGY CORRELATION MATRIX")
    print("=" * 80)
    print(corr_matrix.to_string())

    print("\n✅ All 15 models (3 algorithms x 5 strategies) successfully trained and saved!")

if __name__ == "__main__":
    main()
