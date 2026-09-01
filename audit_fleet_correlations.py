#!/usr/bin/env python3
"""
Numerai 5-Strategy Fleet Performance & Correlation Audit
- Computes out-of-sample metrics (Corr20v2, Sharpe, Drawdown)
- Computes pairwise Spearman correlation matrix
- Dynamically saves results to metrics.json for live dashboard consumption
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import FEATURE_SET, FEATURES_JSON, DATA_DIR, MODEL_DIR
from neutralize import neutralize, rank_01

VOL_DIR = "/Users/ishantpanchal/numerai-quant/models/vol_specialist"
ALPHA_DIR = "/Users/ishantpanchal/numerai-quant/models/alpha_specialist"
TAIL_DIR = "/Users/ishantpanchal/numerai-quant/models/tail_specialist"
RESIDUAL_DIR = "/Users/ishantpanchal/numerai-quant/models/residual_specialist"
METRICS_JSON = "/Users/ishantpanchal/numerai-quant/metrics.json"


def get_feature_list() -> list:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    return meta["feature_sets"][FEATURE_SET]


def main():
    features = get_feature_list()
    val_path = os.path.join(DATA_DIR, "validation.parquet")

    print("=" * 75)
    print("[RESEARCH]  NUMERAI 5-STRATEGY FLEET PERFORMANCE & DYNAMIC METRICS AUDIT")
    print("=" * 75)

    val_cols = ["era", "target"] + features
    print("[LOAD]  Loading validation dataset (4.12M rows)...")
    val_df = pd.read_parquet(val_path, columns=val_cols)
    X_val = val_df[features]
    neutralizer_feats = features[:60]

    # Strategy 1: Core Alpha Ensemble (25% Neutralized)
    print("[AUDIT]  Computing Strategy 1: Core Alpha Ensemble...")
    s1_files = sorted(glob.glob(os.path.join(MODEL_DIR, "lgb_*.pkl")))
    s1_preds = [rank_01(joblib.load(f).predict(X_val)) for f in s1_files]
    val_df["s1_raw"] = pd.DataFrame(s1_preds).T.mean(axis=1).values
    val_df["s1_raw"] = val_df.groupby("era", observed=True)["s1_raw"].rank(pct=True)
    val_df["Strat_1_Core"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(neutralize(era, ["s1_raw"], extra_neutralizers=neutralizer_feats, proportion=0.25)["s1_raw"].values, index=era.index)
    )

    # Strategy 2: Tail-Risk Volatility Specialist (50% Neutralized)
    print("[AUDIT]  Computing Strategy 2: Tail-Risk Volatility Specialist...")
    s2_files = sorted(glob.glob(os.path.join(VOL_DIR, "lgb_*.pkl")))
    s2_preds = [rank_01(joblib.load(f).predict(X_val)) for f in s2_files]
    val_df["s2_raw"] = pd.DataFrame(s2_preds).T.mean(axis=1).values
    val_df["s2_raw"] = val_df.groupby("era", observed=True)["s2_raw"].rank(pct=True)
    val_df["Strat_2_Vol"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(neutralize(era, ["s2_raw"], extra_neutralizers=neutralizer_feats, proportion=0.50)["s2_raw"].values, index=era.index)
    )

    # Strategy 3: Quality Momentum Specialist (35% Neutralized)
    print("[AUDIT]  Computing Strategy 3: Quality Momentum Specialist...")
    s3_files = sorted(glob.glob(os.path.join(ALPHA_DIR, "lgb_*.pkl")))
    s3_preds = [rank_01(joblib.load(f).predict(X_val)) for f in s3_files]
    val_df["s3_raw"] = pd.DataFrame(s3_preds).T.mean(axis=1).values
    val_df["s3_raw"] = val_df.groupby("era", observed=True)["s3_raw"].rank(pct=True)
    val_df["Strat_3_Quality"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(neutralize(era, ["s3_raw"], extra_neutralizers=neutralizer_feats, proportion=0.35)["s3_raw"].values, index=era.index)
    )

    # Strategy 4: Pure Tail-Risk (target_xerxes_20) (40% Neutralized)
    print("[AUDIT]  Computing Strategy 4: Pure Tail-Risk (xerxes_20)...")
    s4_model = joblib.load(os.path.join(TAIL_DIR, "lgb_target_xerxes_20.pkl"))
    val_df["s4_raw"] = s4_model.predict(X_val)
    val_df["s4_raw"] = val_df.groupby("era", observed=True)["s4_raw"].rank(pct=True)
    val_df["Strat_4_Tail"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(neutralize(era, ["s4_raw"], extra_neutralizers=neutralizer_feats, proportion=0.40)["s4_raw"].values, index=era.index)
    )

    # Strategy 5: Pure Residual (target_delta_20) (40% Neutralized)
    print("[AUDIT]  Computing Strategy 5: Pure Residual (delta_20)...")
    s5_model = joblib.load(os.path.join(RESIDUAL_DIR, "lgb_target_delta_20.pkl"))
    val_df["s5_raw"] = s5_model.predict(X_val)
    val_df["s5_raw"] = val_df.groupby("era", observed=True)["s5_raw"].rank(pct=True)
    val_df["Strat_5_Residual"] = val_df.groupby("era", observed=True, group_keys=False).apply(
        lambda era: pd.Series(neutralize(era, ["s5_raw"], extra_neutralizers=neutralizer_feats, proportion=0.40)["s5_raw"].values, index=era.index)
    )

    strategy_cols = ["Strat_1_Core", "Strat_2_Vol", "Strat_3_Quality", "Strat_4_Tail", "Strat_5_Residual"]

    # Calculate Individual Era Correlations & Metrics
    metrics_list = []
    strat_meta = [
        {"id": 1, "name": "Core Alpha Ensemble", "target": "target (4-Target Blend: Cyrus, Agnes, Victor, Jeremy)", "neut": "25% Linear Feature Neutralization", "badge": "Flagship", "color": "#3b82f6"},
        {"id": 2, "name": "Tail-Risk Volatility Specialist", "target": "victor_20 + xerxes_20 + delta_20", "neut": "50% Heavy Feature Neutralization", "badge": "High MMC", "color": "#10b981"},
        {"id": 3, "name": "Quality Momentum Specialist", "target": "target_jeremy_20 + target_agnes_20", "neut": "35% Balanced Feature Neutralization", "badge": "Orthogonal", "color": "#8b5cf6"},
        {"id": 4, "name": "Pure Tail-Risk Specialist", "target": "target_xerxes_20 (Tail Volatility)", "neut": "40% Factor Neutralization", "badge": "Min Drawdown", "color": "#f59e0b"},
        {"id": 5, "name": "Pure Residual Specialist", "target": "target_delta_20 (Uncorrelated Residuals)", "neut": "40% Factor Neutralization", "badge": "Residual Alpha", "color": "#ec4899"}
    ]

    for idx, s in enumerate(strategy_cols):
        def era_corr(era_df):
            return spearmanr(era_df[s], era_df["target"]).statistic
        corrs = val_df.groupby("era", observed=True).apply(era_corr)
        m_corr = float(corrs.mean())
        s_sharpe = float(m_corr / (corrs.std() + 1e-8))
        m_dd = float((corrs.cumsum().cummax() - corrs.cumsum()).max())

        item = strat_meta[idx].copy()
        item["corr"] = f"+{m_corr:.4f}"
        item["sharpe"] = f"{s_sharpe:.3f}"
        item["ann_sharpe"] = f"{s_sharpe * np.sqrt(12):.3f}"
        item["dd"] = f"{m_dd * 100:.2f}%"
        metrics_list.append(item)

    # Cross-Strategy Correlation Matrix
    corr_df = val_df[strategy_cols].corr(method="spearman").round(3)
    corr_matrix_list = corr_df.values.tolist()

    output_payload = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "strategies": metrics_list,
        "correlation_matrix": corr_matrix_list,
        "strategy_names": ["Strat 1 (Core)", "Strat 2 (Vol)", "Strat 3 (Quality)", "Strat 4 (Tail)", "Strat 5 (Residual)"]
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SAVE]  Dynamically saved live performance snapshot to {METRICS_JSON}")

    print("\n" + "=" * 75)
    print("[BENCHMARK]  5-STRATEGY OUT-OF-SAMPLE PERFORMANCE BENCHMARK")
    print("=" * 75)
    print(pd.DataFrame(metrics_list)[["name", "corr", "sharpe", "ann_sharpe", "dd"]].to_string(index=False))
    print("\n" + "=" * 75)
    print("[MATRIX]  PAIRWISE CROSS-STRATEGY CORRELATION MATRIX")
    print("=" * 75)
    print(corr_df.to_string())


if __name__ == "__main__":
    main()
