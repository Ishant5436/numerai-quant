#!/usr/bin/env python3
"""
Numerai 25-Model Fleet 60-Day Out-of-Sample Evaluation Engine.
Computes Spearman correlation, Sharpe, drawdown, cross-model correlation,
and PCA effective bet metrics across validation eras.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    BASE_DIR,
    DATA_DIR,
    FLEET_STRATEGY_MAP_60D,
    FLAGSHIP_ANCHOR_WEIGHT,
    MODEL_60D_DIR,
    ORTHO_60D_DIR,
)
from fleet_submit import load_feature_groups
from neutralize import neutralize, rank_01


def calc_era_spearman(df: pd.DataFrame, pred_col: str = "pred", target_col: str = "target") -> pd.Series:
    """Calculate per-era Spearman correlation between predictions and target."""
    assert pred_col in df.columns, f"Missing prediction column: {pred_col}"
    assert target_col in df.columns, f"Missing target column: {target_col}"
    assert "era" in df.columns, "DataFrame must contain 'era' column"

    era_corrs = {}
    eras = sorted(df["era"].unique())
    assert len(eras) > 0, "No eras present in dataframe"

    for era in eras:
        sub = df[df["era"] == era].dropna(subset=[pred_col, target_col])
        if len(sub) < 50:
            continue
        p_vals = sub[pred_col].values if isinstance(sub[pred_col], pd.Series) else sub[pred_col].iloc[:, 0].values
        t_vals = sub[target_col].values if isinstance(sub[target_col], pd.Series) else sub[target_col].iloc[:, 0].values
        corr, _ = spearmanr(p_vals, t_vals)
        if hasattr(corr, "__len__"):
            corr = float(np.ravel(corr)[0])
        if np.isfinite(corr):
            era_corrs[era] = float(corr)

    assert len(era_corrs) > 0, f"Failed to compute any valid era correlations for {target_col}"
    return pd.Series(era_corrs)


def calc_sharpe(era_corrs: pd.Series) -> float:
    """Calculate annualized Sharpe ratio from a series of per-era correlations."""
    assert len(era_corrs) > 0, "Cannot compute Sharpe on empty correlation series"
    std = float(era_corrs.std())
    if std < 1e-7:
        return 0.0
    # Annualized Sharpe (52 weekly eras per year)
    sharpe = float(era_corrs.mean() / std) * np.sqrt(52.0)
    assert np.isfinite(sharpe), "Calculated Sharpe must be finite"
    return sharpe


def calc_cumulative_drawdown(era_corrs: pd.Series) -> float:
    """Calculate maximum peak-to-trough cumulative drawdown in correlation units."""
    assert len(era_corrs) > 0, "Cannot compute drawdown on empty correlation series"
    cumsum = era_corrs.cumsum()
    running_max = cumsum.cummax()
    drawdown = running_max - cumsum
    max_dd = float(drawdown.max())
    assert max_dd >= 0.0, f"Negative drawdown encountered: {max_dd}"
    return max_dd


def calc_effective_bets(corr_matrix: np.ndarray) -> float:
    """Calculate effective number of independent bets N_eff via PCA eigenvalue entropy."""
    assert corr_matrix.ndim == 2, "Correlation matrix must be 2D"
    assert corr_matrix.shape[0] == corr_matrix.shape[1], "Matrix must be square"
    
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    # Formula: N_eff = (sum lambda_i)^2 / sum (lambda_i^2)
    sum_eig = np.sum(eigenvalues)
    sum_sq_eig = np.sum(eigenvalues ** 2)
    assert sum_sq_eig > 0.0, "Zero eigenvalue sum-of-squares"
    
    n_eff = float((sum_eig ** 2) / sum_sq_eig)
    assert 1.0 <= n_eff <= corr_matrix.shape[0] + 0.1, f"N_eff {n_eff} out of physical bounds"
    return n_eff


def calc_risk_parity_weights(cov_matrix: np.ndarray, sharpes: np.ndarray) -> np.ndarray:
    """Calculate Markowitz risk-parity staking weights from covariance and Sharpe ratios."""
    assert cov_matrix.shape[0] == len(sharpes), "Covariance and Sharpe length mismatch"
    assert len(sharpes) > 0, "Empty Sharpe vector"

    diag_vol = np.sqrt(np.diag(cov_matrix))
    diag_vol = np.maximum(diag_vol, 1e-6)
    clamped_sharpe = np.maximum(sharpes, 0.01)

    raw_weights = (clamped_sharpe / (diag_vol ** 2))
    sum_weights = np.sum(raw_weights)
    assert sum_weights > 0.0, "Sum of raw risk-parity weights must be positive"

    norm_weights = raw_weights / sum_weights
    # Apply 20% concentration ceiling
    capped_weights = np.minimum(norm_weights, 0.20)
    capped_weights /= np.sum(capped_weights)
    
    assert np.isclose(np.sum(capped_weights), 1.0, atol=1e-4), "Weights must sum to 1.0"
    return capped_weights


def load_validation_data(val_path: str, targets: list, features: list, stride: int = 4) -> pd.DataFrame:
    """Load validation.parquet with bounded columns and era stride sampling."""
    assert os.path.exists(val_path), f"Validation parquet missing: {val_path}"
    assert len(targets) > 0, "Target list cannot be empty"
    assert len(features) > 0, "Feature list cannot be empty"

    cols = ["era"] + targets + features
    print(f"[VAL LOAD] Ingesting validation dataset ({len(cols)} columns, stride={stride})...")
    df = pd.read_parquet(val_path, columns=cols)
    assert len(df) > 0, "Read empty validation dataset"

    if stride > 1:
        eras = sorted(df["era"].unique())
        selected = eras[::stride]
        df = df[df["era"].isin(selected)].copy()
        print(f"[VAL INFO] Sampled {len(selected)}/{len(eras)} validation eras")

    assert len(df) > 0, "Empty validation dataset after stride sampling"
    return df


def predict_strategy(
    val_df: pd.DataFrame,
    strat_id: int,
    feat_subset: list,
    neut_prop: float,
    fncv3_feats: list,
    anchor_weight: float = None,
    quintet_raw: np.ndarray = None,
) -> np.ndarray:
    """Generate out-of-sample predictions for a single strategy."""
    assert strat_id in range(1, 26), f"Invalid strat_id: {strat_id}"
    assert len(feat_subset) > 0, f"Empty features for strategy {strat_id}"
    if anchor_weight is None:
        anchor_weight = FLAGSHIP_ANCHOR_WEIGHT if strat_id > 1 else 0.0

    if strat_id == 1:
        if quintet_raw is not None:
            raw_pred = quintet_raw
        else:
            targets_60d = ["target_cyrusd_60", "target_agnes_60", "target_victor_60", "target_jeremy_60", "target_xerxes_60"]
            preds = []
            for t in targets_60d:
                m = joblib.load(os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl"))
                preds.append(rank_01(m.predict(val_df[feat_subset])))
            raw_pred = np.mean(preds, axis=0)
    else:
        model_path = os.path.join(ORTHO_60D_DIR, f"lgb_strat_{strat_id}.pkl")
        assert os.path.exists(model_path), f"Missing model: {model_path}"
        m = joblib.load(model_path)
        raw_pred = m.predict(val_df[feat_subset])

        if anchor_weight > 0.0 and quintet_raw is not None:
            raw_pred = (1.0 - anchor_weight) * rank_01(raw_pred) + anchor_weight * rank_01(quintet_raw)

    df_copy = val_df[["era"] + fncv3_feats].copy()
    df_copy["pred"] = rank_01(raw_pred)
    df_copy = neutralize(df_copy, ["pred"], extra_neutralizers=fncv3_feats, proportion=neut_prop)
    
    final_pred = rank_01(df_copy["pred"].values)
    assert len(final_pred) == len(val_df), "Prediction length mismatch"
    return final_pred


def run_fleet_evaluation(stride: int = 4) -> dict:
    """Execute complete 25-strategy evaluation on validation data."""
    val_path = os.path.join(DATA_DIR, "validation.parquet")
    assert os.path.exists(val_path), f"Validation file missing: {val_path}"
    
    groups = load_feature_groups()
    all_medium = groups["all_medium"]
    fncv3 = groups["fncv3_features"]
    
    # Identify unique targets needed
    targets_needed = sorted(list(set(
        ["target", "target_cyrusd_60"] + [FLEET_STRATEGY_MAP_60D[s][0] for s in range(2, 26)]
    )))
    
    val_df = load_validation_data(val_path, targets_needed, all_medium, stride=stride)
    
    print("[VAL INIT] Precomputing Flagship Quintet raw predictions...")
    targets_60d = ["target_cyrusd_60", "target_agnes_60", "target_victor_60", "target_jeremy_60", "target_xerxes_60"]
    quintet_raw_preds = []
    for t in targets_60d:
        m = joblib.load(os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl"))
        quintet_raw_preds.append(rank_01(m.predict(val_df[all_medium])))
    quintet_raw = np.mean(quintet_raw_preds, axis=0)

    strat_metrics = {}
    fleet_preds = {}

    for strat_id in range(1, 26):
        target_col, feat_key, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
        eval_target = "target_cyrusd_60" if strat_id == 1 else target_col
        feats = groups[feat_key]
        
        preds = predict_strategy(val_df, strat_id, feats, neut_prop, fncv3, quintet_raw=quintet_raw)
        fleet_preds[strat_id] = preds
        
        eval_cols = list(dict.fromkeys(["era", eval_target, "target_cyrusd_60"]))
        eval_df = val_df[eval_cols].copy()
        eval_df["pred"] = preds
        
        era_corrs = calc_era_spearman(eval_df, pred_col="pred", target_col=eval_target)
        mean_corr = float(era_corrs.mean())
        sharpe = calc_sharpe(era_corrs)
        drawdown = calc_cumulative_drawdown(era_corrs)

        # Also calculate benchmark spearman correlation on target_cyrusd_60
        bench_corrs = calc_era_spearman(eval_df, pred_col="pred", target_col="target_cyrusd_60")
        bench_mean = float(bench_corrs.mean())
        bench_sharpe = calc_sharpe(bench_corrs)
        
        strat_metrics[strat_id] = {
            "strategy_id": strat_id,
            "target": target_col,
            "feature_group": feat_key,
            "neutralization": neut_prop,
            "mean_spearman": round(mean_corr, 5),
            "annualized_sharpe": round(sharpe, 3),
            "per_era_sharpe": round(sharpe / np.sqrt(52.0), 3),
            "max_drawdown": round(drawdown, 4),
            "benchmark_spearman_60d": round(bench_mean, 5),
            "benchmark_sharpe_60d": round(bench_sharpe, 3),
            "n_eras": len(era_corrs)
        }
        print(f"[STRAT {strat_id:2d}] Target: {mean_corr:+.4f} (Sharpe {sharpe:+.2f}) | Bench 60d: {bench_mean:+.4f} (Sharpe {bench_sharpe:+.2f}) | DD: {drawdown:.4f}")

    # Compute 25 x 25 cross-strategy correlation matrix
    pred_matrix = np.column_stack([fleet_preds[s] for s in range(1, 26)])
    corr_matrix = np.corrcoef(pred_matrix, rowvar=False)
    n_eff = calc_effective_bets(corr_matrix)
    print(f"\n[FLEET STATS] Effective Independent Bets (N_eff): {n_eff:.2f} / 25.00")

    # Compute risk-parity weights
    cov_matrix = np.cov(pred_matrix, rowvar=False)
    sharpes = np.array([strat_metrics[s]["annualized_sharpe"] for s in range(1, 26)])
    weights = calc_risk_parity_weights(cov_matrix, sharpes)

    for i, strat_id in enumerate(range(1, 26)):
        strat_metrics[strat_id]["staking_weight"] = round(float(weights[i]), 4)

    output = {
        "evaluation_stride": stride,
        "n_validation_eras": val_df["era"].nunique(),
        "effective_independent_bets": round(n_eff, 2),
        "mean_fleet_pairwise_corr": round(float(np.mean(corr_matrix[np.triu_indices(25, k=1)])), 4),
        "strategies": strat_metrics
    }
    
    out_file = os.path.join(DATA_DIR, "fleet_60d_validation_metrics.json")
    repo_file = os.path.join(BASE_DIR, "metrics_fleet_60d.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    with open(repo_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] Evaluation metrics written to {out_file} and {repo_file}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Numerai 25-Model Fleet 60-Day Evaluation Harness")
    parser.add_argument("--stride", type=int, default=4, help="Validation era stride (default: 4)")
    args = parser.parse_args()
    run_fleet_evaluation(stride=args.stride)


if __name__ == "__main__":
    main()
