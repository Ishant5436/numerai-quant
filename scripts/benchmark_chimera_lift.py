#!/usr/bin/env python3
"""
Empirical Benchmark: Quantifying Sharpe & Correlation Lift of Chimera Alphas.
Compares Out-of-Sample performance of LightGBM trained with and without the 33 Chimera Alphas.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DATA_DIR, FEATURES_JSON, CHIMERA_VAULT_PATH
from chimera.alpha_vault import AlphaVault

def main():
    print("=================================================================")
    print("    CHIMERA EMPIRICAL LIFT BENCHMARK: 33 ALPHAS OUT-OF-SAMPLE    ")
    print("=================================================================")

    val_path = os.path.join(DATA_DIR, "validation.parquet")
    assert os.path.exists(val_path), f"Missing validation data: {val_path}"
    
    with open(FEATURES_JSON, "r") as f:
        meta = json.load(f)
    medium_features = meta["feature_sets"]["medium"]
    
    # Use 130 base features to cover all feature indices mined by the daemon
    base_features = medium_features[:130]
    target_col = "target_cyrusd_60"

    print("[*] Reading validation dataset...")
    cols = ["era", target_col] + base_features
    df = pd.read_parquet(val_path, columns=cols)
    df = df.dropna(subset=[target_col])

    unique_eras = sorted(df["era"].unique())
    # Split: first 10 eras for training, next 10 eras for out-of-sample testing
    step = max(1, len(unique_eras) // 24)
    sampled_eras = unique_eras[::step][:20]
    train_eras = sampled_eras[:10]
    test_eras = sampled_eras[10:20]

    train_df = df[df["era"].isin(train_eras)].copy()
    test_df = df[df["era"].isin(test_eras)].copy()

    print(f"[*] Train set: {len(train_df):,} rows across {len(train_eras)} eras")
    print(f"[*] Test set (Out-of-Sample): {len(test_df):,} rows across {len(test_eras)} eras")

    # 1. Load Chimera Alphas from Vault
    vault = AlphaVault.load(CHIMERA_VAULT_PATH)
    print(f"[*] Loaded {len(vault.entries)} verified Chimera Alphas from {CHIMERA_VAULT_PATH}")
    assert len(vault.entries) > 0, "No alphas in vault"

    print("[*] Augmenting feature matrices with C++ SIMD kernel...")
    t0 = time.perf_counter()
    train_df_aug = vault.augment_dataframe(train_df.copy(), feature_cols=base_features)
    test_df_aug = vault.augment_dataframe(test_df.copy(), feature_cols=base_features)
    aug_time = time.perf_counter() - t0
    print(f"[+] Vectorized augmentation complete in {aug_time:.2f}s ({len(train_df_aug) + len(test_df_aug):,} rows)")

    chimera_cols = [e.name for e in vault.entries]

    # LightGBM parameters (fast benchmark)
    params = {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 31,
        "colsample_bytree": 0.3,
        "subsample": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    # Model A: Baseline (Base Features only)
    print("\n[*] Training Model A (Baseline: 100 Raw Features)...")
    model_a = lgb.LGBMRegressor(**params)
    model_a.fit(train_df[base_features], train_df[target_col])
    test_df["pred_a"] = model_a.predict(test_df[base_features])

    # Model B: Chimera-Augmented (Base Features + 33 Alphas)
    aug_features = base_features + chimera_cols
    print(f"[*] Training Model B (Chimera-Augmented: 100 Raw Features + {len(chimera_cols)} Synthetic Alphas)...")
    model_b = lgb.LGBMRegressor(**params)
    model_b.fit(train_df_aug[aug_features], train_df_aug[target_col])
    test_df["pred_b"] = model_b.predict(test_df_aug[aug_features])

    # Out-of-Sample Per-Era Evaluation
    def calc_era_scores(pred_col):
        corrs = []
        for era in test_eras:
            era_data = test_df[test_df["era"] == era]
            c, _ = spearmanr(era_data[pred_col], era_data[target_col])
            corrs.append(c)
        return np.array(corrs)

    scores_a = calc_era_scores("pred_a")
    scores_b = calc_era_scores("pred_b")

    mean_a, std_a = np.mean(scores_a), np.std(scores_a)
    mean_b, std_b = np.mean(scores_b), np.std(scores_b)

    sharpe_a = mean_a / (std_a + 1e-8)
    sharpe_b = mean_b / (std_b + 1e-8)

    pos_a = np.mean(scores_a > 0)
    pos_b = np.mean(scores_b > 0)

    # Feature importances of Model B
    importances = model_b.feature_importances_
    feat_imp = pd.DataFrame({"feature": aug_features, "importance": importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    top_chimera_imp = feat_imp[feat_imp["feature"].isin(chimera_cols)].head(5)

    print("\n" + "="*65)
    print("                 OUT-OF-SAMPLE BENCHMARK RESULTS                ")
    print("="*65)
    print(" Metric                     Model A (Raw)    Model B (+Chimera)    Lift")
    print("-----------------------------------------------------------------")
    print(f" Mean Era Correlation (CORR)   {mean_a:+.4f}           {mean_b:+.4f}        {mean_b - mean_a:+.4f}")
    print(f" Era Std Dev (Volatility)      {std_a:.4f}            {std_b:.4f}        {std_b - std_a:+.4f}")
    print(f" Raw Per-Era Sharpe (mu/sigma) {sharpe_a:+.3f}           {sharpe_b:+.3f}        {sharpe_b - sharpe_a:+.3f}")
    print(f" Annualized Sharpe (x sqrt12)  {sharpe_a*np.sqrt(12):+.3f}           {sharpe_b*np.sqrt(12):+.3f}        {(sharpe_b - sharpe_a)*np.sqrt(12):+.3f}")
    print(f" Positive Era Ratio            {pos_a:.1%}            {pos_b:.1%}        {pos_b - pos_a:+.1%}")
    print("="*65)

    print("\nTop Mined Chimera Alphas by Model Feature Importance:")
    for idx, row in top_chimera_imp.iterrows():
        rank = feat_imp[feat_imp["feature"] == row["feature"]].index[0] + 1
        print(f"  Rank #{rank:2d}: {row['feature']:<18} | Split Gain: {row['importance']:.1f}")

    if sharpe_b > sharpe_a:
        pct_lift = ((sharpe_b - sharpe_a) / max(0.001, abs(sharpe_a))) * 100
        print(f"\n[+] Empirical Result: Chimera Alphas delivered a +{pct_lift:.1f}% increase in Out-of-Sample Sharpe!")
    print("="*65)

if __name__ == "__main__":
    main()
