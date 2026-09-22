#!/usr/bin/env python3
"""
Chimera Alpha Synthesis Campaign Runner.
Runs bounded genetic evolutionary search rounds across diverse horizons
until the target number of verified orthogonal alphas is admitted.
"""
import os
import sys
import json
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter
from chimera.alpha_vault import AlphaVault
from config import DATA_DIR, FEATURES_JSON, CHIMERA_VAULT_PATH

def load_medium_feature_pool() -> List[str]:
    assert os.path.exists(FEATURES_JSON), f"Features metadata missing: {FEATURES_JSON}"
    with open(FEATURES_JSON, "r") as f:
        meta = json.load(f)
    pool = meta.get("feature_sets", {}).get("medium", [])
    assert len(pool) >= 100, f"Medium feature pool too small: {len(pool)}"
    return pool

def sample_mining_slice(
    val_path: str,
    target_name: str,
    feature_pool: List[str],
    n_features: int = 100,
    n_eras: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    assert os.path.exists(val_path), f"Validation file missing: {val_path}"
    assert len(feature_pool) >= n_features, "Requested features exceed pool size"
    
    selected_features = list(np.random.choice(feature_pool, n_features, replace=False))
    cols = ["era", target_name] + selected_features
    df = pd.read_parquet(val_path, columns=cols)
    
    unique_eras = df["era"].unique()
    chosen_eras = np.random.choice(unique_eras, min(n_eras, len(unique_eras)), replace=False)
    sample_df = df[df["era"].isin(chosen_eras)].dropna(subset=[target_name])
    
    feats = np.ascontiguousarray(sample_df[selected_features].values, dtype=np.float32)
    targs = np.ascontiguousarray(sample_df[target_name].values, dtype=np.float32)
    eras = np.ascontiguousarray(sample_df["era"].values)
    return feats, targs, eras, selected_features

def run_synthesis_campaign(target_new_alphas: int = 4, max_cycles: int = 12) -> int:
    assert target_new_alphas > 0, "Target new alphas must be positive"
    assert max_cycles > 0 and max_cycles <= 30, "Max cycles bounded between 1 and 30"
    
    feature_pool = load_medium_feature_pool()
    val_path = os.path.join(DATA_DIR, "validation.parquet")
    schema_names = pq.read_schema(val_path).names
    targets = [t for t in ["target_cyrusd_60", "target_jeremy_60", "target_victor_60", "target_xerxes_60"] if t in schema_names] or ["target"]
    
    initial_vault = AlphaVault.load(CHIMERA_VAULT_PATH)
    start_count = len(initial_vault.entries)
    goal_count = start_count + target_new_alphas
    print(f"[*] Starting Alpha Campaign: Vault currently has {start_count} alphas. Target: {goal_count}")
    
    filter_gate = TriHurdleFilter(min_sharpe=0.70, max_factor_corr=0.15, min_positive_era_ratio=0.55)
    
    for cycle in range(1, max_cycles + 1):
        target_name = np.random.choice(targets)
        feats, targs, eras, sel_feats = sample_mining_slice(val_path, target_name, feature_pool, n_features=100, n_eras=10)
        
        synth = GeneticSynthesizer(
            num_features=len(sel_feats),
            pop_size=40,
            tournament_size=3,
            mutation_rate=0.40,
            crossover_rate=0.65,
            max_depth=4,
            vault_path=CHIMERA_VAULT_PATH,
            filter_gate=filter_gate
        )
        
        print(f"[*] Cycle {cycle}/{max_cycles}: Mining {len(feats):,} rows on {target_name}...")
        synth.evolve_on_dataset(feats, targs, eras, generations=6)
        
        current_vault = AlphaVault.load(CHIMERA_VAULT_PATH)
        current_count = len(current_vault.entries)
        print(f"[+] Cycle {cycle} Complete: Vault now holds {current_count} alphas (admitted this campaign: {current_count - start_count})")
        
        if current_count >= goal_count:
            print(f"[SUCCESS] Goal reached! Admitted {current_count - start_count} new alphas to vault.")
            break
            
    final_vault = AlphaVault.load(CHIMERA_VAULT_PATH)
    return len(final_vault.entries) - start_count

def main():
    start_t = time.time()
    admitted = run_synthesis_campaign(target_new_alphas=4, max_cycles=12)
    elapsed = time.time() - start_t
    print(f"[+] Campaign finished in {elapsed:.1f}s. Total newly admitted alphas: {admitted}")

if __name__ == "__main__":
    main()
