#!/usr/bin/env python3
"""
Chimera Multi-Horizon Alpha Synthesis Batch Runner.
Executes systematic genetic synthesis rounds across orthogonal 60d targets.
"""
import os
import sys
import json
import time
from typing import List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter
from config import DATA_DIR, FEATURES_JSON, CHIMERA_VAULT_PATH

def load_features_metadata() -> List[str]:
    assert os.path.exists(FEATURES_JSON), f"Missing {FEATURES_JSON}"
    with open(FEATURES_JSON, "r") as f:
        meta = json.load(f)
    features = meta.get("feature_sets", {}).get("medium", [])
    assert len(features) >= 120, "Insufficient medium features"
    return features

def sample_dataset_slice(
    val_path: str,
    target_name: str,
    selected_features: List[str],
    n_eras: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert os.path.exists(val_path), f"Validation file missing: {val_path}"
    assert len(selected_features) > 0, "No features provided"
    
    cols = ["era", target_name] + selected_features
    df = pd.read_parquet(val_path, columns=cols)
    unique_eras = df["era"].unique()
    chosen_eras = np.random.choice(unique_eras, min(n_eras, len(unique_eras)), replace=False)
    sample_df = df[df["era"].isin(chosen_eras)].dropna(subset=[target_name])
    
    feats = np.ascontiguousarray(sample_df[selected_features].values, dtype=np.float32)
    targs = np.ascontiguousarray(sample_df[target_name].values, dtype=np.float32)
    eras = np.ascontiguousarray(sample_df["era"].values)
    return feats, targs, eras

def run_single_synthesis_round(
    target_name: str,
    all_features: List[str],
    round_idx: int,
    pop_size: int = 40,
    generations: int = 5,
    min_sharpe: float = 0.72
) -> int:
    assert len(all_features) >= 100, "Need at least 100 features"
    assert pop_size >= 10, "Population size must be at least 10"
    
    val_path = os.path.join(DATA_DIR, "validation.parquet")
    n_sub = min(120, len(all_features))
    selected = list(np.random.choice(all_features, n_sub, replace=False))
    
    feats, targs, eras = sample_dataset_slice(val_path, target_name, selected, n_eras=12)
    filter_gate = TriHurdleFilter(min_sharpe=min_sharpe, max_factor_corr=0.15, min_positive_era_ratio=0.60)
    
    synth = GeneticSynthesizer(
        num_features=len(selected),
        pop_size=pop_size,
        tournament_size=3,
        mutation_rate=0.35,
        crossover_rate=0.65,
        max_depth=4,
        vault_path=CHIMERA_VAULT_PATH,
        filter_gate=filter_gate
    )
    
    vault_before = len(synth.vault.entries)
    print(f"[*] Round {round_idx}: Mining {len(feats):,} rows on {target_name}...")
    synth.evolve_on_dataset(feats, targs, eras, generations=generations)
    vault_after = len(synth.vault.entries)
    
    new_found = vault_after - vault_before
    print(f"[+] Round {round_idx} Complete. New Alphas Admitted: {new_found} (Total: {vault_after})")
    return new_found

def main():
    start_time = time.time()
    all_features = load_features_metadata()
    targets = ["target_cyrusd_60", "target_jeremy_60", "target_victor_60", "target_xerxes_60"]
    assert len(targets) == 4, "Must configure exactly 4 targets"
    
    total_new = 0
    for idx, target in enumerate(targets, 1):
        found = run_single_synthesis_round(
            target_name=target,
            all_features=all_features,
            round_idx=idx,
            pop_size=45,
            generations=6,
            min_sharpe=0.72
        )
        total_new += found
        
    elapsed = time.time() - start_time
    print(f"[+] All rounds finished in {elapsed:.1f}s. Total new alphas discovered: {total_new}")

if __name__ == "__main__":
    main()
