#!/usr/bin/env python3
"""
Chimera Alpha Mining CLI.
Mines non-linear, orthogonal symbolic alpha formulas on Apple Silicon M5 Pro
using validation eras from data/validation.parquet.
"""
import os
import json
import time
import argparse
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from typing import Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter
from config import DATA_DIR, FEATURES_JSON, CHIMERA_VAULT_PATH

def load_mining_dataset(
    n_eras: int = 15,
    max_features: int = 150,
    target_name: str = "target_cyrusd_60",
    random_sample: bool = False,
    seed: Optional[int] = None
):
    assert n_eras > 0, "n_eras must be positive"
    assert max_features > 0, "max_features must be positive"
    if seed is not None:
        np.random.seed(seed)

    print(f"[*] Loading feature metadata from {FEATURES_JSON}...")
    with open(FEATURES_JSON, "r") as f:
        feat_meta = json.load(f)
    medium_features = feat_meta.get("feature_sets", {}).get("medium", [])
    assert len(medium_features) > 0, "No medium features found"

    if random_sample:
        selected_features = list(np.random.choice(medium_features, min(max_features, len(medium_features)), replace=False))
    else:
        selected_features = medium_features[:max_features]

    val_path = os.path.join(DATA_DIR, "validation.parquet")
    assert os.path.exists(val_path), f"Missing validation.parquet at {val_path}"

    print(f"[*] Reading {n_eras} eras from {val_path}...")
    schema_names = pq.read_schema(val_path).names
    actual_target = target_name if target_name in schema_names else "target"

    cols_to_load = ["era", actual_target] + selected_features
    df = pd.read_parquet(val_path, columns=cols_to_load)
    
    unique_eras = df["era"].unique()
    if random_sample:
        chosen_eras = np.random.choice(unique_eras, min(n_eras, len(unique_eras)), replace=False)
    else:
        step = max(1, len(unique_eras) // n_eras)
        chosen_eras = unique_eras[::step][:n_eras]
    
    sample_df = df[df["era"].isin(chosen_eras)].copy()
    sample_df = sample_df.dropna(subset=[actual_target])
    
    print(f"[*] Dataset ready: {len(sample_df):,} rows across {len(chosen_eras)} eras ({len(selected_features)} features, target={actual_target})")
    
    features_matrix = np.ascontiguousarray(sample_df[selected_features].values, dtype=np.float32)
    target_vector = np.ascontiguousarray(sample_df[actual_target].values, dtype=np.float32)
    eras_vector = np.ascontiguousarray(sample_df["era"].values)
    
    return features_matrix, target_vector, eras_vector, selected_features, actual_target

def main():
    parser = argparse.ArgumentParser(description="Chimera Symbolic Alpha Miner")
    parser.add_argument("--eras", type=int, default=12, help="Number of validation eras to evaluate")
    parser.add_argument("--features", type=int, default=100, help="Number of base features to sample")
    parser.add_argument("--generations", type=int, default=6, help="Generations per evolutionary run")
    parser.add_argument("--pop-size", type=int, default=40, help="Population size per generation")
    parser.add_argument("--target", type=str, default="target_cyrusd_60", help="Target column to optimize")
    parser.add_argument("--min-sharpe", type=float, default=0.75, help="Minimum per-era Sharpe")
    parser.add_argument("--max-corr", type=float, default=0.15, help="Maximum correlation to any base feature")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample features and eras")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    start_t = time.time()
    feats, targets, eras, feat_names, target_used = load_mining_dataset(
        n_eras=args.eras,
        max_features=args.features,
        target_name=args.target,
        random_sample=args.random_sample,
        seed=args.seed
    )

    filter_gate = TriHurdleFilter(
        min_sharpe=args.min_sharpe,
        max_factor_corr=args.max_corr,
        min_positive_era_ratio=0.60
    )

    synthesizer = GeneticSynthesizer(
        num_features=len(feat_names),
        pop_size=args.pop_size,
        tournament_size=3,
        mutation_rate=0.35,
        crossover_rate=0.65,
        max_depth=4,
        vault_path=CHIMERA_VAULT_PATH,
        filter_gate=filter_gate
    )

    print(f"[*] Commencing Genetic Synthesis ({args.generations} generations, pop={args.pop_size})...")
    vault = synthesizer.evolve_on_dataset(feats, targets, eras, generations=args.generations)
    
    elapsed = time.time() - start_t
    print(f"[+] Synthesis Complete in {elapsed:.1f}s. Total Alphas in Vault: {len(vault.entries)}")
    for i, entry in enumerate(vault.entries):
        print(f"    - {entry.name}: {entry.formula} | Sharpe: {entry.sharpe:.3f}, Corr: {entry.mean_corr:.4f}, PosEras: {entry.positive_era_ratio:.1%}")

if __name__ == "__main__":
    main()
