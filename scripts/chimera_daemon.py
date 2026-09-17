#!/usr/bin/env python3
"""
Chimera Autonomous Background Alpha Mining Daemon.
Runs continuously on Apple Silicon at low process priority (nice 15),
rotating validation eras and factor families to discover non-linear alpha formulas.
"""
import os
import sys
import time
import json
import signal
import logging
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter
from config import DATA_DIR, FEATURES_JSON, CHIMERA_VAULT_PATH

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "chimera_daemon.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CHIMERA-DAEMON] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

RUNNING = True

def handle_signal(signum, frame):
    global RUNNING
    logging.info(f"Shutdown signal received ({signum}). Gracefully stopping daemon...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def run_daemon_mining_cycle(cycle_id: int):
    val_path = os.path.join(DATA_DIR, "validation.parquet")
    if not os.path.exists(val_path):
        logging.error(f"Validation dataset missing at {val_path}")
        return

    with open(FEATURES_JSON, "r") as f:
        feat_meta = json.load(f)
    medium_features = feat_meta.get("feature_sets", {}).get("medium", [])
    
    # Select random subset of 120 features for this cycle to explore diverse factor interactions
    n_sub_features = 120
    selected_features = list(np.random.choice(medium_features, n_sub_features, replace=False))

    schema_names = pq.read_schema(val_path).names
    targets = ["target_cyrusd_60", "target_jeremy_60", "target_victor_60", "target_xerxes_60"]
    available_targets = [t for t in targets if t in schema_names] or ["target"]
    target_name = np.random.choice(available_targets)

    cols_to_load = ["era", target_name] + selected_features
    df = pd.read_parquet(val_path, columns=cols_to_load)
    
    unique_eras = df["era"].unique()
    # Random sample of 12 eras
    chosen_eras = np.random.choice(unique_eras, min(12, len(unique_eras)), replace=False)
    sample_df = df[df["era"].isin(chosen_eras)].dropna(subset=[target_name])

    features_matrix = np.ascontiguousarray(sample_df[selected_features].values, dtype=np.float32)
    target_vector = np.ascontiguousarray(sample_df[target_name].values, dtype=np.float32)
    eras_vector = np.ascontiguousarray(sample_df["era"].values)

    filter_gate = TriHurdleFilter(min_sharpe=0.75, max_factor_corr=0.15, min_positive_era_ratio=0.60)
    synth = GeneticSynthesizer(
        num_features=len(selected_features),
        pop_size=35,
        tournament_size=3,
        mutation_rate=0.35,
        crossover_rate=0.65,
        max_depth=4,
        vault_path=CHIMERA_VAULT_PATH,
        filter_gate=filter_gate
    )

    logging.info(f"Cycle {cycle_id}: Mining {len(sample_df):,} rows on {target_name} across {len(chosen_eras)} eras...")
    vault = synth.evolve_on_dataset(features_matrix, target_vector, eras_vector, generations=5)
    logging.info(f"Cycle {cycle_id} complete. Vault contains {len(vault.entries)} verified alphas.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Interval seconds between mining rounds")
    args = parser.parse_args()

    # Set low process priority so foreground tasks are never blocked
    try:
        os.nice(15)
        logging.info("Process priority set to nice 15 (background idle execution).")
    except Exception as e:
        logging.warning(f"Could not set nice level: {e}")

    logging.info("Chimera Alpha Mining Daemon started.")
    cycle = 1
    while RUNNING:
        try:
            run_daemon_mining_cycle(cycle)
            cycle += 1
        except Exception as e:
            logging.error(f"Error in cycle {cycle}: {e}", exc_info=True)

        if not RUNNING:
            break
        logging.info(f"Sleeping for {args.interval}s until next cycle...")
        for _ in range(args.interval):
            if not RUNNING:
                break
            time.sleep(1)

    logging.info("Chimera Daemon stopped gracefully.")

if __name__ == "__main__":
    main()
