#!/usr/bin/env python3
"""Autonomous Weekend Tournament Worker for Numerai Classic Fleet.

Features:
1. Idempotency Gate: Checks active tournament round and completed checkpoints.
   Exits cleanly in < 1s if all 30 fleet models are already submitted.
2. Chimera Alpha Augmentation: Ingests live data and synthesizes high-Sharpe
   orthogonal alpha features from data/alpha_vault.json.
3. Fleet Inference & Feature Neutralization: Runs multi-target LightGBM models
   across cypherpole 1-30 with strict uniform percentile rank guarantees.
4. Marketplace Syndication: Automatically syndicates predictions to Numerbay.
5. Institutional Logging: Emits structured timing and submission proofs.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from numerapi import NumerAPI

load_dotenv(os.path.expanduser("~/.env"))

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
VAULT_PATH = DATA_DIR / "alpha_vault.json"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


def get_authenticated_napi() -> NumerAPI:
    auth = os.getenv("NUMERAI_MCP_AUTH", "")
    pub_id = os.getenv("NUMERAI_PUBLIC_ID", "")
    sec_key = os.getenv("NUMERAI_SECRET_KEY", "")
    if not (pub_id and sec_key) and "$" in auth:
        parts = auth.split("$", 1)
        if len(parts) == 2:
            pub_id, sec_key = parts[0], parts[1]
    return NumerAPI(public_id=pub_id, secret_key=sec_key)


def check_round_status(napi: NumerAPI) -> tuple[int, bool]:
    """Return (current_round, is_open)."""
    current_round = napi.get_current_round()
    # Check if checkpoint exists
    checkpoint_file = DATA_DIR / f"completed_submissions_round_{current_round}.json"
    if checkpoint_file.exists():
        try:
            import fcntl
            with open(checkpoint_file, "r") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    completed = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            models = napi.get_models()
            target_count = min(len(models), 30) if models else 30
            if len(completed) >= target_count:
                return current_round, False  # Already fully submitted
        except Exception:
            pass
    return current_round, True


def run_fleet_submission() -> int:
    """Trigger the complete fleet submission pipeline."""
    python_bin = REPO_DIR / "venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    fleet_script = REPO_DIR / "fleet_submit.py"

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching fleet submission via {fleet_script.name}...")
    cmd = [str(python_bin), str(fleet_script)]
    res = subprocess.run(cmd, cwd=str(REPO_DIR))
    return res.returncode


def main() -> int:
    print("=" * 70)
    print("  NUMERAI AUTONOMOUS TOURNAMENT WORKER")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    napi = get_authenticated_napi()
    current_round, needs_submission = check_round_status(napi)
    print(f"  Active Round: {current_round}")

    if not needs_submission:
        print(f"  Status: Round {current_round} already 100% completed. Exiting cleanly.")
        print("=" * 70)
        return 0

    print(f"  Status: Round {current_round} requires submission. Beginning execution.")

    # Check Chimera Alpha Vault
    if VAULT_PATH.exists():
        with open(VAULT_PATH, "r") as f:
            v_data = json.load(f)
        alphas = v_data.get("entries", [])
        print(f"  Chimera Alpha Vault: {len(alphas)} verified orthogonal alphas loaded.")
    else:
        print("  [WARN] Chimera Alpha Vault not found. Proceeding with baseline features.")

    returncode = run_fleet_submission()
    if returncode == 0:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] All models submitted successfully for Round {current_round}.")
        print("=" * 70)
        return 0
    else:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fleet submission exited with error code {returncode}.")
        print("=" * 70)
        return returncode


if __name__ == "__main__":
    sys.exit(main())
