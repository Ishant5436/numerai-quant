#!/usr/bin/env python3
"""
Numerai Account Model Registration Engine: Strategies 16 through 30
Idempotently queries active models on the connected account and registers missing model slots.
Ensures zero secrets logged or echoed to console.
"""

import os
import sys
import time
from dotenv import load_dotenv
from numerapi import NumerAPI

load_dotenv(os.path.expanduser("~/.env"))

NEW_MODELS = [
    "cypherpole_bravo",      # Strat 16: Fundamental Value Spread
    "cypherpole_charlie",    # Strat 17: Low-Beta Defensive
    "cypherpole_delta",      # Strat 18: Pure Residual Alpha
    "cypherpole_echo",       # Strat 19: Mean Reversion Dynamic
    "cypherpole_ralph",      # Strat 20: Factor Momentum Velocity
    "cypherpole_rowan",      # Strat 21: High-Sharpe Quality
    "cypherpole_sam",        # Strat 22: Macro Tail Liquidity
    "cypherpole_tyler",      # Strat 23: Earnings Quality Momentum
    "cypherpole_waldo",      # Strat 24: Sentiment Divergence
    "cypherpole_victor",     # Strat 25: Volatility-Adjusted Alpha
    "cypherpole_claudia",    # Strat 26: Orthogonal Risk Parity
    "cypherpole_agnes",      # Strat 27: Orthogonal Residual Spread
    "cypherpole_caroline",   # Strat 28: Long-Horizon Growth Trend
    "cypherpole_ender",      # Strat 29: 60-Day Macro Core
    "cypherpole_supernova"   # Strat 30: Tri-Ensemble Meta-Blend
]

ADD_MODEL_MUTATION = """
mutation($name: String!, $tournament: Int!) {
    addModel(name: $name, tournament: $tournament) {
        id
        name
    }
}
"""


def main():
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")

    if "$" in auth and not (public_id and secret_key):
        public_id, secret_key = auth.split("$", 1)

    if not (public_id and secret_key):
        print("[ERROR] Numerai API credentials missing from environment.")
        sys.exit(1)

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    existing_models = napi.get_models()
    print(f"Current registered account models count: {len(existing_models)}")

    registered_count = 0
    skipped_count = 0
    failed_count = 0

    for model_name in NEW_MODELS:
        if model_name in existing_models:
            print(f"[EXISTS] Model '{model_name}' is already registered (ID: {existing_models[model_name]}).")
            skipped_count += 1
            continue

        print(f"[REGISTERING] Submitting registration for model '{model_name}'...")
        try:
            vars_dict = {"name": model_name, "tournament": 8}
            res = napi.raw_query(ADD_MODEL_MUTATION, vars_dict, authorization=True)
            if "errors" in res:
                print(f"[ERROR] Failed to register '{model_name}': {res['errors']}")
                failed_count += 1
            else:
                model_data = res.get("data", {}).get("addModel", {})
                new_id = model_data.get("id", "unknown")
                print(f"[SUCCESS] Registered '{model_name}' successfully (ID: {new_id}).")
                registered_count += 1
                time.sleep(1.0)
        except Exception as e:
            print(f"[EXCEPTION] Failed registering '{model_name}': {e}")
            failed_count += 1

    updated_models = napi.get_models()
    print("\n" + "=" * 60)
    print(f"Registration Summary: {registered_count} registered, {skipped_count} existing, {failed_count} failed.")
    print(f"Total active models on account: {len(updated_models)} / 30")
    print("=" * 60)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
