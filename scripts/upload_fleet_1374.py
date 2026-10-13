#!/usr/bin/env python3
"""
Upload Round 1374 Fleet Predictions to NumerAPI.
Executes upload of remaining fleet models with atomic checkpointing.
"""

import json
import os
import sys
import tempfile
import time
from typing import Set
from numerapi import NumerAPI


def load_checkpoint(checkpoint_path: str) -> Set[str]:
    """Load set of completed models from checkpoint."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    return set()


def save_checkpoint(checkpoint_path: str, completed: Set[str]) -> None:
    """Atomically save set of completed models."""
    dir_name = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tf:
        json.dump(sorted(list(completed)), tf)
        temp_name = tf.name
    os.replace(temp_name, checkpoint_path)


def main() -> int:
    try:
        napi = NumerAPI()
        acc = napi.get_account()
    except Exception as e:
        print(f"[FATAL] Failed to initialize NumerAPI: {e}")
        return 1
    status = acc.get("status")
    print(f"[INFO] Numerai Account: {acc.get('username')} | Status: {status}")
    if status != "VERIFIED":
        print(f"[ERROR] Account not verified (status: {status}). Aborting.")
        return 1

    current_round = napi.get_current_round()
    target_round = 1374
    print(f"[INFO] Current Round: {current_round} | Target Round: {target_round}")

    models = napi.get_models()
    print(f"[INFO] Found {len(models)} models on account.")

    checkpoint_path = os.path.join("data", f"completed_submissions_round_{target_round}.json")
    completed = load_checkpoint(checkpoint_path)
    # cypherpole was already uploaded
    completed.add("cypherpole")
    save_checkpoint(checkpoint_path, completed)
    print(f"[INFO] Initial completed models: {len(completed)}/25")

    success_count = len(completed)
    failed_models = []

    for model_name, model_id in sorted(models.items()):
        if model_name in completed:
            print(f"[SKIP] {model_name} already completed.")
            continue

        pred_file = os.path.join("data", f"predictions_{model_name}_round_{target_round}.csv")
        if not os.path.exists(pred_file):
            print(f"[ERROR] Prediction file missing for {model_name}: {pred_file}")
            failed_models.append(model_name)
            continue

        file_size = os.path.getsize(pred_file)
        print(f"[RUN] Uploading {model_name} ({file_size} bytes) -> model_id {model_id}...")

        attempt_success = False
        for attempt in range(1, 4):
            try:
                sub_id = napi.upload_predictions(pred_file, model_id=model_id)
                print(f"[OK] {model_name} submitted successfully! Submission ID: {sub_id}")
                completed.add(model_name)
                save_checkpoint(checkpoint_path, completed)
                success_count += 1
                attempt_success = True
                break
            except Exception as e:
                print(f"[WARN] Attempt {attempt}/3 failed for {model_name}: {e}")
                time.sleep(3.0 * attempt)

        if not attempt_success:
            failed_models.append(model_name)

        time.sleep(1.5)

    print("=" * 60)
    print(f"[SUMMARY] Total Completed: {success_count}/25 | Failed: {len(failed_models)}")
    if failed_models:
        print(f"[FAILED MODELS]: {failed_models}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
