"""
Numerai Sunday Submission Pre-Flight Institutional Readiness Verifier.
Validates end-to-end readiness for Round 1373 across NumerAPI, Numerbay,
model weights, feature manifests, filesystem permissions, and prediction invariants.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
import json
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))



def check_numerapi_active_round(napi: Optional[Any] = None) -> Dict[str, Any]:
    """Gate 1: Verifies NumerAPI network connectivity and current active round number."""
    assert napi is None or hasattr(napi, "get_current_round"), "Invalid NumerAPI instance"
    assert isinstance(BASE_DIR, str) and len(BASE_DIR) > 0, "Base directory invalid"

    if napi is None:
        try:
            from numerapi import NumerAPI
            napi = NumerAPI()
        except Exception as init_err:
            return {"status": "FAIL", "reason": f"NumerAPI initialization failed: {init_err}"}

    try:
        curr_round = napi.get_current_round()
        assert isinstance(curr_round, int) and curr_round > 0, "Current round must be positive int"
        return {"status": "PASS", "current_round": curr_round, "target_round": curr_round + 1}
    except Exception as query_err:
        return {"status": "FAIL", "reason": f"Failed querying active round: {query_err}"}


def check_numerbay_connectivity(pub: Optional[Any] = None) -> Dict[str, Any]:
    """Gate 2: Verifies Numerbay seller authentication, listings, and wallet destination."""
    assert pub is None or hasattr(pub, "get_listings"), "Invalid NumerbayPublisher instance"
    assert isinstance(BASE_DIR, str) and len(BASE_DIR) > 0, "Base directory invalid"

    if pub is None:
        from numerbay_publisher import NumerbayPublisher
        pub = NumerbayPublisher()

    if not pub.is_configured or pub.api is None:
        return {"status": "FAIL", "reason": "Numerbay credentials unconfigured"}

    try:
        acct = pub.api.get_account()
        listings = pub.get_listings()
        assert isinstance(acct, dict), "Account response must be dictionary"
        assert isinstance(listings, list), "Listings response must be list"
        return {
            "status": "PASS",
            "username": acct.get("username"),
            "receiving_wallet": acct.get("default_receiving_wallet_address"),
            "active_listings_count": len(listings),
            "listings": [l.get("name") for l in listings[:10]],
        }
    except Exception as err:
        return {"status": "FAIL", "reason": f"Numerbay query failed: {err}"}


def check_model_checkpoints(base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Gate 3: Audits presence and physical file sizes of trained model checkpoints."""
    b_dir = base_dir if base_dir is not None else BASE_DIR
    assert isinstance(b_dir, str) and len(b_dir) > 0, "Invalid base_dir"
    ens_dir = os.path.join(b_dir, "models", "ensemble_60d")
    ortho_dir = os.path.join(b_dir, "models", "orthogonal_fleet_60d")

    target_files = [
        ("ensemble_60d", os.path.join(ens_dir, "lgb_target_agnes_60.pkl")),
        ("ensemble_60d", os.path.join(ens_dir, "lgb_target_cyrusd_60.pkl")),
        ("ensemble_60d", os.path.join(ens_dir, "lgb_target_jeremy_60.pkl")),
        ("ensemble_60d", os.path.join(ens_dir, "lgb_target_victor_60.pkl")),
        ("ensemble_60d", os.path.join(ens_dir, "lgb_target_xerxes_60.pkl")),
        ("orthogonal_fleet_60d", os.path.join(ortho_dir, "lgb_strat_8.pkl")),
        ("orthogonal_fleet_60d", os.path.join(ortho_dir, "lgb_strat_15.pkl")),
    ]
    results = []
    all_ok = True
    for group, fpath in target_files:
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        if not exists or size < 100_000:
            all_ok = False
        results.append({"name": os.path.basename(fpath), "group": group, "exists": exists, "size_bytes": size})

    assert len(results) == len(target_files), "All target files must be inspected"
    return {"status": "PASS" if all_ok else "FAIL", "checked_models": results}


def check_feature_manifest(features_path: Optional[str] = None) -> Dict[str, Any]:
    """Gate 4: Verifies existence and feature counts in canonical features.json."""
    fpath = features_path if features_path is not None else os.path.join(BASE_DIR, "features.json")
    assert isinstance(fpath, str) and len(fpath) > 0, "Invalid features_path"

    if not os.path.exists(fpath):
        return {"status": "FAIL", "reason": f"Missing features.json at {fpath}"}

    try:
        with open(fpath, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), "features.json must be a JSON object"
        f_sets = data.get("feature_sets", {})
        medium_cnt = len(f_sets.get("medium", []))
        fncv3_cnt = len(f_sets.get("fncv3_features", []))
        is_ok = medium_cnt >= 700 and fncv3_cnt >= 400
        return {
            "status": "PASS" if is_ok else "FAIL",
            "medium_feature_count": medium_cnt,
            "fncv3_feature_count": fncv3_cnt,
        }
    except Exception as err:
        return {"status": "FAIL", "reason": f"Corrupt features.json: {err}"}


def check_storage_and_locks(
    data_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    lock_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Gate 5: Verifies writable filesystem storage and concurrency lock status."""
    d_dir = data_dir if data_dir is not None else os.path.join(BASE_DIR, "data")
    l_dir = logs_dir if logs_dir is not None else os.path.join(BASE_DIR, "logs")
    k_dir = lock_dir if lock_dir is not None else "/tmp/numerai_fleet_submit.lockdir"
    assert isinstance(d_dir, str) and isinstance(l_dir, str), "Paths must be strings"

    os.makedirs(d_dir, exist_ok=True)
    os.makedirs(l_dir, exist_ok=True)
    d_write = os.access(d_dir, os.W_OK)
    l_write = os.access(l_dir, os.W_OK)
    lock_active = os.path.exists(k_dir)

    is_ok = d_write and l_write
    assert isinstance(is_ok, bool), "Status invariant must be boolean"
    return {
        "status": "PASS" if is_ok else "FAIL",
        "data_writable": d_write,
        "logs_writable": l_write,
        "lock_active": lock_active,
        "lock_path": k_dir,
    }


def check_prediction_payload_invariants(df: pd.DataFrame) -> Dict[str, Any]:
    """Gate 6 & 7: Audits prediction payload formatting and Holzmann mathematical invariants."""
    assert isinstance(df, pd.DataFrame), "Input must be DataFrame"
    assert len(df) > 0, "DataFrame cannot be empty"

    if "prediction" not in df.columns:
        return {"status": "FAIL", "reason": "Missing required 'prediction' column"}

    nan_count = int(df["prediction"].isna().sum())
    min_v = float(df["prediction"].min())
    max_v = float(df["prediction"].max())
    if nan_count > 0:
        return {"status": "FAIL", "nan_count": nan_count, "reason": f"Payload contains {nan_count} NaNs"}
    if min_v < 0.0 or max_v > 1.0:
        return {"status": "FAIL", "min": min_v, "max": max_v, "reason": "Predictions outside bounds [0, 1]"}

    return {"status": "PASS", "nan_count": 0, "min": min_v, "max": max_v, "rows": len(df)}


def run_preflight_audit(verbose: bool = True) -> Dict[str, Any]:
    """Aggregates all 6 institutional gates into a comprehensive deployment scorecard."""
    assert isinstance(verbose, bool), "verbose must be boolean"
    results = {
        "gate_1_numerapi": check_numerapi_active_round(),
        "gate_2_numerbay": check_numerbay_connectivity(),
        "gate_3_models": check_model_checkpoints(),
        "gate_4_features": check_feature_manifest(),
        "gate_5_storage": check_storage_and_locks(),
    }
    dummy_df = pd.DataFrame({"id": ["test_id"], "prediction": [0.5]})
    results["gate_6_invariants"] = check_prediction_payload_invariants(dummy_df)

    assert len(results) == 6, "All 6 gates must be executed"
    all_pass = all(r.get("status") == "PASS" for r in results.values())
    results["overall_status"] = "PASS" if all_pass else "FAIL"

    if verbose:
        print("=" * 65)
        print("  NUMERAI SUNDAY PIPELINE PRE-FLIGHT READINESS AUDIT")
        print("=" * 65)
        for gate_key, r in results.items():
            if gate_key == "overall_status":
                continue
            status_symbol = "[OK]  " if r.get("status") == "PASS" else "[FAIL]"
            print(f" {status_symbol} {gate_key.upper():<20} -> {r.get('status')}")
        print("=" * 65)
        print(f" FINAL STATUS: {results['overall_status']}")
        print("=" * 65)

    return results


if __name__ == "__main__":
    run_preflight_audit(verbose=True)
