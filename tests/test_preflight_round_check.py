"""
Unit and invariant tests for the Pre-Flight Round Check module.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import ast
import os
import pytest
import pandas as pd
import numpy as np
from scripts.preflight_round_check import (
    check_model_checkpoints,
    check_feature_manifest,
    check_storage_and_locks,
    check_prediction_payload_invariants,
    run_preflight_audit,
)


def test_check_model_checkpoints_exist():
    """Verify that all core flagship model files are verified by Gate 3."""
    res = check_model_checkpoints()
    assert isinstance(res, dict)
    assert res["status"] in ["PASS", "FAIL"]
    assert "checked_models" in res
    assert len(res["checked_models"]) >= 5
    for m in res["checked_models"]:
        assert m["exists"] is True
        assert m["size_bytes"] > 500_000


def test_check_feature_manifest():
    """Verify that feature manifest exists and contains required feature keys."""
    res = check_feature_manifest()
    assert isinstance(res, dict)
    assert res["status"] == "PASS"
    assert res["medium_feature_count"] > 600
    assert res["fncv3_feature_count"] > 300


def test_check_storage_and_locks(tmp_path):
    """Verify that storage and locks check validates directories and permissions."""
    test_data = tmp_path / "data"
    test_logs = tmp_path / "logs"
    test_data.mkdir()
    test_logs.mkdir()

    res = check_storage_and_locks(data_dir=str(test_data), logs_dir=str(test_logs))
    assert isinstance(res, dict)
    assert res["status"] == "PASS"
    assert res["data_writable"] is True
    assert res["logs_writable"] is True


def test_check_prediction_payload_invariants():
    """Verify Gate 7 catches invalid predictions and passes valid uniform payloads."""
    valid_df = pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0.1, 0.5, 0.9]})
    valid_res = check_prediction_payload_invariants(valid_df)
    assert valid_res["status"] == "PASS"
    assert valid_res["nan_count"] == 0

    invalid_df = pd.DataFrame({"id": ["a", "b"], "prediction": [-0.1, 1.2]})
    invalid_res = check_prediction_payload_invariants(invalid_df)
    assert invalid_res["status"] == "FAIL"
    assert "bounds" in invalid_res["reason"].lower()


def test_power_of_10_invariants():
    """Verify Holzmann Power of 10 invariants across preflight_round_check.py."""
    filepath = "scripts/preflight_round_check.py"
    assert os.path.exists(filepath), f"File {filepath} must exist"
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) >= 4, "Must implement at least 4 modular functions"
    for fn in funcs:
        lines = fn.end_lineno - fn.lineno + 1
        asserts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
        assert lines <= 60, f"Function {fn.name} exceeds 60 lines ({lines})"
        assert asserts >= 2, f"Function {fn.name} has fewer than 2 asserts ({asserts})"
