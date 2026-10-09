"""
Tests for scripts/buyer_quick_submit.py.
Validates turnkey submission workflow, file validation, and Power of 10 safety invariants.
"""
import os
import pytest
import pandas as pd
import numpy as np
from scripts.buyer_quick_submit import (
    validate_prediction_file,
    blend_predictions,
    audit_safety_invariants_local,
)


def test_validate_prediction_file_valid(tmp_path):
    """Verify validation passes for properly formatted prediction CSV."""
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(100)],
        "prediction": np.linspace(0.01, 0.99, 100),
    })
    csv_file = tmp_path / "predictions_cypherpole.csv"
    df.to_csv(csv_file, index=False)

    valid, msg = validate_prediction_file(str(csv_file))
    assert valid is True, "Valid CSV must pass"
    assert "VALID" in msg, "Must indicate valid status"


def test_validate_prediction_file_invalid_columns(tmp_path):
    """Verify validation fails when required prediction column is missing."""
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(50)],
        "wrong_col": np.linspace(0.1, 0.9, 50),
    })
    csv_file = tmp_path / "bad.csv"
    df.to_csv(csv_file, index=False)

    valid, msg = validate_prediction_file(str(csv_file))
    assert valid is False, "Invalid columns must fail validation"


def test_blend_predictions():
    """Verify convex blending between user predictions and cypherpole predictions."""
    df_user = pd.DataFrame({
        "id": ["a", "b", "c"],
        "prediction": [0.2, 0.5, 0.8],
    })
    df_cypher = pd.DataFrame({
        "id": ["a", "b", "c"],
        "prediction": [0.4, 0.6, 0.7],
    })
    blended = blend_predictions(df_user, df_cypher, cypher_weight=0.30)
    assert len(blended) == 3, "Length must match"
    assert "prediction" in blended.columns, "Must have prediction column"
    assert 0.0 <= blended["prediction"].min() <= blended["prediction"].max() <= 1.0


def test_buyer_quick_submit_power_of_10():
    """Verify Holzmann Power of 10 invariants on buyer_quick_submit.py."""
    passed = audit_safety_invariants_local()
    assert passed is True, "Power of 10 invariants must pass"
