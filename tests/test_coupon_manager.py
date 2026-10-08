"""
Unit and Invariant Tests for Numerbay Promotional Coupon Manager.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import ast
import os
import pytest


def test_validate_coupon_payload():
    """Verify coupon validation rejects out-of-bounds parameters deterministically."""
    from scripts.numerbay_coupon_manager import validate_coupon_payload

    valid_spec = {
        "code": "CYPHER100",
        "applicability": "products",
        "applicable_product_ids": [1216],
        "discount_percent": 100,
        "quantity_total": 5,
        "discount_mode": "percent",
    }
    is_valid, msg = validate_coupon_payload(valid_spec)
    assert is_valid is True, f"Expected valid payload, got error: {msg}"
    assert len(msg) > 0, "Status message cannot be empty"

    invalid_spec = valid_spec.copy()
    invalid_spec["discount_percent"] = 150  # Out of bounds
    is_bad, _ = validate_coupon_payload(invalid_spec)
    assert is_bad is False, "Should reject discount > 100%"


def test_build_trial_coupon_spec():
    """Verify trial coupon builder produces conforming schema for product 1216."""
    from scripts.numerbay_coupon_manager import build_trial_coupon_spec

    spec = build_trial_coupon_spec(code="CYPHER100", product_id=1216, quantity=5)
    assert isinstance(spec, dict), "Spec must be a dictionary"
    assert spec["code"] == "CYPHER100", "Code mismatch"
    assert spec["discount_percent"] == 100, "Must be 100% discount"
    assert spec["max_discount"] == 1.5, "Max discount must match single trial price"
    assert 1216 in spec["applicable_product_ids"], "Product ID 1216 must be included"
    assert spec["quantity_total"] == 5, "Total quantity must be 5"


def test_create_coupon_on_numerbay_mock():
    """Verify create coupon API call handles mock responses gracefully."""
    from scripts.numerbay_coupon_manager import (
        build_trial_coupon_spec,
        create_coupon_on_numerbay,
    )

    class MockPublisher:
        is_configured = True
        username = "cypherpole"

        class MockApi:
            token = "fake_token_123"

        api = MockApi()

    pub = MockPublisher()
    spec = build_trial_coupon_spec(code="CYPHER100", product_id=1216, quantity=5)
    res = create_coupon_on_numerbay(pub, spec, dry_run=True)
    assert isinstance(res, dict), "Result must be a dictionary"
    assert res["status"] == "DRY_RUN", "Dry run must return DRY_RUN status"
    assert res["code"] == "CYPHER100", "Result must return coupon code"


def test_power_of_10_invariants():
    """Verify that numerbay_coupon_manager.py adheres to Holzmann Power of 10 rules."""
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "numerbay_coupon_manager.py"
    )
    assert os.path.exists(module_path), f"Module not found at {module_path}"

    with open(module_path, "r") as f:
        tree = ast.parse(f.read())

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) >= 3, "Must contain at least 3 functions"

    for fn in functions:
        line_length = fn.end_lineno - fn.lineno + 1
        assert line_length <= 60, f"Function {fn.name} exceeds 60 lines ({line_length})"

        assert_count = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
        assert assert_count >= 2, (
            f"Function {fn.name} has only {assert_count} asserts (minimum 2)"
        )
