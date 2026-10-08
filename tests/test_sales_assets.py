"""
Unit and invariant tests for Sales Assets and Storefront Marketing modules.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import ast
import pytest
from scripts.update_storefront_copy import build_storefront_payload
from scripts.generate_sales_assets import generate_comparative_alpha_chart


def test_storefront_payload_builder():
    """Verify storefront payload contains coupon code and valid option specs."""
    payload = build_storefront_payload("cypherpole", 1216, 5.0, 1.5)
    assert isinstance(payload, dict)
    assert "description" in payload
    assert "CYPHER100" in payload["description"]
    assert "options" in payload
    assert len(payload["options"]) == 2
    
    trial_opt = next(o for o in payload["options"] if o["quantity"] == 1)
    assert trial_opt["price"] == 1.5
    assert "CYPHER100" in trial_opt["description"]


def test_generate_sales_chart_creates_image(tmp_path):
    """Verify sales chart generator creates valid PNG image file."""
    output_png = tmp_path / "test_alpha_chart.png"
    result_path = generate_comparative_alpha_chart(str(output_png))
    
    assert os.path.exists(result_path)
    assert result_path == str(output_png)
    assert os.path.getsize(result_path) > 10_000  # Non-trivial image payload


def test_power_of_10_invariants():
    """Verify Holzmann Power of 10 rules across sales marketing scripts."""
    target_scripts = [
        "scripts/update_storefront_copy.py",
        "scripts/generate_sales_assets.py",
    ]
    for script in target_scripts:
        assert os.path.exists(script), f"File {script} must exist"
        with open(script, "r") as f:
            tree = ast.parse(f.read())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(funcs) >= 2, f"{script} must contain at least 2 functions"
        for fn in funcs:
            lines = fn.end_lineno - fn.lineno + 1
            asserts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
            assert lines <= 60, f"Function {fn.name} in {script} exceeds 60 lines ({lines})"
            assert asserts >= 2, f"Function {fn.name} in {script} has fewer than 2 asserts ({asserts})"
