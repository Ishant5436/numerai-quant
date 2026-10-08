"""
Unit and invariant tests for Staker Inbound Billboard and Portfolio Synergy modules.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import ast
import pytest
import numpy as np
from scripts.sync_leaderboard_billboard import build_billboard_specs
from scripts.staker_synergy_analyzer import simulate_portfolio_blend, format_synergy_markdown_table


def test_build_billboard_payloads():
    """Verify billboard specs contain valid link, bio text, and trial coupon."""
    specs = build_billboard_specs("cypherpole", "test-uuid-1234")
    assert isinstance(specs, dict)
    assert specs["model_id"] == "test-uuid-1234"
    assert "CYPHER100" in specs["bio"]
    assert "https://numerbay.ai" in specs["link_url"]
    assert len(specs["link_text"]) > 0


def test_simulate_portfolio_blending():
    """Verify portfolio synergy math proves Sharpe improvement and diversification."""
    np.random.seed(42)
    # Staker baseline with moderate correlation and noise
    staker_corrs = np.random.normal(0.015, 0.025, 60)
    # Cypherpole orthogonal high alpha
    cypher_corrs = np.random.normal(0.060, 0.020, 60)

    result = simulate_portfolio_blend(staker_corrs, cypher_corrs, blend_weight=0.30)
    assert isinstance(result, dict)
    assert result["blended_sharpe"] > result["baseline_sharpe"]
    assert result["blended_mean_corr"] > result["baseline_mean_corr"]
    assert result["sharpe_improvement_pct"] > 0.0


def test_format_synergy_markdown_table():
    """Verify synergy markdown table generates clean tabular text."""
    metrics = {
        "baseline_mean_corr": 0.015,
        "blended_mean_corr": 0.028,
        "baseline_sharpe": 0.60,
        "blended_sharpe": 1.40,
        "sharpe_improvement_pct": 133.3,
    }
    table = format_synergy_markdown_table(metrics)
    assert isinstance(table, str)
    assert "Baseline" in table
    assert "Blended" in table
    assert "133.3%" in table


def test_power_of_10_invariants():
    """Verify Holzmann Power of 10 rules across outreach scripts."""
    target_scripts = [
        "scripts/sync_leaderboard_billboard.py",
        "scripts/staker_synergy_analyzer.py",
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
