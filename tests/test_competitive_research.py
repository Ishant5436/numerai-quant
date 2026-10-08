"""
Unit and Invariant Tests for Competitive Alpha Analysis Engine.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import ast
import os
import pytest
import numpy as np


def test_compute_alpha_differential():
    """Verify calculation of alpha excess yield ratio against platform benchmark."""
    from scripts.competitive_analysis import compute_alpha_differential

    cypherpole_corj = 0.0607
    benchmark_corj = 0.0192

    res = compute_alpha_differential(cypherpole_corj, benchmark_corj)
    assert isinstance(res, dict), "Result must be a dict"
    assert "ratio" in res and "outperformance_pct" in res, "Missing required keys"
    assert res["ratio"] > 3.0, f"Expected ratio > 3.0x, got {res['ratio']:.2f}"
    assert res["outperformance_pct"] > 200.0, (
        f"Expected > 200% outperformance, got {res['outperformance_pct']:.1f}%"
    )


def test_generate_comparative_staker_matrix():
    """Verify net profit calculation shows positive excess return across all capital tiers."""
    from scripts.competitive_analysis import generate_comparative_staker_matrix

    staker_capital = [50.0, 100.0, 250.0]
    cypherpole_corj = 0.0607
    benchmark_corj = 0.0192
    cypherpole_price = 5.0
    benchmark_price = 0.8  # 4 * 0.2 NMR

    matrix = generate_comparative_staker_matrix(
        staker_capital, cypherpole_corj, benchmark_corj, cypherpole_price, benchmark_price
    )
    assert len(matrix) == 3, "Matrix must have 3 entries"
    for row in matrix:
        assert row["cypherpole_net_profit"] > 0.0, "Cypherpole must be profitable"
        assert row["excess_profit_nmr"] > 0.0, (
            "Cypherpole must generate higher net profit than benchmark"
        )
        assert row["capital"] in staker_capital, "Capital row mismatch"


def test_format_conversion_comparison_markdown():
    """Verify markdown table generation produces complete and valid markdown."""
    from scripts.competitive_analysis import (
        generate_comparative_staker_matrix,
        format_conversion_comparison_markdown,
    )

    matrix = generate_comparative_staker_matrix(
        [50.0, 100.0], 0.0607, 0.0192, 5.0, 0.8
    )
    md = format_conversion_comparison_markdown(matrix)
    assert isinstance(md, str), "Markdown output must be a string"
    assert "| Capital |" in md or "| Staker Capital |" in md, "Missing header"
    assert "CYPHERPOLE" in md, "Missing brand identifier"


def test_power_of_10_invariants():
    """Verify that competitive_analysis.py adheres to Holzmann Power of 10 rules."""
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "competitive_analysis.py"
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
