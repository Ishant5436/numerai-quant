"""
Competitive Alpha Analysis and Staker Economics Engine.
Compares model predictive power and net profit against Numerbay platform benchmarks.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2, bounded loops.
"""
from typing import Dict, Any, List
import numpy as np


def compute_alpha_differential(
    model_corj: float, benchmark_corj: float
) -> Dict[str, float]:
    """Computes alpha ratio and percentage outperformance relative to benchmark."""
    assert isinstance(model_corj, (int, float)) and model_corj > 0.0, "Invalid model_corj"
    assert isinstance(benchmark_corj, (int, float)) and benchmark_corj > 0.0, "Invalid benchmark_corj"

    m_val = float(model_corj)
    b_val = float(benchmark_corj)
    ratio = m_val / b_val
    outperformance_pct = (ratio - 1.0) * 100.0

    res = {
        "model_corj": round(m_val, 4),
        "benchmark_corj": round(b_val, 4),
        "ratio": round(ratio, 2),
        "outperformance_pct": round(outperformance_pct, 1),
    }
    assert res["ratio"] > 0.0, "Ratio must be positive"
    return res


def generate_comparative_staker_matrix(
    staker_capital_list: List[float],
    cypherpole_corj: float,
    benchmark_corj: float,
    cypherpole_price_monthly: float,
    benchmark_price_monthly: float,
) -> List[Dict[str, Any]]:
    """Generates comparative net profit matrices across capital tiers."""
    assert isinstance(staker_capital_list, list), "Capital tiers must be a list"
    assert len(staker_capital_list) > 0, "Capital list cannot be empty"

    matrix = []
    # Bounded iteration
    for cap in staker_capital_list[:50]:
        c_val = float(cap)
        # Expected monthly return = 4 rounds * capital * correlation
        c_gross = 4.0 * c_val * float(cypherpole_corj)
        b_gross = 4.0 * c_val * float(benchmark_corj)

        c_net = c_gross - float(cypherpole_price_monthly)
        b_net = b_gross - float(benchmark_price_monthly)
        excess = c_net - b_net

        matrix.append(
            {
                "capital": c_val,
                "cypherpole_gross": round(c_gross, 2),
                "cypherpole_net_profit": round(c_net, 2),
                "benchmark_gross": round(b_gross, 2),
                "benchmark_net_profit": round(b_net, 2),
                "excess_profit_nmr": round(excess, 2),
            }
        )

    assert len(matrix) == len(staker_capital_list), "Matrix length mismatch"
    return matrix


def format_conversion_comparison_markdown(matrix: List[Dict[str, Any]]) -> str:
    """Formats matrix into GitHub Markdown table for conversion documentation."""
    assert isinstance(matrix, list), "Matrix must be a list"
    assert len(matrix) > 0, "Matrix cannot be empty"

    lines = [
        "| Staker Capital | Benchmark Net Profit | CYPHERPOLE Net Profit | Excess Profit Generated |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for row in matrix[:50]:
        lines.append(
            f"| **{row['capital']:.0f} NMR** | {row['benchmark_net_profit']:.2f} NMR | "
            f"**{row['cypherpole_net_profit']:.2f} NMR** | **+{row['excess_profit_nmr']:.2f} NMR** |"
        )

    table_md = "\n".join(lines)
    assert len(table_md) > 50, "Output markdown table too short"
    return table_md


if __name__ == "__main__":
    diff = compute_alpha_differential(0.0607, 0.0192)
    print("=" * 60)
    print("Competitive Alpha Analysis: CYPHERPOLE vs Top Seller Benchmark")
    print("=" * 60)
    print(f"Alpha Multiple: {diff['ratio']}x (+{diff['outperformance_pct']}%)")

    caps = [50.0, 100.0, 250.0, 500.0]
    mat = generate_comparative_staker_matrix(caps, 0.0607, 0.0192, 5.0, 0.8)
    print(format_conversion_comparison_markdown(mat))
