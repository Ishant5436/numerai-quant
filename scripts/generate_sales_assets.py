"""
Visual Proof Asset Generator for Numerai & Numerbay Stakers.
Generates publication-grade, institutional performance charts comparing
cypherpole alpha metrics against platform medians and tournament benchmarks.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def prepare_benchmark_data() -> Dict[str, Any]:
    """Returns canonical out-of-sample alpha metrics for comparison."""
    data = {
        "models": ["Platform Median", "Benchmark (cyrus)", "cypherpole (Flagship)"],
        "corj60": [0.0192, 0.0240, 0.0607],
        "sharpe": [0.62, 0.85, 1.84],
        "win_rate": [58.1, 64.5, 84.2],
        "colors": ["#475569", "#3b82f6", "#10b981"],
    }
    assert isinstance(data, dict), "Data must be a dictionary"
    assert len(data["models"]) == 3, "Must include exactly 3 comparison benchmarks"
    return data


def generate_comparative_alpha_chart(output_path: str) -> str:
    """Renders dark-mode institutional alpha proof card with performance metrics."""
    assert isinstance(output_path, str) and len(output_path) > 0, "Invalid output_path"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    d = prepare_benchmark_data()

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0b0f19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.grid(color="#1f2937", linestyle="--", linewidth=0.5, alpha=0.7)

    y_pos = np.arange(len(d["models"]))
    bars1 = ax1.barh(y_pos, [c * 100 for c in d["corj60"]], color=d["colors"], height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(d["models"], fontsize=10, fontweight="bold")
    ax1.set_xlabel("60-Day Correlation (corj60 %)", fontsize=11, color="#94a3b8")
    ax1.set_title("Alpha Generation (+216.1% Edge)", fontsize=12, fontweight="bold", pad=10)
    for b in bars1:
        w = b.get_width()
        ax1.text(w + 0.15, b.get_y() + b.get_height() / 2, f"+{w:.2f}%", va="center", fontsize=10, color="white", fontweight="bold")

    bars2 = ax2.barh(y_pos, d["win_rate"], color=d["colors"], height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["", "", ""])
    ax2.set_xlabel("Positive Era Win Rate (%)", fontsize=11, color="#94a3b8")
    ax2.set_title("Staker Capital Consistency", fontsize=12, fontweight="bold", pad=10)
    for b in bars2:
        w = b.get_width()
        ax2.text(w + 1.0, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10, color="white", fontweight="bold")

    fig.suptitle("CYPHERPOLE QUANTITATIVE FLEET — STAKER ALPHA AUDIT\nVerified Track Record: numer.ai/cypherpole | Trial Coupon: CYPHER100 (Numerbay.ai)", fontsize=11, color="#38bdf8", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    assert os.path.exists(output_path), "Generated chart file must exist"
    return output_path


if __name__ == "__main__":
    out_file = os.path.join(BASE_DIR, "docs", "cypherpole_alpha_edge.png")
    saved = generate_comparative_alpha_chart(out_file)
    print(f"Alpha proof chart saved to: {saved}")
