"""
Unit and Invariant Tests for Purged Walk-Forward Cross-Validation.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import pandas as pd
from walk_forward_cv import (
    FOLD_DEFINITIONS,
    PURGE_BUFFER_ERAS,
    compute_regime_metrics,
    get_purged_fold_eras,
)


def test_fold_definitions_completeness():
    """Assert all 4 chronological folds are defined without temporal gaps."""
    assert len(FOLD_DEFINITIONS) == 4, f"Expected 4 folds, got {len(FOLD_DEFINITIONS)}"
    for fold_id in range(1, 5):
        assert fold_id in FOLD_DEFINITIONS, f"Fold {fold_id} missing"
        info = FOLD_DEFINITIONS[fold_id]
        assert info["start_era"] <= info["end_era"], f"Invalid fold range in {info['name']}"
        assert len(info["name"]) > 0, "Empty fold name"

    # Verify contiguous boundaries
    assert FOLD_DEFINITIONS[1]["end_era"] == "0739"
    assert FOLD_DEFINITIONS[2]["start_era"] == "0740"
    assert FOLD_DEFINITIONS[2]["end_era"] == "0904"
    assert FOLD_DEFINITIONS[3]["start_era"] == "0905"
    assert FOLD_DEFINITIONS[3]["end_era"] == "1069"
    assert FOLD_DEFINITIONS[4]["start_era"] == "1070"
    assert FOLD_DEFINITIONS[4]["end_era"] == "1234"


def test_get_purged_fold_eras_invariants():
    """Assert era slicing trims exactly the purge buffer and prevents leakage."""
    mock_eras = [f"{i:04d}" for i in range(500, 750)]
    purged = get_purged_fold_eras(mock_eras, "0575", "0739", purge_buffer=PURGE_BUFFER_ERAS)

    # Total eras in 0575..0739 inclusive is 165
    assert len(purged) == 165 - PURGE_BUFFER_ERAS, f"Expected {165 - PURGE_BUFFER_ERAS}, got {len(purged)}"
    # First era in purged must be 0575 + 8 = 0583
    assert purged[0] == "0583", f"Expected first purged era to be 0583, got {purged[0]}"
    assert purged[-1] == "0739", f"Expected last purged era to be 0739, got {purged[-1]}"


def test_compute_regime_metrics_calculation():
    """Assert statistical calculations for Sharpe, drawdown, and hit rate."""
    corrs = pd.Series([0.03, 0.04, -0.01, 0.02, 0.05, 0.01])
    metrics = compute_regime_metrics(corrs)

    assert metrics["mean_corr"] > 0.0, "Mean correlation must be positive"
    assert metrics["raw_era_sharpe"] > 0.0, "Sharpe must be positive"
    assert 0.0 <= metrics["max_drawdown"] <= 1.0, "Drawdown out of bounds"
    assert metrics["hit_rate"] == round(5 / 6, 3), f"Expected hit rate 0.833, got {metrics['hit_rate']}"
    assert metrics["n_eras"] == 6, f"Expected 6 eras, got {metrics['n_eras']}"
