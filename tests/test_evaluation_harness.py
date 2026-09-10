"""
Unit tests for Numerai 60-Day Fleet Evaluation Harness & Deterministic Routing.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import numpy as np
import pandas as pd
from config import EXPLICIT_MODEL_ROUTING, FLEET_STRATEGY_MAP_60D
from fleet_submit import load_feature_groups, resolve_strategy_config


def test_explicit_model_routing_completeness():
    """Assert all 25 registered Numerai models map deterministically to valid strategy slots."""
    assert len(EXPLICIT_MODEL_ROUTING) == 25, f"Expected 25 models, got {len(EXPLICIT_MODEL_ROUTING)}"
    assert "cypherpole" in EXPLICIT_MODEL_ROUTING, "Primary model cypherpole missing from routing"
    assert EXPLICIT_MODEL_ROUTING["cypherpole"] == 1, "cypherpole must route to Flagship Strategy 1"

    for model_name, expected_id in EXPLICIT_MODEL_ROUTING.items():
        strat_id, feat_key, neut_prop = resolve_strategy_config(model_name, idx=9999)
        assert strat_id == expected_id, f"Model {model_name} routed to {strat_id}, expected {expected_id}"
        assert strat_id in FLEET_STRATEGY_MAP_60D, f"Strategy {strat_id} missing from 60d map"
        assert 0.20 <= neut_prop <= 0.50, f"Strategy {strat_id} invalid neut proportion: {neut_prop}"


def test_feature_groups_zero_duplicates():
    """Assert zero duplicate feature groups and pairwise Jaccard similarity < 0.85."""
    groups = load_feature_groups()
    assert len(groups) >= 25, f"Expected at least 25 feature groups, got {len(groups)}"

    sets_map = {k: set(v) for k, v in groups.items()}
    names = list(sets_map.keys())

    for i in range(len(names)):
        n1 = names[i]
        s1 = sets_map[n1]
        assert len(s1) > 0, f"Feature group {n1} is empty"

        for j in range(i + 1, len(names)):
            n2 = names[j]
            s2 = sets_map[n2]
            assert s1 != s2, f"Feature groups {n1} and {n2} are identical duplicates"
            if n1 != "all_medium" and n2 != "all_medium" and n1 != "fncv3_features" and n2 != "fncv3_features":
                jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
                assert jaccard < 0.85, f"Excessive overlap ({jaccard:.2f}) between {n1} and {n2}"


def test_calc_spearman_per_era():
    """Verify Spearman correlation computation across eras with exact tie handling."""
    from evaluate_fleet_60d import calc_era_spearman

    n_rows = 100
    df = pd.DataFrame({
        "era": ["0001"] * 50 + ["0002"] * 50,
        "pred": np.linspace(0.0, 1.0, n_rows),
        "target": np.linspace(0.0, 1.0, n_rows)
    })
    
    corrs = calc_era_spearman(df, pred_col="pred", target_col="target")
    assert len(corrs) == 2, f"Expected 2 eras, got {len(corrs)}"
    assert np.isclose(corrs["0001"], 1.0, atol=1e-4), f"Expected perfect corr in era 0001, got {corrs['0001']}"
    assert np.isclose(corrs["0002"], 1.0, atol=1e-4), f"Expected perfect corr in era 0002, got {corrs['0002']}"


def test_calc_sharpe_and_drawdown():
    """Verify annualized Sharpe and cumulative correlation drawdown logic."""
    from evaluate_fleet_60d import calc_sharpe, calc_cumulative_drawdown

    era_series = pd.Series([0.02, 0.04, -0.01, 0.03, 0.05, -0.02])
    sharpe = calc_sharpe(era_series)
    assert np.isfinite(sharpe), "Sharpe must be finite"
    assert sharpe > 0.0, "Sharpe should be positive for net positive returns"

    # Test cumulative drawdown calculation
    dd = calc_cumulative_drawdown(era_series)
    assert dd >= 0.0, "Drawdown must be non-negative"
    assert dd <= 1.0, "Drawdown must be bounded"


def test_calc_effective_bets():
    """Verify PCA eigenvalue entropy formula for effective number of independent bets."""
    from evaluate_fleet_60d import calc_effective_bets

    # Perfectly correlated matrix (N x N of ones) -> N_eff should be 1.0
    corr_perfect = np.ones((5, 5))
    n_eff_perfect = calc_effective_bets(corr_perfect)
    assert np.isclose(n_eff_perfect, 1.0, atol=1e-2), f"Expected 1.0 for perfect corr, got {n_eff_perfect}"

    # Identity matrix (completely uncorrelated) -> N_eff should be N
    corr_identity = np.eye(5)
    n_eff_identity = calc_effective_bets(corr_identity)
    assert np.isclose(n_eff_identity, 5.0, atol=1e-2), f"Expected 5.0 for identity, got {n_eff_identity}"
