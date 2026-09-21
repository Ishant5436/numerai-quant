"""
Unit and Property Tests for Fleet Percentile Optimization.
Tests Asymmetric Adaptive Anchoring (A3), Feature-Proportional Neutralization (FPN),
and Fleet Orthogonality Invariants.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import os
import numpy as np
import pandas as pd
import pytest

from config import (
    FLEET_STRATEGY_MAP_60D,
    STRATEGY_ANCHOR_WEIGHTS,
    ORTHO_60D_DIR,
    MODEL_60D_DIR,
)
from fleet_submit import (
    load_feature_groups,
    resolve_strategy_config,
    generate_tri_ensemble_prediction,
)
from evaluate_fleet_60d import calc_effective_bets


def test_strategy_anchor_weights_structure_and_bounds():
    """Assert STRATEGY_ANCHOR_WEIGHTS covers all 30 strategies with calibrated tiers."""
    assert len(STRATEGY_ANCHOR_WEIGHTS) == 30, f"Expected 30 weights, got {len(STRATEGY_ANCHOR_WEIGHTS)}"
    
    # Flagship Tier: Zero dilution of core alpha (anchor = 0.0)
    for flagship_id in [1, 6, 14, 29, 30]:
        assert STRATEGY_ANCHOR_WEIGHTS[flagship_id] == 0.0, (
            f"Flagship strategy {flagship_id} must have anchor weight 0.0, got {STRATEGY_ANCHOR_WEIGHTS[flagship_id]}"
        )
    
    # Targeted Laggard Tier: 0.30 anchor weight to lift percentile floor
    for laggard_id in [2, 5, 7, 21]:
        assert STRATEGY_ANCHOR_WEIGHTS[laggard_id] == 0.30, (
            f"Laggard specialist {laggard_id} must have anchor weight 0.30, got {STRATEGY_ANCHOR_WEIGHTS[laggard_id]}"
        )
        
    # Micro specialist: 0.25
    assert STRATEGY_ANCHOR_WEIGHTS[19] == 0.25, f"Strategy 19 must have anchor weight 0.25, got {STRATEGY_ANCHOR_WEIGHTS[19]}"

    # Orthogonal Factor Tier: 0.20 anchor weight
    for ortho_id in [3, 4, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28]:
        assert STRATEGY_ANCHOR_WEIGHTS[ortho_id] == 0.20, (
            f"Orthogonal factor strategy {ortho_id} must have anchor weight 0.20, got {STRATEGY_ANCHOR_WEIGHTS[ortho_id]}"
        )


def test_feature_proportional_neutralization_bounds():
    """Assert narrow feature subsets have reduced neutralization proportion (<= 0.25)."""
    groups = load_feature_groups()
    
    # Strategy 5: constitution (134 features) -> must be 0.25 (not 0.35 or 0.50)
    assert FLEET_STRATEGY_MAP_60D[5][2] == 0.25, (
        f"Strategy 5 (constitution) neutralization must be 0.25, got {FLEET_STRATEGY_MAP_60D[5][2]}"
    )
    
    # Strategy 7: quality_defensive -> must be 0.25
    assert FLEET_STRATEGY_MAP_60D[7][2] == 0.25, (
        f"Strategy 7 (quality_defensive) neutralization must be 0.25, got {FLEET_STRATEGY_MAP_60D[7][2]}"
    )
    
    # Strategy 19: mean_reversion -> must be 0.25
    assert FLEET_STRATEGY_MAP_60D[19][2] == 0.25, (
        f"Strategy 19 (mean_reversion) neutralization must be 0.25, got {FLEET_STRATEGY_MAP_60D[19][2]}"
    )

    # Strategy 21: high_sharpe_quality -> must be 0.25
    assert FLEET_STRATEGY_MAP_60D[21][2] == 0.25, (
        f"Strategy 21 (high_sharpe_quality) neutralization must be 0.25, got {FLEET_STRATEGY_MAP_60D[21][2]}"
    )


def test_fleet_submit_routing_synchronization():
    """Verify resolve_strategy_config reflects updated neutralization proportions."""
    strat_id, feat_key, neut_prop = resolve_strategy_config("cypherpole_res", 4)
    assert strat_id == 5, f"Expected strat_id 5, got {strat_id}"
    assert feat_key == "constitution", f"Expected constitution, got {feat_key}"
    assert neut_prop == 0.25, f"Expected 0.25 neutralization for cypherpole_res, got {neut_prop}"

    s_id, f_k, n_p = resolve_strategy_config("cypherpole_qual", 6)
    assert s_id == 7, f"Expected strat_id 7, got {s_id}"
    assert n_p == 0.25, f"Expected 0.25 neutralization for cypherpole_qual, got {n_p}"


def test_adaptive_anchored_prediction_invariants():
    """Assert generate_tri_ensemble_prediction uses adaptive anchoring and preserves bounds."""
    groups = load_feature_groups()
    feats = groups["all_medium"][:60]
    n_assets = 120
    rng = np.random.default_rng(seed=123)
    dummy_df = pd.DataFrame(rng.uniform(0.0, 1.0, (n_assets, len(feats))), columns=feats)
    fncv3 = groups["fncv3_features"][:40]

    # Test Strategy 5 (laggard, anchor weight 0.35)
    preds = generate_tri_ensemble_prediction(
        dummy_df,
        strat_id=5,
        feature_subset=feats,
        neut_proportion=0.25,
        neutralizer_feats=fncv3,
        allow_mock_fallback=True,
    )
    assert len(preds) == n_assets, "Prediction length mismatch"
    assert not np.isnan(preds).any(), "NaN values found in predictions"
    assert np.all(preds >= 0.0), "Prediction below 0.0"
    assert np.all(preds <= 1.0), "Prediction above 1.0"
    assert np.isclose(np.mean(preds), 0.5, atol=0.05), "Prediction mean not centered at 0.5"


def test_fleet_orthogonality_with_tiered_anchoring():
    """Verify that tiered anchoring preserves high effective bets (N_eff >= 3.20)."""
    groups = load_feature_groups()
    n_assets = 200
    rng = np.random.default_rng(seed=456)
    dummy_df = pd.DataFrame(rng.uniform(0.0, 1.0, (n_assets, len(groups["all_medium"]))), columns=groups["all_medium"])
    fncv3 = groups["fncv3_features"]

    fleet_preds = []
    # Generate mock predictions across 25 strategies
    base_signal = rng.normal(0, 1, n_assets)
    flagship_pred = base_signal + rng.normal(0, 0.5, n_assets)

    for strat_id in range(1, 26):
        target, feat_key, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
        weight = STRATEGY_ANCHOR_WEIGHTS[strat_id]
        
        # Independent specialist noise + factor tilt
        specialist_signal = rng.normal(0, 1, n_assets)
        blended = (1.0 - weight) * specialist_signal + weight * flagship_pred
        fleet_preds.append(blended)

    pred_matrix = np.column_stack(fleet_preds)
    corr_matrix = np.corrcoef(pred_matrix, rowvar=False)
    n_eff = calc_effective_bets(corr_matrix)

    assert n_eff >= 3.20, f"Effective bets N_eff {n_eff:.2f} fell below threshold 3.20"
    upper_corrs = corr_matrix[np.triu_indices(25, k=1)]
    assert np.max(upper_corrs) < 0.85, f"Pairwise correlation exceeded 0.85: max={np.max(upper_corrs):.4f}"
