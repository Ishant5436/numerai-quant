"""
Unit and Integration Tests for Numerai 30-Model Tournament Fleet Scaling
Validates strategies 16 through 30, feature partitions, 30-slot routing, and rank guarantees.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fleet_submit import load_feature_groups, resolve_strategy_config, generate_tri_ensemble_prediction


@pytest.fixture
def feature_groups():
    return load_feature_groups()


@pytest.fixture
def synthetic_live_data(feature_groups):
    np.random.seed(42)
    n_samples = 250
    medium_feats = feature_groups["all_medium"]
    data = np.random.choice([0, 1, 2, 3, 4], size=(n_samples, len(medium_feats)))
    df = pd.DataFrame(data, columns=medium_feats, index=[f"id_{i}" for i in range(n_samples)])
    return df


def test_feature_groups_contain_all_30_strategies(feature_groups):
    required_keys = [
        "all_medium", "fundamental", "momentum", "macro", "constitution",
        "quality_defensive", "trend_velocity", "value_capital", "macro_tail",
        "alpha_conviction", "volatility_defensive", "risk_parity", "macro_hedged",
        # Strategies 16 - 30 expansion groups:
        "fundamental_value", "low_beta_defensive", "residual_alpha", "mean_reversion",
        "factor_momentum", "high_sharpe_quality", "macro_tail_liquidity", "earnings_quality",
        "sentiment_divergence", "vol_adjusted_alpha", "orthogonal_risk_parity", "residual_spread",
        "growth_trend"
    ]
    medium_set = set(feature_groups["all_medium"])
    for key in required_keys:
        assert key in feature_groups, f"Missing feature group '{key}' in load_feature_groups()"
        subset = feature_groups[key]
        assert len(subset) > 0, f"Feature group '{key}' is unexpectedly empty"
        assert set(subset).issubset(medium_set), f"Feature group '{key}' contains non-medium features"


def test_resolve_strategy_config_30_slots(feature_groups):
    expected_models = [
        ("cypherpole", 1, "all_medium"),
        ("cypherpole_fund", 2, "fundamental"),
        ("cypherpole_mom", 3, "momentum"),
        ("cypherpole_macro", 4, "macro"),
        ("cypherpole_res", 5, "constitution"),
        ("cypherpole_cyrus", 6, "all_medium"),
        ("cypherpole_qual", 7, "quality_defensive"),
        ("cypherpole_vel", 8, "trend_velocity"),
        ("cypherpole_val", 9, "value_capital"),
        ("cypherpole_tail", 10, "macro_tail"),
        ("cypherpole_alpha", 11, "alpha_conviction"),
        ("cypherpole_vol", 12, "volatility_defensive"),
        ("cypherpole_sharpe", 13, "risk_parity"),
        ("cypherpole_deep", 14, "all_medium"),
        ("cypherpole_hedged", 15, "macro_hedged"),
        ("cypherpole_bravo", 16, "fundamental_value"),
        ("cypherpole_charlie", 17, "low_beta_defensive"),
        ("cypherpole_delta", 18, "residual_alpha"),
        ("cypherpole_echo", 19, "mean_reversion"),
        ("cypherpole_ralph", 20, "factor_momentum"),
        ("cypherpole_rowan", 21, "high_sharpe_quality"),
        ("cypherpole_sam", 22, "macro_tail_liquidity"),
        ("cypherpole_tyler", 23, "earnings_quality"),
        ("cypherpole_waldo", 24, "sentiment_divergence"),
        ("cypherpole_victor", 25, "vol_adjusted_alpha"),
        ("cypherpole_claudia", 26, "orthogonal_risk_parity"),
        ("cypherpole_agnes", 27, "residual_spread"),
        ("cypherpole_caroline", 28, "growth_trend"),
        ("cypherpole_ender", 29, "all_medium"),
        ("cypherpole_supernova", 30, "all_medium"),
    ]

    for model_name, expected_id, expected_group in expected_models:
        strat_id, group_key, neut = resolve_strategy_config(model_name, expected_id - 1)
        assert strat_id == expected_id, f"Model '{model_name}' mapped to strat_id {strat_id}, expected {expected_id}"
        assert group_key == expected_group, f"Model '{model_name}' mapped to group '{group_key}', expected '{expected_group}'"
        assert group_key in feature_groups, f"Mapped group '{group_key}' not present in feature_groups"
        assert 0.20 <= neut <= 0.50, f"Neutralization proportion {neut} out of [0.20, 0.50] for '{model_name}'"

    # Test modulo slot routing across generic unbranded names for indices 0..29
    for idx in range(30):
        strat_id, group_key, neut = resolve_strategy_config(f"unbranded_model_{idx}", idx)
        assert 1 <= strat_id <= 30
        assert group_key in feature_groups
        assert 0.20 <= neut <= 0.50


def test_generate_predictions_strategies_16_to_30(feature_groups, synthetic_live_data):
    neutralizer_feats = feature_groups["all_medium"][:60]
    for strat_id in range(16, 31):
        _, group_key, neut_prop = resolve_strategy_config(f"slot_{strat_id}", strat_id - 1)
        feature_subset = feature_groups[group_key]

        preds = generate_tri_ensemble_prediction(
            synthetic_live_data,
            strat_id,
            feature_subset,
            neut_prop,
            neutralizer_feats,
            allow_mock_fallback=True
        )

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(synthetic_live_data),)
        assert not np.isnan(preds).any(), f"NaNs detected in predictions for strategy {strat_id}"
        assert not np.isinf(preds).any(), f"Infs detected in predictions for strategy {strat_id}"
        assert preds.min() >= 0.0, f"Prediction out of lower bound [0.0] for strategy {strat_id}"
        assert preds.max() <= 1.0, f"Prediction out of upper bound [1.0] for strategy {strat_id}"
        assert np.std(preds) > 0.05, f"Degenerate zero-variance predictions for strategy {strat_id}"


def test_cross_strategy_correlation_bounded(feature_groups, synthetic_live_data):
    neutralizer_feats = feature_groups["all_medium"][:60]
    preds_dict = {}
    for strat_id in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        _, group_key, neut_prop = resolve_strategy_config(f"strat_{strat_id}", strat_id - 1)
        preds_dict[strat_id] = generate_tri_ensemble_prediction(
            synthetic_live_data,
            strat_id,
            feature_groups[group_key],
            neut_prop,
            neutralizer_feats,
            allow_mock_fallback=True
        )

    strat_ids = list(preds_dict.keys())
    for i in range(len(strat_ids)):
        for j in range(i + 1, len(strat_ids)):
            id_a, id_b = strat_ids[i], strat_ids[j]
            r, _ = spearmanr(preds_dict[id_a], preds_dict[id_b])
            assert abs(r) < 0.90, f"Excessive correlation {r:.3f} between Strategy {id_a} and {id_b}"


def test_real_weights_load_and_predict_strategies_16_to_25(feature_groups, synthetic_live_data):
    """Verifies that all 10 trained model files (16-25) load from disk with zero mock fallback when present."""
    from fleet_submit import ORTHO_DIR
    weights_exist = all(os.path.exists(os.path.join(ORTHO_DIR, f"lgb_strat_{sid}.pkl")) for sid in range(16, 26))
    if not weights_exist:
        pytest.skip("Model weights are gitignored; test runs in local environment where weights are generated.")

    neutralizer_feats = feature_groups["all_medium"][:60]
    for strat_id in range(16, 26):
        _, group_key, neut_prop = resolve_strategy_config(f"strat_{strat_id}", strat_id - 1)
        feature_subset = feature_groups[group_key]

        preds = generate_tri_ensemble_prediction(
            synthetic_live_data,
            strat_id,
            feature_subset,
            neut_prop,
            neutralizer_feats,
            allow_mock_fallback=False
        )

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(synthetic_live_data),)
        assert not np.isnan(preds).any(), f"NaNs detected in live model predictions for strategy {strat_id}"
        assert not np.isinf(preds).any(), f"Infs detected in live model predictions for strategy {strat_id}"
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0
        assert np.std(preds) > 0.05

