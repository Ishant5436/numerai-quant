"""
Unit and Property Tests for 25-Model Fleet 60-Day Architecture Upgrade.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""

import os
import numpy as np
import pandas as pd

from config import FLEET_STRATEGY_MAP_60D, ORTHO_60D_DIR
from fleet_submit import load_feature_groups, generate_tri_ensemble_prediction


def test_fleet_strategy_map_60d_completeness():
    """Assert all 25 strategies map to valid 60-day targets and existing feature groups."""
    assert len(FLEET_STRATEGY_MAP_60D) == 25, f"Expected 25 strategies, got {len(FLEET_STRATEGY_MAP_60D)}"
    groups = load_feature_groups()

    for strat_id in range(1, 26):
        assert strat_id in FLEET_STRATEGY_MAP_60D, f"Strategy {strat_id} missing from 60d map"
        target, feat_group, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
        
        # Target assertion
        if strat_id == 1:
            assert target == "5_target_quintet"
        else:
            assert target.endswith("_60"), f"Strategy {strat_id} target '{target}' does not end with '_60'"
        
        # Feature group assertion
        assert feat_group in groups, f"Strategy {strat_id} references unknown feature group: {feat_group}"
        assert len(groups[feat_group]) > 0, f"Feature group '{feat_group}' is empty"
        
        # Neutralization proportion assertion
        assert 0.20 <= neut_prop <= 0.50, f"Strategy {strat_id} invalid neut proportion: {neut_prop}"


def test_fleet_60d_model_files_exist():
    """TDD Gate: Verify that all 24 60-day models exist in ORTHO_60D_DIR."""
    assert os.path.exists(ORTHO_60D_DIR), f"Directory missing: {ORTHO_60D_DIR}"
    
    missing = []
    for strat_id in range(2, 26):
        model_file = os.path.join(ORTHO_60D_DIR, f"lgb_strat_{strat_id}.pkl")
        if not os.path.exists(model_file) or os.path.getsize(model_file) < 10_000:
            missing.append(f"strat_{strat_id}")
    
    assert len(missing) == 0, f"Missing 60d model weights: {missing}"


def test_fleet_60d_inference_and_neutralization_invariants():
    """Assert generate_tri_ensemble_prediction produces uniform rank vectors on [0.0, 1.0]."""
    groups = load_feature_groups()
    feats = groups["all_medium"][:50]
    n_assets = 100
    rng = np.random.default_rng(seed=42)
    dummy_df = pd.DataFrame(rng.uniform(0.0, 1.0, (n_assets, len(feats))), columns=feats)
    fncv3 = groups["fncv3_features"][:50]

    # Test strategy 1 (quintet) and strategy 2 with mock fallback
    for s_id in [1, 2]:
        preds = generate_tri_ensemble_prediction(
            dummy_df,
            strat_id=s_id,
            feature_subset=feats,
            neut_proportion=0.35,
            neutralizer_feats=fncv3,
            allow_mock_fallback=True
        )
        assert len(preds) == n_assets
        assert not np.isnan(preds).any()
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)
        assert np.isclose(np.mean(preds), 0.5, atol=0.05)


def test_all_25_strategies_real_inference():
    """Verify genuine model inference (zero mock fallback) across all 25 fleet strategies."""
    groups = load_feature_groups()
    fncv3 = groups["fncv3_features"]
    n_assets = 20
    rng = np.random.default_rng(seed=99)
    dummy_df = pd.DataFrame(rng.uniform(0.0, 1.0, (n_assets, len(groups["all_medium"]))), columns=groups["all_medium"])

    for strat_id in range(1, 26):
        target, feat_key, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
        feature_subset = groups[feat_key]
        
        preds = generate_tri_ensemble_prediction(
            dummy_df,
            strat_id=strat_id,
            feature_subset=feature_subset,
            neut_proportion=neut_prop,
            neutralizer_feats=fncv3,
            allow_mock_fallback=False
        )
        assert len(preds) == n_assets, f"Strategy {strat_id} returned wrong length"
        assert not np.isnan(preds).any(), f"Strategy {strat_id} produced NaNs"
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0), f"Strategy {strat_id} out of bounds"

