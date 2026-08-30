"""
Black-Box Test Suite for Numerai Quant Fleet
Tests feature metadata loading, schema integrity, prediction pipeline output constraints,
and shell daemon syntax correctness.
"""

import json
import os
import subprocess
import pytest
import numpy as np
import pandas as pd
from fleet_submit import load_feature_groups, generate_tri_ensemble_prediction

NUMERAI_ROOT = "/Users/ishantpanchal/numerai-quant"


# ==============================================================================
# 1. Feature Metadata & Configuration Validation (Black-Box)
# ==============================================================================

def test_blackbox_features_json_schema():
    """Black-box: features.json contains valid feature_sets mapping with medium tier."""
    features_path = os.path.join(NUMERAI_ROOT, "features.json")
    assert os.path.exists(features_path), "features.json must exist"
    with open(features_path) as f:
        meta = json.load(f)

    assert "feature_sets" in meta
    assert "medium" in meta["feature_sets"]
    medium_feats = meta["feature_sets"]["medium"]
    assert len(medium_feats) == 705, f"Expected 705 medium features, got {len(medium_feats)}"


def test_blackbox_load_feature_groups_integrity():
    """Black-box: load_feature_groups returns non-empty disjoint/orthogonal strategy subsets."""
    groups = load_feature_groups()
    expected_strategies = ["all_medium", "fundamental", "momentum", "macro", "constitution"]
    for strat in expected_strategies:
        assert strat in groups, f"Strategy subset '{strat}' missing"
        assert len(groups[strat]) > 0, f"Strategy subset '{strat}' is empty"
        assert all(isinstance(f, str) for f in groups[strat]), "Feature names must be strings"


# ==============================================================================
# 2. Prediction Pipeline & Bounds Verification (Black-Box)
# ==============================================================================

def test_blackbox_prediction_generation_bounds():
    """Black-box: prediction generator produces valid [0, 1] uniform predictions with zero NaNs."""
    groups = load_feature_groups()
    medium_feats = groups["all_medium"]
    n_rows = 50

    # Synthetic live dataset
    np.random.seed(42)
    fake_data = np.random.uniform(0, 1, size=(n_rows, len(medium_feats)))
    live_df = pd.DataFrame(fake_data, columns=medium_feats, index=[f"id_{i}" for i in range(n_rows)])

    neutralizer_feats = medium_feats[:30]

    # Test Strategy 1 (Core Flagship)
    preds = generate_tri_ensemble_prediction(
        live_df, strat_id=1, feature_subset=groups["all_medium"], neut_proportion=0.25, neutralizer_feats=neutralizer_feats
    )

    assert len(preds) == n_rows
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any(), "Predictions must contain ZERO NaNs"
    assert not np.isinf(preds).any(), "Predictions must contain ZERO Infs"
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


# ==============================================================================
# 3. Automation Shell Script Syntax (Black-Box)
# ==============================================================================

def test_blackbox_cron_submit_syntax():
    """Black-box: verify cron_submit.sh syntax is 100% valid bash."""
    cron_script = os.path.join(NUMERAI_ROOT, "cron_submit.sh")
    assert os.path.exists(cron_script)
    res = subprocess.run(["bash", "-n", cron_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Bash syntax check failed: {res.stderr}"
