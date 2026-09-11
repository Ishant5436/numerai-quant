"""
Unit and Property Tests for 60-Day Multi-Target Ensemble and FNCv3 Neutralization.
Adheres strictly to Power of 10 invariants and Numerai v5.0 standards.
"""

import json
import os
import numpy as np
import pandas as pd
from neutralize import neutralize, rank_01

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_60d_config_targets():
    """Verify that config exports the 60-day target quintet and FNCv3 definitions."""
    from config import ENSEMBLE_TARGETS_60D, FNCV3_SET, NEUTRALIZATION_PROPORTION, MODEL_60D_DIR

    assert len(ENSEMBLE_TARGETS_60D) == 5
    expected_targets = [
        "target_cyrusd_60",
        "target_agnes_60",
        "target_victor_60",
        "target_jeremy_60",
        "target_xerxes_60"
    ]
    for target in expected_targets:
        assert target in ENSEMBLE_TARGETS_60D

    assert FNCV3_SET == "fncv3_features"
    assert 0.20 <= NEUTRALIZATION_PROPORTION <= 0.50
    assert os.path.basename(MODEL_60D_DIR) == "ensemble_60d"


def test_fncv3_features_count_in_fleet_groups():
    """Verify load_feature_groups extracts the canonical 400 FNCv3 features from features.json."""
    from fleet_submit import load_feature_groups

    features_path = os.path.join(BASE_DIR, "features.json")
    assert os.path.exists(features_path)
    with open(features_path) as f:
        meta = json.load(f)

    expected_fncv3 = meta["feature_sets"]["fncv3_features"]
    assert len(expected_fncv3) == 400

    groups = load_feature_groups()
    assert "fncv3_features" in groups
    assert len(groups["fncv3_features"]) == 400
    assert groups["fncv3_features"] == expected_fncv3


def test_qr_neutralization_fncv3_invariants():
    """Test mathematical invariants of QR feature neutralization against 400 factors."""
    n_assets = 1000
    n_factors = 400
    rng = np.random.default_rng(seed=42)

    x_mat = rng.standard_normal((n_assets, n_factors))
    feat_cols = [f"factor_{i}" for i in range(n_factors)]
    df = pd.DataFrame(x_mat, columns=feat_cols)
    raw_scores = rng.standard_normal(n_assets)
    df["pred"] = rank_01(raw_scores)

    # 1. Output length and NaN invariance
    neutralized_df = neutralize(df, ["pred"], extra_neutralizers=feat_cols, proportion=0.35)
    neut_pred = neutralized_df["pred"].values

    assert len(neut_pred) == n_assets
    assert not np.isnan(neut_pred).any()
    assert np.all(neut_pred >= 0.0)
    assert np.all(neut_pred <= 1.0)

    # 2. Assert rank distribution uniformity
    assert np.isclose(np.mean(neut_pred), 0.5, atol=0.05)


def test_generate_predictions_with_60d_ensemble_fallback():
    """Verify generate_tri_ensemble_prediction handles 60d models or mock fallback safely."""
    from fleet_submit import generate_tri_ensemble_prediction, load_feature_groups

    groups = load_feature_groups()
    feats = groups["all_medium"][:100]
    n_rows = 50
    rng = np.random.default_rng(seed=123)

    dummy_df = pd.DataFrame(rng.uniform(0.0, 1.0, (n_rows, len(feats))), columns=feats)
    neutralizer_feats = groups.get("fncv3_features", groups["all_medium"])[:50]

    preds = generate_tri_ensemble_prediction(
        dummy_df,
        strat_id=1,
        feature_subset=feats,
        neut_proportion=0.35,
        neutralizer_feats=neutralizer_feats,
        allow_mock_fallback=True
    )

    assert len(preds) == n_rows
    assert not np.isnan(preds).any()
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_60d_serialized_model_artifacts_exist():
    """Verify that all 5 trained 60-day LightGBM models exist and have valid structure."""
    import joblib
    import pytest
    from config import ENSEMBLE_TARGETS_60D, MODEL_60D_DIR

    weights_exist = all(os.path.exists(os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl")) for t in ENSEMBLE_TARGETS_60D)
    if not weights_exist:
        pytest.skip("60-day model weights are gitignored; test runs in local environment where weights are generated.")

    for target in ENSEMBLE_TARGETS_60D:
        model_path = os.path.join(MODEL_60D_DIR, f"lgb_{target}.pkl")
        assert os.path.exists(model_path), f"Missing model: {model_path}"
        assert os.path.getsize(model_path) > 10_000, f"Model file too small: {model_path}"
        model = joblib.load(model_path)
        assert hasattr(model, "predict"), f"Loaded object missing predict method: {target}"

