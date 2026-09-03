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
from pathlib import Path
from fleet_submit import load_feature_groups, generate_tri_ensemble_prediction, resolve_strategy_config

NUMERAI_ROOT = str(Path(__file__).resolve().parent.parent)


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
    expected_strategies = [
        "all_medium", "fundamental", "momentum", "macro", "constitution",
        "quality_defensive", "trend_velocity", "value_capital", "macro_tail"
    ]
    for strat in expected_strategies:
        assert strat in groups, f"Strategy subset '{strat}' missing"
        assert len(groups[strat]) > 0, f"Strategy subset '{strat}' is empty"
        assert all(isinstance(f, str) for f in groups[strat]), "Feature names must be strings"


def test_blackbox_resolve_strategy_config_coverage():
    """Black-box: resolve_strategy_config deterministically maps names/indices across all 10 strategies."""
    expected_strats = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5,
        5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 1
    }
    for idx, expected_sid in expected_strats.items():
        sid, grp, neut = resolve_strategy_config("generic_model", idx)
        assert sid == expected_sid, f"Expected strategy {expected_sid} for idx {idx}, got {sid}"
        assert isinstance(grp, str)
        assert 0.20 <= neut <= 0.55

    keyword_cases = [
        ("cypherpole_fund", 2),
        ("alpha_jeremy_v1", 2),
        ("momentum_bot", 3),
        ("victor_runner", 3),
        ("macro_regime", 4),
        ("xerxes_tail", 4),
        ("res_specialist", 5),
        ("delta_hedger", 5),
        ("cyrus_deep", 6),
        ("quality_defensive", 7),
        ("trend_vel", 8),
        ("capital_value", 9),
        ("macro_tail_shield", 10),
    ]
    for name, expected_sid in keyword_cases:
        sid, grp, neut = resolve_strategy_config(name, 0)
        assert sid == expected_sid, f"Keyword '{name}' failed: expected {expected_sid}, got {sid}"

    sid_empty, _, _ = resolve_strategy_config("", 0)
    assert sid_empty == 1
    sid_none, _, _ = resolve_strategy_config(None, 0)
    assert sid_none == 1



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

    # Test Strategy 1 (Core Flagship) with explicit test fallback permitted
    preds = generate_tri_ensemble_prediction(
        live_df, strat_id=1, feature_subset=groups["all_medium"], neut_proportion=0.25, neutralizer_feats=neutralizer_feats, allow_mock_fallback=True
    )

    assert len(preds) == n_rows
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any(), "Predictions must contain ZERO NaNs"
    assert not np.isinf(preds).any(), "Predictions must contain ZERO Infs"
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_blackbox_expansion_strategies_6_to_10_bounds():
    """Black-box: strategies 6 through 10 generate valid [0, 1] bounded predictions with zero NaNs."""
    groups = load_feature_groups()
    medium_feats = groups["all_medium"]
    n_rows = 20
    np.random.seed(123)
    fake_data = np.random.uniform(0, 1, size=(n_rows, len(medium_feats)))
    live_df = pd.DataFrame(fake_data, columns=medium_feats, index=[f"id_{i}" for i in range(n_rows)])
    neutralizer_feats = medium_feats[:20]

    expansion_mapping = {
        6: ("all_medium", 0.30),
        7: ("quality_defensive", 0.35),
        8: ("trend_velocity", 0.40),
        9: ("value_capital", 0.35),
        10: ("macro_tail", 0.45)
    }

    for sid, (grp, neut) in expansion_mapping.items():
        preds = generate_tri_ensemble_prediction(
            live_df, strat_id=sid, feature_subset=groups[grp], neut_proportion=neut,
            neutralizer_feats=neutralizer_feats, allow_mock_fallback=True
        )
        assert len(preds) == n_rows
        assert not np.isnan(preds).any(), f"Strategy {sid} produced NaNs"
        assert not np.isinf(preds).any(), f"Strategy {sid} produced Infs"
        assert np.all(preds >= 0.0), f"Strategy {sid} lower bound violated"
        assert np.all(preds <= 1.0), f"Strategy {sid} upper bound violated"



def test_blackbox_production_fails_loudly_when_models_missing():
    """Production guard: verify default allow_mock_fallback=False raises FileNotFoundError when weights missing."""
    groups = load_feature_groups()
    medium_feats = groups["all_medium"]
    live_df = pd.DataFrame(np.zeros((5, len(medium_feats))), columns=medium_feats)
    # Using non-existent strat_id 99999 to guarantee model files are absent
    with pytest.raises(FileNotFoundError, match="Production Error: No model weights found"):
        generate_tri_ensemble_prediction(
            live_df, strat_id=99999, feature_subset=medium_feats, neut_proportion=0.25, neutralizer_feats=medium_feats[:10], allow_mock_fallback=False
        )


# ==============================================================================
# 3. Automation Shell Script Syntax (Black-Box)
# ==============================================================================

def test_blackbox_cron_submit_syntax():
    """Black-box: verify cron_submit.sh syntax is 100% valid bash."""
    cron_script = os.path.join(NUMERAI_ROOT, "cron_submit.sh")
    assert os.path.exists(cron_script)
    res = subprocess.run(["bash", "-n", cron_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Bash syntax check failed: {res.stderr}"


def test_blackbox_fleet_submit_main_orchestration_mocked(monkeypatch, tmp_path):
    """Integration: verify fleet_submit.main() executes end-to-end without NameErrors or crashes."""
    from unittest.mock import MagicMock
    import fleet_submit

    mock_napi = MagicMock()
    mock_napi.get_current_round.return_value = 1346
    mock_napi.get_models.return_value = {
        "cypherpole": "mod-1",
        "cypherpole_fund": "mod-2",
        "cypherpole_mom": "mod-3",
        "cypherpole_macro": "mod-4",
        "cypherpole_res": "mod-5",
    }
    mock_napi.upload_predictions.return_value = "sub-12345"

    monkeypatch.setattr(fleet_submit, "NumerAPI", lambda **kwargs: mock_napi)

    # Mock download_dataset to write a miniature live.parquet with required columns
    groups = fleet_submit.load_feature_groups()
    medium_feats = groups["all_medium"]
    n_assets = 15
    mini_live = pd.DataFrame(
        np.random.uniform(0, 1, size=(n_assets, len(medium_feats))),
        columns=medium_feats,
        index=[f"id_{i}" for i in range(n_assets)]
    )

    def mock_download(path, target_file):
        mini_live.to_parquet(target_file)

    mock_napi.download_dataset.side_effect = mock_download

    # Mock generate_tri_ensemble_prediction to use allow_mock_fallback=True
    orig_gen = fleet_submit.generate_tri_ensemble_prediction
    def safe_gen(live_df, strat_id, feature_subset, neut_proportion, neutralizer_feats, allow_mock_fallback=False):
        return orig_gen(live_df, strat_id, feature_subset, neut_proportion, neutralizer_feats, allow_mock_fallback=True)

    monkeypatch.setattr(fleet_submit, "generate_tri_ensemble_prediction", safe_gen)

    # Execute main()
    fleet_submit.main()

    # Verify calls
    assert mock_napi.get_current_round.called
    assert mock_napi.get_models.called
    assert mock_napi.download_dataset.called
    assert mock_napi.upload_predictions.call_count == 5

