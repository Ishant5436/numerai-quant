"""
Unit tests for NumerbayPublisher module.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from numerbay_publisher import NumerbayPublisher, validate_prediction_file


def test_validate_prediction_file_valid(tmp_path):
    """Verify validation passes for valid uniform prediction CSV."""
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(100)],
        "prediction": np.linspace(0.01, 0.99, 100)
    })
    csv_file = tmp_path / "valid_preds.csv"
    df.to_csv(csv_file, index=False)
    
    is_valid, msg = validate_prediction_file(str(csv_file))
    assert is_valid is True
    assert "OK" in msg


def test_validate_prediction_file_nans(tmp_path):
    """Verify validation detects NaNs in prediction column."""
    df = pd.DataFrame({
        "id": ["id_1", "id_2", "id_3"],
        "prediction": [0.1, np.nan, 0.9]
    })
    csv_file = tmp_path / "nan_preds.csv"
    df.to_csv(csv_file, index=False)
    
    is_valid, msg = validate_prediction_file(str(csv_file))
    assert is_valid is False
    assert "NaN" in msg


def test_validate_prediction_file_out_of_bounds(tmp_path):
    """Verify validation detects values outside [0, 1]."""
    df = pd.DataFrame({
        "id": ["id_1", "id_2"],
        "prediction": [-0.05, 1.5]
    })
    csv_file = tmp_path / "oob_preds.csv"
    df.to_csv(csv_file, index=False)
    
    is_valid, msg = validate_prediction_file(str(csv_file))
    assert is_valid is False
    assert "bounds" in msg.lower()


def test_numerbay_publisher_no_credentials():
    """Verify publisher initializes gracefully in unconfigured mode."""
    with patch.dict(os.environ, {}, clear=True):
        pub = NumerbayPublisher(username="", password="")
        assert pub.is_configured is False
        res = pub.publish_predictions("cypherpole_hedged", "/fake/path.csv")
        assert res["status"] == "SKIPPED_UNCONFIGURED"


def test_numerbay_publisher_mock_upload():
    """Verify publisher handles mock upload to Numerbay API."""
    mock_api = MagicMock()
    mock_api.get_my_listings.return_value = [
        {"name": "cypherpole_hedged", "sku": "numerai-predictions-cypherpole_hedged", "id": 101}
    ]
    mock_api.upload_artifact.return_value = {"id": 999, "status": "uploaded"}

    pub = NumerbayPublisher(username="testuser", password="testpassword")
    pub.api = mock_api
    pub.is_configured = True

    df = pd.DataFrame({"id": ["a", "b"], "prediction": [0.2, 0.8]})
    res = pub.publish_dataframe("cypherpole_hedged", df)
    
    assert res["status"] == "SUCCESS"
    assert res["artifact_id"] == 999
    mock_api.upload_artifact.assert_called_once()


def test_syndicate_primary_models_unconfigured(tmp_path):
    """Verify syndicate_primary_models handles unconfigured environment safely."""
    from numerbay_publisher import syndicate_primary_models
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_file = data_dir / "predictions_cypherpole_round_1369.csv"
    pd.DataFrame({"id": ["id1"], "prediction": [0.5]}).to_csv(sample_file, index=False)

    with patch.dict(os.environ, {}, clear=True):
        res = syndicate_primary_models(str(tmp_path), models=["cypherpole"])
        assert isinstance(res, dict)
        assert res["status"] in ["SKIPPED_UNCONFIGURED", "NO_FILES_FOUND"]


def test_publish_predictions_passes_explicit_product_id(tmp_path):
    """Verify publisher passes exact integer product_id to upload_artifact."""
    from numerbay_publisher import MODEL_PRODUCT_ID_MAP
    mock_api = MagicMock()
    mock_api.upload_artifact.return_value = {"id": 1234, "status": "uploaded"}

    pub = NumerbayPublisher(username="testuser", password="testpassword")
    pub.api = mock_api
    pub.is_configured = True

    df = pd.DataFrame({"id": ["id_1"], "prediction": [0.75]})
    csv_file = tmp_path / "test_preds.csv"
    df.to_csv(csv_file, index=False)

    res = pub.publish_predictions("cypherpole", str(csv_file))
    assert res["status"] == "SUCCESS"
    assert mock_api.upload_artifact.call_count == 1
    call_kwargs = mock_api.upload_artifact.call_args.kwargs
    assert call_kwargs.get("product_id") == 1216
    assert call_kwargs.get("product_full_name") == "numerai-predictions-cypherpole"


def test_find_latest_prediction_file_with_target_round(tmp_path):
    """Verify target_round selector matches explicit round file."""
    from numerbay_publisher import find_latest_prediction_file
    round_1372 = tmp_path / "predictions_cypherpole_round_1372.csv"
    round_1373 = tmp_path / "predictions_cypherpole_round_1373.csv"
    
    pd.DataFrame({"id": ["1"], "prediction": [0.5]}).to_csv(round_1372, index=False)
    pd.DataFrame({"id": ["1"], "prediction": [0.6]}).to_csv(round_1373, index=False)

    match_1373 = find_latest_prediction_file(str(tmp_path), "cypherpole", target_round=1373)
    match_1372 = find_latest_prediction_file(str(tmp_path), "cypherpole", target_round=1372)
    match_none = find_latest_prediction_file(str(tmp_path), "cypherpole", target_round=9999)

    assert match_1373 == str(round_1373)
    assert match_1372 == str(round_1372)
    assert match_none is None


def test_model_product_id_map_invariants():
    """Verify MODEL_PRODUCT_ID_MAP integrity and known IDs."""
    from numerbay_publisher import MODEL_PRODUCT_ID_MAP
    assert isinstance(MODEL_PRODUCT_ID_MAP, dict)
    assert len(MODEL_PRODUCT_ID_MAP) >= 3
    for model_name, p_id in MODEL_PRODUCT_ID_MAP.items():
        assert isinstance(model_name, str) and len(model_name) > 0
        assert isinstance(p_id, int) and p_id > 0
    assert MODEL_PRODUCT_ID_MAP["cypherpole"] == 1216
    assert MODEL_PRODUCT_ID_MAP["cypherpole_hedged"] == 1215
    assert MODEL_PRODUCT_ID_MAP["cypherpole_vel"] == 1219


