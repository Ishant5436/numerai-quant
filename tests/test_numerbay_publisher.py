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
