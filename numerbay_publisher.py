"""
Numerbay Zero-Capital Publisher Bridge.
Enables autonomous syndication and monetization of verified Numerai model predictions
by publishing weekly prediction artifacts to the Numerbay decentralized marketplace.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def validate_prediction_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validates DataFrame columns, bounds, and absence of NaNs."""
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert len(df) > 0, "DataFrame cannot be empty"

    if "prediction" not in df.columns:
        return False, "Missing required 'prediction' column"

    nan_count = int(df["prediction"].isna().sum())
    if nan_count > 0:
        return False, f"Found {nan_count} NaN values in prediction column"

    min_val = float(df["prediction"].min())
    max_val = float(df["prediction"].max())
    if min_val < 0.0 or max_val > 1.0:
        return False, f"Predictions out of bounds: min={min_val}, max={max_val}"

    return True, "Payload valid and strictly uniform in [0, 1] (OK)"


def validate_prediction_file(file_path: str) -> Tuple[bool, str]:
    """Inspects a saved prediction CSV for submission validity."""
    assert isinstance(file_path, str) and len(file_path) > 0, "Invalid file path"
    assert os.path.exists(file_path), f"File does not exist: {file_path}"

    try:
        df = pd.read_csv(file_path, nrows=50000)
        return validate_prediction_dataframe(df)
    except Exception as err:
        return False, f"File read error: {err}"


class NumerbayPublisher:
    """
    Manages autonomous artifact publishing and sales telemetry on Numerbay.ai.
    Operates gracefully when credentials are not configured.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        user = username if username is not None else os.environ.get("NUMERBAY_USERNAME", "")
        pwd = password if password is not None else os.environ.get("NUMERBAY_PASSWORD", "")
        
        assert isinstance(user, str), "Username must be a string"
        assert isinstance(pwd, str), "Password must be a string"

        self.username = user.strip()
        self.password = pwd.strip()
        self.is_configured = bool(self.username and self.password)
        self.api = None

        if self.is_configured:
            try:
                from numerbay import NumerBay
                self.api = NumerBay(username=self.username, password=self.password)
            except Exception as init_err:
                print(f"[WARN] NumerBay API initialization failed: {init_err}")
                self.is_configured = False

    def get_listings(self) -> List[Dict[str, Any]]:
        """Retrieves active product listings associated with the seller account."""
        assert isinstance(self.is_configured, bool), "State invariant violated"
        if not self.is_configured or self.api is None:
            return []
        try:
            listings = self.api.get_my_listings()
            assert isinstance(listings, list), "Listings response must be a list"
            return listings[:50]  # Bounded output
        except Exception as err:
            print(f"[ERROR] Failed fetching Numerbay listings: {err}")
            return []

    def publish_dataframe(self, model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Uploads in-memory prediction DataFrame to Numerbay product listing."""
        assert isinstance(model_name, str) and len(model_name) > 0, "Invalid model name"
        assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"

        is_valid, msg = validate_prediction_dataframe(df)
        if not is_valid:
            return {"status": "INVALID_PAYLOAD", "error": msg}

        if not self.is_configured or self.api is None:
            return {"status": "SKIPPED_UNCONFIGURED", "message": "NumerBay credentials not set"}

        try:
            target_sku = f"numerai-predictions-{model_name.lower()}"
            res = self.api.upload_artifact(df=df, product_full_name=target_sku)
            artifact_id = res.get("id") if isinstance(res, dict) else None
            return {"status": "SUCCESS", "artifact_id": artifact_id, "sku": target_sku}
        except Exception as err:
            return {"status": "UPLOAD_FAILED", "error": str(err)}

    def publish_predictions(self, model_name: str, file_path: str) -> Dict[str, Any]:
        """Uploads saved prediction CSV file to Numerbay product listing."""
        assert isinstance(model_name, str) and len(model_name) > 0, "Invalid model name"
        assert isinstance(file_path, str) and len(file_path) > 0, "Invalid file path"

        if not self.is_configured or self.api is None:
            return {"status": "SKIPPED_UNCONFIGURED", "message": "NumerBay credentials not set"}

        if not os.path.exists(file_path):
            return {"status": "FILE_NOT_FOUND", "error": f"Missing file: {file_path}"}

        is_valid, msg = validate_prediction_file(file_path)
        if not is_valid:
            return {"status": "INVALID_PAYLOAD", "error": msg}

        try:
            target_sku = f"numerai-predictions-{model_name.lower()}"
            res = self.api.upload_artifact(file_path=file_path, product_full_name=target_sku)
            artifact_id = res.get("id") if isinstance(res, dict) else None
            return {"status": "SUCCESS", "artifact_id": artifact_id, "sku": target_sku}
        except Exception as err:
            return {"status": "UPLOAD_FAILED", "error": str(err)}

    def get_revenue_summary(self) -> Dict[str, Any]:
        """Calculates total sales volume and active subscriber count."""
        assert isinstance(self.is_configured, bool), "State invariant violated"
        if not self.is_configured or self.api is None:
            return {"total_sales": 0, "total_nmr_earned": 0.0, "status": "UNCONFIGURED"}

        try:
            sales = self.api.get_my_sales(active_only=False)
            assert isinstance(sales, list), "Sales must be a list"
            total_sales = len(sales)
            total_nmr = sum(float(s.get("price", 0.0)) for s in sales[:500])
            return {
                "total_sales": total_sales,
                "total_nmr_earned": total_nmr,
                "status": "ACTIVE"
            }
        except Exception as err:
            return {"status": "QUERY_ERROR", "error": str(err)}
