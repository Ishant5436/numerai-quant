"""
Numerbay Storefront Copy & Promotion Optimizer.
Updates live Numerbay product listings with high-converting descriptions,
track record verification links, and prominent CYPHER100 trial coupon callouts.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def build_storefront_payload(
    model_name: str, product_id: int, monthly_price: float, trial_price: float
) -> Dict[str, Any]:
    """Builds optimized listing payload with coupon code and institutional pitch."""
    assert isinstance(model_name, str) and len(model_name) > 0, "Invalid model_name"
    assert isinstance(product_id, int) and product_id > 0, "product_id must be positive int"
    assert monthly_price > 0 and trial_price > 0, "Prices must be positive"

    desc = (
        f"Flagship Numerai Ensemble (Top 1.5% globally, corj60 +0.0607 vs +0.0192 platform median). "
        f"Delivers 3.1x higher alpha yield for tournament stakers. Verified track record: "
        f"https://numer.ai/{model_name}. FREE TRIAL: Use coupon code CYPHER100 at checkout "
        f"for a 100% discount on the single-round trial (limited to first 5 stakers)."
    )
    if "hedged" in model_name:
        desc = (
            f"Macro-Hedged Specialist. 50% linear QR feature neutralization against canonical risk factors. "
            f"Designed for downside drawdown protection during market shocks. Verified: https://numer.ai/{model_name}."
        )
    elif "vel" in model_name:
        desc = (
            f"Trend Velocity Specialist. Captures factor breakout momentum with 40% neutralization. "
            f"Uncorrelated multi-horizon alpha. Verified track record: https://numer.ai/{model_name}."
        )

    options = [
        {
            "is_on_platform": True,
            "description": "Single Round Trial (1 weekly round — use coupon code CYPHER100 for 100% off)",
            "quantity": 1,
            "price": trial_price,
            "currency": "NMR",
            "mode": "file",
            "is_active": True,
        },
        {
            "is_on_platform": True,
            "description": "Monthly Subscription (4 consecutive weekly tournament rounds)",
            "quantity": 4,
            "price": monthly_price,
            "currency": "NMR",
            "mode": "file",
            "is_active": True,
        },
    ]
    assert len(options) == 2, "Options array must contain exactly 2 tiers"
    return {"description": desc, "is_active": True, "is_daily": True, "options": options}


def push_storefront_update(
    pub: Any, model_name: str, product_id: int, monthly_price: float, trial_price: float
) -> Dict[str, Any]:
    """Transmits listing update to Numerbay backend via REST API."""
    assert hasattr(pub, "api") and pub.is_configured, "Publisher must be configured"
    assert isinstance(product_id, int) and product_id > 0, "Invalid product_id"

    payload = build_storefront_payload(model_name, product_id, monthly_price, trial_price)
    headers = {"Authorization": f"Bearer {pub.api.token}"}
    from numerbay import API_ENDPOINT_URL

    res = requests.put(f"{API_ENDPOINT_URL}/products/{product_id}", json=payload, headers=headers)
    assert res.status_code in [200, 400, 401, 403, 404], "Unexpected HTTP response code"
    return {
        "model": model_name,
        "product_id": product_id,
        "status": "UPDATED" if res.status_code == 200 else "ERROR",
        "code": res.status_code,
    }


def update_all_storefront_listings(dry_run: bool = False) -> List[Dict[str, Any]]:
    """Updates all 3 managed Numerbay listings with high-converting marketing copy."""
    assert isinstance(dry_run, bool), "dry_run must be boolean"
    from numerbay_publisher import NumerbayPublisher

    pub = NumerbayPublisher()
    targets = [
        {"name": "cypherpole", "id": 1216, "monthly": 0.70, "trial": 0.20},
        {"name": "cypherpole_hedged", "id": 1215, "monthly": 0.70, "trial": 0.20},
        {"name": "cypherpole_vel", "id": 1219, "monthly": 0.70, "trial": 0.20},
    ]
    results = []
    for t in targets:
        if dry_run or not pub.is_configured:
            results.append({"model": t["name"], "status": "DRY_RUN", "product_id": t["id"]})
        else:
            r = push_storefront_update(pub, t["name"], t["id"], t["monthly"], t["trial"])
            results.append(r)

    assert len(results) == len(targets), "Must process all targets"
    return results


if __name__ == "__main__":
    print("Executing Numerbay Storefront Optimization...")
    res = update_all_storefront_listings(dry_run=False)
    for item in res:
        print(f" - {item['model']} (ID: {item['product_id']}): {item['status']}")
