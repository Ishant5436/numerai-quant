"""
Numerbay Promotional Coupon and Trial Code Manager.
Automates generation and lifecycle management of trial discount coupons for stakers.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2, bounded loops.
"""
from typing import Dict, Any, Tuple
import requests


def validate_coupon_payload(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates coupon specification parameters against marketplace constraints."""
    assert isinstance(spec, dict), "Spec must be a dictionary"
    assert len(spec) > 0, "Spec cannot be empty"

    code = spec.get("code")
    if not isinstance(code, str) or len(code.strip()) == 0:
        return False, "Invalid or missing coupon code"

    discount_pct = spec.get("discount_percent")
    if not isinstance(discount_pct, int) or discount_pct < 1 or discount_pct > 100:
        return False, f"Discount percent {discount_pct} outside valid range [1, 100]"

    prod_ids = spec.get("applicable_product_ids")
    if not isinstance(prod_ids, list) or len(prod_ids) == 0:
        return False, "Applicable product IDs must be a non-empty list"

    qty = spec.get("quantity_total")
    if not isinstance(qty, int) or qty <= 0:
        return False, f"Invalid quantity total: {qty}"

    return True, "Payload valid and ready for registration"


def build_trial_coupon_spec(
    code: str, product_id: int, quantity: int
) -> Dict[str, Any]:
    """Constructs a conforming 100% off trial coupon specification."""
    assert isinstance(code, str) and len(code.strip()) > 0, "Invalid coupon code"
    assert isinstance(product_id, int) and product_id > 0, "Invalid product ID"
    assert isinstance(quantity, int) and quantity > 0, "Quantity must be positive"

    spec = {
        "code": code.strip().upper(),
        "applicability": "products",
        "applicable_product_ids": [product_id],
        "discount_percent": 100,
        "max_discount": 1.5,
        "quantity_total": quantity,
        "discount_mode": "percent",
    }
    assert spec["discount_percent"] == 100, "Trial discount must be exactly 100%"
    assert spec["max_discount"] > 0.0, "Max discount must be positive"
    return spec


def create_coupon_on_numerbay(
    publisher, spec: Dict[str, Any], dry_run: bool = False
) -> Dict[str, Any]:
    """Submits promotional coupon creation request to Numerbay API."""
    assert publisher is not None, "Publisher instance cannot be None"
    assert isinstance(spec, dict), "Coupon spec must be a dictionary"
    assert isinstance(dry_run, bool), "dry_run must be boolean"

    is_valid, msg = validate_coupon_payload(spec)
    if not is_valid:
        return {"status": "INVALID_SPEC", "error": msg}

    code = str(spec.get("code"))
    if dry_run or not getattr(publisher, "is_configured", False):
        return {"status": "DRY_RUN", "code": code, "spec": spec}

    from numerbay import API_ENDPOINT_URL

    username = getattr(publisher, "username", "cypherpole")
    token = getattr(publisher.api, "token", "")
    url = f"{API_ENDPOINT_URL}/coupons/{username}"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"coupon_in": spec, "message": "Promotional trial access for tournament stakers"}

    try:
        res = requests.post(url, json=body, headers=headers, timeout=10)
        if res.status_code in [200, 201]:
            return {"status": "SUCCESS", "code": code, "data": res.json()}
        return {"status": "FAILED", "code": code, "http_code": res.status_code, "detail": res.text}
    except Exception as err:
        return {"status": "ERROR", "code": code, "error": str(err)}


if __name__ == "__main__":
    import os
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from numerbay_publisher import NumerbayPublisher

    pub = NumerbayPublisher()
    trial_spec = build_trial_coupon_spec(code="CYPHER100", product_id=1216, quantity=5)
    print("=" * 60)
    print("Numerbay Trial Coupon Manager")
    print("=" * 60)
    out = create_coupon_on_numerbay(pub, trial_spec, dry_run=False)
    print(f"Status: {out.get('status')} | Code: {out.get('code')}")
    if out.get("detail"):
        print(f"Detail: {out.get('detail')}")
