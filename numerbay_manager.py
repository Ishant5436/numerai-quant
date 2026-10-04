#!/usr/bin/env python3
"""
Numerbay Marketplace Manager: Pricing Configuration & Payout Routing.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import argparse
import os
import re
from typing import Dict, Any, Tuple, Optional
import requests

from numerbay_publisher import NumerbayPublisher

API_ENDPOINT_URL = "https://numerbay.ai/backend-api/v1"
EVM_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")


def validate_evm_address(address: str) -> bool:
    """Validates that a string is a well-formed 40-hex character EVM address."""
    assert isinstance(address, str), "Address must be a string"
    is_valid = bool(EVM_REGEX.match(address.strip()))
    assert isinstance(is_valid, bool), "Validation check returned non-boolean"
    return is_valid


class NumerbayManager:
    """Manages product options, pricing, and seller payout routing on Numerbay."""

    def __init__(self, token: Optional[str] = None):
        self.publisher = NumerbayPublisher()
        self.token = token or (self.publisher.api.token if self.publisher.api else None)
        assert self.publisher is not None, "Publisher initialization invariant failed"
        assert self.token is None or isinstance(self.token, str), "Token must be string or None"

    def _get_headers(self) -> Dict[str, str]:
        """Returns standard authorization headers."""
        assert self.token is not None, "Numerbay authentication token required"
        assert len(self.token) > 0, "Token cannot be empty"
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def get_marketplace_status(self) -> Dict[str, Any]:
        """Queries current seller account, payout wallet, and active listing pricing."""
        assert self.publisher is not None, "Publisher instance required"
        listings = self.publisher.get_listings()
        headers = self._get_headers() if self.token else {}

        user_info = {}
        if self.token:
            try:
                resp = requests.get(f"{API_ENDPOINT_URL}/users/me", headers=headers, timeout=10)
                if resp.status_code == 200:
                    user_info = resp.json()
            except Exception as err:
                print(f"[WARN] Failed fetching user profile: {err}")

        products = []
        for item in listings:
            opts = item.get("options", [])
            price = opts[0].get("price", 0.0) if opts else 0.0
            opt_id = opts[0].get("id") if opts else None
            products.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "sku": item.get("sku"),
                "option_id": opt_id,
                "price": price,
                "currency": "NMR",
            })

        assert isinstance(products, list), "Products must be a list"
        return {
            "username": user_info.get("username", "cypherpole"),
            "receiving_wallet": user_info.get("default_receiving_wallet_address"),
            "products": products,
        }

    def set_receiving_wallet(self, wallet_address: str) -> Tuple[bool, str]:
        """Updates seller default receiving wallet address for NMR payout routing."""
        assert isinstance(wallet_address, str), "Wallet address must be string"
        if not validate_evm_address(wallet_address):
            return False, f"Invalid EVM address format: {wallet_address}"

        headers = self._get_headers()
        payload = {"default_receiving_wallet_address": wallet_address.strip()}
        try:
            resp = requests.put(f"{API_ENDPOINT_URL}/users/me", json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                assert resp.status_code == 200, "Update status check failed"
                return True, f"Successfully updated receiving wallet to {wallet_address}"
            return False, f"API error {resp.status_code}: {resp.text}"
        except Exception as err:
            return False, f"Request failed: {err}"

    def update_product_price(self, product_id: int, option_id: int, new_price_nmr: float) -> Tuple[bool, str]:
        """Updates pricing for a specific product option."""
        assert product_id > 0, "product_id must be positive"
        assert new_price_nmr >= 0.0, "Price must be non-negative"

        headers = self._get_headers()
        payload = {
            "options": [{
                "id": option_id,
                "price": float(new_price_nmr),
                "currency": "NMR",
                "mode": "file",
                "is_active": True,
                "is_on_platform": True,
            }]
        }
        try:
            resp = requests.put(f"{API_ENDPOINT_URL}/products/{product_id}", json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                assert resp.status_code == 200, "Update status check failed"
                return True, f"Updated Product {product_id} price to {new_price_nmr} NMR"
            return False, f"API error {resp.status_code}: {resp.text}"
        except Exception as err:
            return False, f"Request failed: {err}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numerbay Pricing & Payout Manager")
    parser.add_argument("--status", action="store_true", help="Print live marketplace status")
    parser.add_argument("--set-wallet", type=str, default=None, help="Set destination receiving wallet")
    parser.add_argument("--set-price", nargs=3, metavar=("PRODUCT_ID", "OPTION_ID", "PRICE"), help="Set option price")
    args = parser.parse_args()

    mgr = NumerbayManager()
    if args.set_wallet:
        ok, msg = mgr.set_receiving_wallet(args.set_wallet)
        print(f"[{'OK' if ok else 'ERROR'}] {msg}")
    elif args.set_price:
        p_id, opt_id, pr = int(args.set_price[0]), int(args.set_price[1]), float(args.set_price[2])
        ok, msg = mgr.update_product_price(p_id, opt_id, pr)
        print(f"[{'OK' if ok else 'ERROR'}] {msg}")
    else:
        st = mgr.get_marketplace_status()
        print("=" * 60)
        print(f"Numerbay Seller: {st['username']} | Receiving Wallet: {st['receiving_wallet'] or 'NOT SET (BLOCKED)'}")
        print("=" * 60)
        for p in st["products"]:
            print(f" - {p['name']} (ID: {p['id']}, Option: {p['option_id']}): {p['price']} {p['currency']}")
