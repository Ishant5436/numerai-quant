#!/usr/bin/env python3
"""
Unit and Mock Tests for the Numerbay Manager Module.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from numerbay_manager import validate_evm_address, NumerbayManager


def test_validate_evm_address_valid():
    """Verify validation passes for valid 40-hex character EVM addresses."""
    valid_addr = "0x71C84941916321287cE1772F2fB1E97C475a8984"
    assert validate_evm_address(valid_addr) is True
    assert validate_evm_address(valid_addr.lower()) is True


def test_validate_evm_address_invalid():
    """Verify validation rejects malformed, short, or non-hex addresses."""
    assert validate_evm_address("0x123") is False
    assert validate_evm_address("71C84941916321287cE1772F2fB1E97C475a8984") is False
    assert validate_evm_address("0xZZZZ4941916321287cE1772F2fB1E97C475a8984") is False
    assert validate_evm_address("") is False


def test_get_marketplace_status_mocked():
    """Verify parsing of marketplace status from mocked API responses."""
    mgr = NumerbayManager(token="fake_token")
    mock_listings = [
        {
            "id": 1216,
            "name": "cypherpole",
            "sku": "numerai-predictions-cypherpole",
            "options": [{"id": 1777, "price": 1.0, "currency": "NMR"}],
        }
    ]
    mock_user_resp = MagicMock()
    mock_user_resp.status_code = 200
    mock_user_resp.json.return_value = {
        "username": "cypherpole",
        "default_receiving_wallet_address": "0x71C84941916321287cE1772F2fB1E97C475a8984",
    }

    with patch.object(mgr.publisher, "get_listings", return_value=mock_listings):
        with patch("requests.get", return_value=mock_user_resp):
            status = mgr.get_marketplace_status()
            assert isinstance(status, dict)
            assert status["username"] == "cypherpole"
            assert len(status["products"]) == 1
            assert status["products"][0]["price"] == 1.0


def test_set_receiving_wallet_mocked():
    """Verify updating receiving wallet sends correct PUT request."""
    mgr = NumerbayManager(token="fake_token")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"default_receiving_wallet_address": "0x71C84941916321287cE1772F2fB1E97C475a8984"}

    target_wallet = "0x71C84941916321287cE1772F2fB1E97C475a8984"
    with patch("requests.put", return_value=mock_resp) as mock_put:
        success, msg = mgr.set_receiving_wallet(target_wallet)
        assert success is True, f"Failed: {msg}"
        mock_put.assert_called_once()
