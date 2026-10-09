"""
Unit and Invariant Tests for Autonomous Sales Telemetry Daemon.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import ast
import json
import os
import pytest


def test_load_and_save_sales_history(tmp_path):
    """Verify state persistence loads defaults when absent and persists atomically."""
    from scripts.sales_telemetry_daemon import load_sales_history, save_sales_history

    fake_file = str(tmp_path / "sales_history.json")
    history = load_sales_history(fake_file)
    assert isinstance(history, dict), "Must return dict"
    assert "known_order_ids" in history and "total_nmr_earned" in history, "Missing keys"

    history["known_order_ids"].append(999)
    history["total_nmr_earned"] = 5.0
    saved = save_sales_history(fake_file, history)
    assert saved is True, "Save must return True on success"

    reloaded = load_sales_history(fake_file)
    assert 999 in reloaded["known_order_ids"], "Reloaded history must contain order ID"
    assert reloaded["total_nmr_earned"] == 5.0, "Reloaded NMR earned must match"


def test_detect_new_orders():
    """Verify differential order detection identifies unrecorded sales."""
    from scripts.sales_telemetry_daemon import detect_new_orders

    history = {"known_order_ids": [101, 102], "total_nmr_earned": 3.0}
    current_sales = [
        {"id": 101, "price": 1.5, "product": {"name": "cypherpole"}},
        {"id": 102, "price": 1.5, "product": {"name": "cypherpole"}},
        {"id": 103, "price": 5.0, "product": {"name": "cypherpole"}},
    ]
    new_orders = detect_new_orders(history, current_sales)
    assert len(new_orders) == 1, "Must detect exactly 1 new order"
    assert new_orders[0]["id"] == 103, "Detected order ID must be 103"
    assert float(new_orders[0]["price"]) == 5.0, "Detected price must be 5.0"


def test_dispatch_sale_alert():
    """Verify alert formatting runs without exceptions."""
    from scripts.sales_telemetry_daemon import dispatch_sale_alert

    mock_order = {
        "id": 888,
        "price": 5.0,
        "currency": "NMR",
        "product": {"name": "cypherpole"},
        "buyer": {"username": "quant_whale"},
    }
    res = dispatch_sale_alert(mock_order, enable_system_notify=False)
    assert res is True, "Alert dispatch must return True"
    assert isinstance(mock_order["price"], float), "Price must be float"


def test_poll_and_update_sales_telemetry(tmp_path):
    """Verify complete polling cycle aggregates sales correctly."""
    from scripts.sales_telemetry_daemon import poll_and_update_sales_telemetry

    fake_file = str(tmp_path / "sales_history.json")

    class MockPublisher:
        is_configured = True

        def get_revenue_summary(self):
            return {"status": "ACTIVE", "total_sales": 2, "total_nmr_earned": 10.0}

        def get_sales(self):
            return [
                {"id": 201, "price": 5.0, "product": {"name": "cypherpole"}},
                {"id": 202, "price": 5.0, "product": {"name": "cypherpole"}},
            ]

    pub = MockPublisher()
    res = poll_and_update_sales_telemetry(pub, fake_file, notify=False)
    assert isinstance(res, dict), "Result must be a dict"
    assert res["status"] == "SUCCESS", "Status must be SUCCESS"
    assert res["new_orders_count"] == 2, "Must report 2 new orders"
    assert res["total_nmr_earned"] == 10.0, "Total NMR earned must be 10.0"


def test_dispatch_discord_webhook():
    """Verify Discord webhook dispatcher formats payload and handles invalid URL gracefully."""
    from scripts.sales_telemetry_daemon import dispatch_discord_webhook

    order = {"id": 1, "price": 5.0, "currency": "NMR", "product": {"name": "cypherpole"}}
    res = dispatch_discord_webhook(order, webhook_url="invalid-url")
    assert res is False
    assert isinstance(order, dict)



def test_power_of_10_invariants():
    """Verify that sales_telemetry_daemon.py adheres to Holzmann Power of 10 rules."""
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "sales_telemetry_daemon.py"
    )
    assert os.path.exists(module_path), f"Module not found at {module_path}"

    with open(module_path, "r") as f:
        tree = ast.parse(f.read())

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) >= 4, "Must contain at least 4 functions"

    for fn in functions:
        line_length = fn.end_lineno - fn.lineno + 1
        assert line_length <= 60, f"Function {fn.name} exceeds 60 lines ({line_length})"

        assert_count = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
        assert assert_count >= 2, (
            f"Function {fn.name} has only {assert_count} asserts (minimum 2)"
        )
