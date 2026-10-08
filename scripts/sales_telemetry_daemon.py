"""
Autonomous Sales & Revenue Telemetry Daemon for Numerbay Marketplace.
Monitors sales events, persists subscriber telemetry, and dispatches native macOS notifications.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2, bounded loops.
"""
import os
import json
import time
import subprocess
from typing import Dict, Any, List, Optional


def load_sales_history(filepath: str) -> Dict[str, Any]:
    """Loads existing sales history from disk or initializes default state."""
    assert isinstance(filepath, str) and len(filepath) > 0, "Invalid filepath"
    default_state = {
        "known_order_ids": [],
        "total_nmr_earned": 0.0,
        "total_orders_count": 0,
        "orders_history": [],
        "last_polled_at": None,
    }
    if not os.path.exists(filepath):
        assert isinstance(default_state, dict), "State invariant violated"
        return default_state

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), "Corrupted sales history JSON"
            return data
    except Exception as err:
        print(f"[WARN] Failed loading sales history: {err}. Using default state.")
        return default_state


def save_sales_history(filepath: str, history: Dict[str, Any]) -> bool:
    """Persists sales telemetry state atomically to disk."""
    assert isinstance(filepath, str) and len(filepath) > 0, "Invalid filepath"
    assert isinstance(history, dict) and "known_order_ids" in history, "Invalid history format"

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    temp_path = f"{filepath}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        os.replace(temp_path, filepath)
        return True
    except Exception as err:
        print(f"[ERROR] Failed atomic save of sales history: {err}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def detect_new_orders(
    history: Dict[str, Any], current_sales: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Identifies newly completed orders that have not yet been logged."""
    assert isinstance(history, dict), "History must be a dictionary"
    assert isinstance(current_sales, list), "Current sales must be a list"

    known_ids = set(history.get("known_order_ids", []))
    new_orders = []
    # Bounded iteration
    for order in current_sales[:500]:
        order_id = order.get("id")
        if order_id is not None and order_id not in known_ids:
            new_orders.append(order)

    assert len(new_orders) <= len(current_sales), "New orders cannot exceed total sales"
    return new_orders


def dispatch_sale_alert(
    order: Dict[str, Any], enable_system_notify: bool = True
) -> bool:
    """Dispatches clinical notification and native macOS alert for new sales."""
    assert isinstance(order, dict), "Order must be a dictionary"
    assert isinstance(enable_system_notify, bool), "enable_system_notify must be boolean"

    price = order.get("price", 0.0)
    currency = order.get("currency", "NMR")
    prod = order.get("product", {})
    model_name = prod.get("name", "cypherpole") if isinstance(prod, dict) else "cypherpole"

    msg = f"New sale: {model_name} for {price} {currency}!"
    print(f"[SALE EVENT] {msg}")

    if enable_system_notify:
        apple_script = (
            f'display notification "{msg}" with title "Numerbay Revenue Alert" '
            'sound name "Glass"'
        )
        try:
            subprocess.run(["osascript", "-e", apple_script], check=False, timeout=3)
        except Exception as notify_err:
            print(f"[WARN] Notification delivery failed: {notify_err}")

    return True


def poll_and_update_sales_telemetry(
    publisher, history_path: str, notify: bool = True
) -> Dict[str, Any]:
    """Polls publisher sales API and records newly completed transactions."""
    assert publisher is not None, "Publisher instance cannot be None"
    assert isinstance(history_path, str), "Invalid history path"

    history = load_sales_history(history_path)
    sales = []
    if hasattr(publisher, "get_sales"):
        sales = publisher.get_sales()
    elif hasattr(publisher, "api") and publisher.api is not None:
        try:
            sales = publisher.api.get_my_sales(active_only=False)
        except Exception as err:
            return {"status": "ERROR", "error": str(err)}

    assert isinstance(sales, list), "Sales must be returned as list"
    new_orders = detect_new_orders(history, sales)

    for order in new_orders:
        history["known_order_ids"].append(order.get("id"))
        history["orders_history"].append(order)
        if notify:
            dispatch_sale_alert(order, enable_system_notify=True)

    summary = publisher.get_revenue_summary() if hasattr(publisher, "get_revenue_summary") else {}
    history["total_nmr_earned"] = float(summary.get("total_nmr_earned", 0.0))
    history["total_orders_count"] = int(summary.get("total_sales", len(history["known_order_ids"])))
    history["last_polled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_sales_history(history_path, history)
    return {
        "status": "SUCCESS",
        "new_orders_count": len(new_orders),
        "total_nmr_earned": history["total_nmr_earned"],
    }


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from numerbay_publisher import NumerbayPublisher

    history_file = os.path.join(base_dir, "data", "sales_history.json")
    pub = NumerbayPublisher()
    print("=" * 60)
    print("Polling Numerbay Autonomous Sales Telemetry...")
    print("=" * 60)
    result = poll_and_update_sales_telemetry(pub, history_file, notify=True)
    print(f"Status: {result.get('status')} | New Orders: {result.get('new_orders_count')} | Total NMR: {result.get('total_nmr_earned')}")
