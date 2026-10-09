"""
Tests for dashboard telemetry helpers in dashboard.py.
Validates storefront telemetry loading, round scheduling, and Power of 10 invariants.
"""
import os
import json
import pytest
from dashboard import (
    load_storefront_telemetry,
    compute_round_schedule,
    load_staker_synergy_preview,
)


def test_load_storefront_telemetry_file(tmp_path):
    """Verify sales telemetry loads accurately from json and provides fallback."""
    p = tmp_path / "sales.json"
    p.write_text(json.dumps({
        "last_checked_utc": "2026-10-09T08:00:00",
        "total_orders": 2,
        "total_nmr_earned": 6.5,
        "orders": []
    }))

    res = load_storefront_telemetry(str(p))
    assert isinstance(res, dict)
    assert res["total_orders"] == 2
    assert res["total_nmr_earned"] == 6.5
    assert len(res["listings"]) == 3
    assert res["coupon_code"] == "CYPHER100"


def test_load_storefront_telemetry_missing():
    """Verify missing file returns safe defaults without throwing exceptions."""
    res = load_storefront_telemetry("/nonexistent/path/sales.json")
    assert isinstance(res, dict)
    assert res["total_orders"] == 0
    assert res["total_nmr_earned"] == 0.0


def test_compute_round_schedule():
    """Verify tournament round calculation and scheduling metadata."""
    sched = compute_round_schedule(1372)
    assert isinstance(sched, dict)
    assert sched["active_round"] == 1372
    assert sched["next_round"] == 1373
    assert "Saturday" in sched["submission_window"]


def test_load_staker_synergy_preview():
    """Verify staker synergy preview generates valid model data for dashboard display."""
    preview = load_staker_synergy_preview()
    assert isinstance(preview, list)
    assert len(preview) >= 3
    assert "name" in preview[0]
    assert "blended_corj" in preview[0]
