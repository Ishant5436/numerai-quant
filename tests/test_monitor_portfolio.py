"""
Unit tests for portfolio monitoring script.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import pytest
from scripts.monitor_portfolio import check_hackathon_pipeline_countdown, run_monitoring_pulse


def test_check_hackathon_pipeline_countdown():
    """Verify countdown returns all 5 tracks with non-negative hours."""
    report = check_hackathon_pipeline_countdown()
    assert len(report) == 5, f"Expected 5 tracks, got {len(report)}"
    assert "buidl_ctc_fall_2026" in report
    assert "somnia_dreamdex" in report
    assert "keeperhub_mcp_bounty" in report
    assert "optimism_foundation_grant" in report
    assert "weex_ai_wars_ii" in report

    for k, v in report.items():
        assert v["hours_remaining"] >= 0.0, f"Negative hours in {k}"
        assert isinstance(v["is_past_deadline"], bool)


def test_run_monitoring_pulse():
    """Verify monitoring pulse generates telemetry dictionary and persists to disk."""
    pulse = run_monitoring_pulse()
    assert "timestamp_utc" in pulse
    assert "numerbay" in pulse
    assert "numerai" in pulse
    assert "hackathons" in pulse
    assert pulse["numerai"]["total_fleet_size"] == 30
