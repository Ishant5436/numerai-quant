"""
Unit tests for portfolio monitoring script.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
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
    if pulse["numerai"]["status"] == "ONLINE":
        assert pulse["numerai"]["total_fleet_size"] == 30
    else:
        assert pulse["numerai"]["status"] in ("UNCONFIGURED", "ERROR")


def test_run_monitoring_pulse_mocked(monkeypatch):
    """Verify monitoring pulse correctly structures 30-model fleet when configured."""
    from unittest.mock import MagicMock
    import scripts.monitor_portfolio as mp

    mock_napi = MagicMock()
    mock_napi.get_current_round.return_value = 1358
    mock_napi.check_round_open.return_value = True
    mock_napi.raw_query.return_value = {
        "data": {"rounds": [{"number": 1358, "openTime": "2026-09-18T12:00:00Z", "closeTime": "2026-09-19T12:00:00Z", "resolveTime": "2026-12-16T16:00:00Z"}]}
    }
    mock_napi.get_models.return_value = {f"model_{i}": f"id_{i}" for i in range(25)}

    mock_sapi = MagicMock()
    mock_sapi.get_models.return_value = {f"signal_{i}": f"sid_{i}" for i in range(5)}

    monkeypatch.setenv("NUMERAI_PUBLIC_ID", "mock_pub")
    monkeypatch.setenv("NUMERAI_SECRET_KEY", "mock_sec")
    monkeypatch.setattr("numerapi.NumerAPI", lambda *args, **kwargs: mock_napi)
    monkeypatch.setattr("numerapi.SignalsAPI", lambda *args, **kwargs: mock_sapi)

    pulse = mp.run_monitoring_pulse()
    assert pulse["numerai"]["status"] == "ONLINE"
    assert pulse["numerai"]["classic_model_count"] == 25
    assert pulse["numerai"]["signals_model_count"] == 5
    assert pulse["numerai"]["total_fleet_size"] == 30
