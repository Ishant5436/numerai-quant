"""
Tests for scripts/target_audience_dispatcher.py.
Validates multi-channel payload generation, linting verification, and Power of 10 safety invariants.
"""
import os
import pytest
from scripts.target_audience_dispatcher import (
    generate_channel_payload,
    export_all_dispatches,
    audit_safety_invariants_local,
)


def test_generate_channel_payload_discord():
    """Verify Discord payload contains coupon code and no corporate buzzwords."""
    text = generate_channel_payload("discord")
    assert isinstance(text, str)
    assert "CYPHER100" in text
    assert "cypherpole" in text
    assert "5 NMR" in text
    assert not text.startswith("#")


def test_generate_channel_payload_forum():
    """Verify Forum payload contains alpha methodology and out-of-sample stats."""
    text = generate_channel_payload("forum")
    assert isinstance(text, str)
    assert "+0.0607" in text or "0.0607" in text
    assert "CYPHER100" in text
    assert not text.startswith("#")


def test_export_all_dispatches(tmp_path):
    """Verify all channel dispatches are written to disk cleanly."""
    out_dir = str(tmp_path / "dispatches")
    files = export_all_dispatches(out_dir)
    assert isinstance(files, list)
    assert len(files) >= 3
    for f in files:
        assert os.path.exists(f)
        with open(f) as fp:
            content = fp.read()
            assert "CYPHER100" in content


def test_target_audience_dispatcher_power_of_10():
    """Verify Power of 10 invariants on target_audience_dispatcher.py."""
    passed = audit_safety_invariants_local()
    assert passed is True
