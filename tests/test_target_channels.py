"""
Unit and invariant tests for the Target Channel Blueprint and Registry module.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import ast
import pytest
from scripts.channel_targeting_engine import get_channel_registry, format_channel_action_guide


def test_get_channel_registry_contains_primary_channels():
    """Verify registry contains all 4 verified staker community channels."""
    reg = get_channel_registry()
    assert isinstance(reg, list)
    assert len(reg) >= 4

    channel_names = [c["name"] for c in reg]
    assert any("discord" in n.lower() for n in channel_names)
    assert any("forum" in n.lower() for n in channel_names)
    assert any("leaderboard" in n.lower() for n in channel_names)
    assert any("reddit" in n.lower() for n in channel_names)

    for c in reg:
        assert "url" in c and len(c["url"]) > 0
        assert "intent_level" in c and c["intent_level"] in ["CRITICAL", "HIGH", "MEDIUM"]
        assert "actionable_copy_file" in c


def test_format_channel_action_guide_generates_markdown():
    """Verify channel action guide formats valid markdown table and links."""
    guide = format_channel_action_guide()
    assert isinstance(guide, str)
    assert len(guide) > 200
    assert "discord.gg/numerai" in guide or "forum.numer.ai" in guide
    assert "CYPHER100" in guide


def test_power_of_10_invariants():
    """Verify Holzmann Power of 10 rules across channel_targeting_engine.py."""
    filepath = "scripts/channel_targeting_engine.py"
    assert os.path.exists(filepath), f"File {filepath} must exist"
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) >= 2, f"{filepath} must contain at least 2 functions"
    for fn in funcs:
        lines = fn.end_lineno - fn.lineno + 1
        asserts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
        assert lines <= 60, f"Function {fn.name} in {filepath} exceeds 60 lines ({lines})"
        assert asserts >= 2, f"Function {fn.name} in {filepath} has fewer than 2 asserts ({asserts})"
