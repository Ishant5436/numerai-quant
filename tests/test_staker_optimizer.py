"""
Tests for scripts/staker_optimizer.py
Validates live staker synergy calculations, ROI projections, and Power of 10 invariants.
"""
import os
import pytest
import numpy as np
from scripts.staker_optimizer import (
    calculate_staker_synergy,
    compute_staking_roi,
    format_staker_case_study,
    audit_safety_invariants_local,
)


def test_calculate_staker_synergy():
    """Verify synergy calculations produce valid improvements and no NaNs."""
    metrics = calculate_staker_synergy(
        staker_name="bpa_apollo",
        staker_corj=0.0254,
        staker_stake=3888.3,
        cypher_corj=0.0607,
        blend_weight=0.25,
    )
    assert isinstance(metrics, dict), "Must return dictionary"
    assert "blended_corj" in metrics, "Missing blended_corj"
    assert metrics["blended_corj"] > 0.0254, "Blended correlation must exceed baseline"
    assert metrics["blended_corj"] < 0.0607, "Blended correlation must be bounded by cypherpole"
    assert metrics["corj_improvement_pct"] > 0.0, "Improvement percent must be positive"


def test_compute_staking_roi():
    """Verify staker NMR payout calculation and monthly ROI on 5 NMR fee."""
    roi_data = compute_staking_roi(
        stake_nmr=1000.0,
        baseline_corj=0.015,
        blended_corj=0.026,
        subscription_fee_nmr=5.0,
    )
    assert isinstance(roi_data, dict), "Must return dict"
    assert roi_data["net_monthly_gain_nmr"] > 0.0, "Net monthly gain must be positive for large stakers"
    assert roi_data["roi_pct"] > 0.0, "ROI must be positive"
    assert "break_even_stake" in roi_data, "Must include break even stake calculation"


def test_format_staker_case_study():
    """Verify formatted pitch report is unformatted anti-AI text without corporate fluff."""
    text = format_staker_case_study(
        staker_name="bpa_apollo",
        stake_nmr=3888.3,
        baseline_corj=0.0254,
        blended_corj=0.0342,
        net_gain_nmr=34.2,
        roi_pct=584.0,
    )
    assert isinstance(text, str), "Must return string"
    assert "bpa_apollo" in text, "Must mention staker name"
    assert "cypherpole" in text, "Must mention cypherpole"
    assert "3,888" in text or "3888" in text, "Must mention stake amount"
    assert not text.startswith("#"), "Must not start with markdown header (anti-AI format)"


def test_staker_optimizer_power_of_10():
    """Verify Power of 10 safety invariants on staker_optimizer.py."""
    passed = audit_safety_invariants_local()
    assert passed is True, "Power of 10 invariants must pass"
