import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

from evaluate_fleet_60d import calc_sharpe
from fleet_submit import _save_checkpoint, _load_checkpoint
from chimera.alpha_vault import AlphaVault, AlphaEntry

def test_calc_sharpe_single_era():
    """Verify single era returns 0.0 instead of dividing by zero (degrees of freedom < 2)."""
    s = pd.Series([0.05])
    sharpe = calc_sharpe(s)
    assert sharpe == 0.0
    assert isinstance(sharpe, float)

def test_calc_sharpe_zero_variance():
    """Verify series with 0 standard deviation returns 0.0 without ZeroDivisionError."""
    s = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03])
    sharpe = calc_sharpe(s)
    assert sharpe == 0.0
    assert not np.isnan(sharpe)

def test_calc_sharpe_valid_distribution():
    """Verify standard Sharpe ratio formula matches manual calculation."""
    vals = [0.01, 0.03, -0.01, 0.04, 0.02]
    s = pd.Series(vals)
    expected_mean = np.mean(vals)
    expected_std = np.std(vals, ddof=1)
    expected_sharpe = float(expected_mean / expected_std) * np.sqrt(52.0)

    sharpe = calc_sharpe(s)
    assert np.isclose(sharpe, expected_sharpe, atol=1e-5)
    assert np.isfinite(sharpe)

def test_atomic_checkpoint_persistence():
    """Verify _save_checkpoint atomically writes without file corruption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = os.path.join(tmpdir, "checkpoint.json")
        models = {"model_alpha", "model_beta", "model_gamma"}

        _save_checkpoint(ckpt_file, models)
        assert os.path.exists(ckpt_file)

        loaded = _load_checkpoint(ckpt_file)
        assert loaded == models

def test_alpha_vault_atomic_save_and_reload():
    """Verify AlphaVault save writes atomically and reloads cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = os.path.join(tmpdir, "test_vault.json")
        vault = AlphaVault(vault_path=vault_path)
        entry = AlphaEntry(
            name="chimera_alpha_test",
            formula="add(feat(0), feat(1))",
            instructions=[(1, 0, 0, 0, 0, 0.0), (1, 1, 0, 0, 1, 0.0), (3, 2, 0, 1, 0, 0.0)],
            sharpe=1.25,
            mean_corr=0.035,
            max_factor_corr=0.08,
            positive_era_ratio=0.65
        )
        vault.add_entry(entry)
        vault.save()

        assert os.path.exists(vault_path)
        reloaded = AlphaVault.load(vault_path=vault_path)
        assert len(reloaded.entries) == 1
        assert reloaded.entries[0].formula == "add(feat(0), feat(1))"
        assert reloaded.entries[0].sharpe == 1.25
