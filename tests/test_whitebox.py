"""
White-Box Test Suite for Numerai Quant Architecture
Focuses on mathematical invariants of rank_01 percentile transforms,
linear feature neutralization projections, and ensemble weight constraints.
"""

import pytest
import numpy as np
import pandas as pd
from neutralize import rank_01, neutralize


# ==============================================================================
# 1. Percentile Rank Transformation Invariants (White-Box)
# ==============================================================================

def test_whitebox_rank_01_uniform_distribution():
    """White-box: rank_01 maps standard inputs to symmetric [0, 1] uniform percentiles."""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    res = rank_01(data)
    expected = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    np.testing.assert_allclose(res, expected, atol=1e-6)
    assert np.all(res >= 0.0)
    assert np.all(res <= 1.0)


def test_whitebox_rank_01_nan_safety_isolated():
    """White-box: arrays containing NaNs assign neutral 0.5 to NaNs and rank valid elements."""
    data = np.array([10.0, np.nan, 30.0, np.nan, 20.0])
    res = rank_01(data)

    # Indices 1 and 3 must be exactly 0.5
    assert res[1] == 0.5
    assert res[3] == 0.5

    # Valid values 10, 30, 20 have 3 elements -> ranks (1-0.5)/3=1/6, (3-0.5)/3=5/6, (2-0.5)/3=3/6=0.5
    assert np.isclose(res[0], 1.0 / 6.0)
    assert np.isclose(res[4], 3.0 / 6.0)
    assert np.isclose(res[2], 5.0 / 6.0)


def test_whitebox_rank_01_all_nans_graceful():
    """White-box: an array with 100% NaNs returns neutral 0.5 without exception."""
    data = np.array([np.nan, np.nan, np.nan])
    res = rank_01(data)
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(res, expected)


def test_whitebox_rank_01_ties_handling():
    """White-box: identical values (ties) receive identical average rank 0.5."""
    data = np.array([100.0, 100.0, 100.0, 100.0])
    res = rank_01(data)
    expected = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(res, expected)


# ==============================================================================
# 2. Linear Feature Neutralization Mathematical Invariants (White-Box)
# ==============================================================================

def test_whitebox_neutralize_zero_proportion_identity():
    """White-box: neutralizing with proportion=0.0 returns identical predictions."""
    np.random.seed(42)
    df = pd.DataFrame({
        "pred": np.random.uniform(0, 1, 100),
        "feat_1": np.random.randn(100),
        "feat_2": np.random.randn(100),
    })
    res = neutralize(df, ["pred"], extra_neutralizers=["feat_1", "feat_2"], proportion=0.0)
    np.testing.assert_allclose(res["pred"].values, df["pred"].values)


def test_whitebox_neutralize_orthogonalization_property():
    """White-box: at proportion=1.0, prediction is completely orthogonalized to features."""
    np.random.seed(42)
    n_samples = 500
    feat_1 = np.random.randn(n_samples)
    # Artificial strong correlation between prediction and feat_1
    pred_raw = feat_1 * 2.0 + np.random.randn(n_samples) * 0.5

    df = pd.DataFrame({
        "pred": pred_raw,
        "feat_1": feat_1,
    })

    # Raw correlation is high
    corr_before = np.corrcoef(df["pred"], df["feat_1"])[0, 1]
    assert abs(corr_before) > 0.80

    # Neutralize with proportion=1.0
    res = neutralize(df, ["pred"], extra_neutralizers=["feat_1"], proportion=1.0)
    corr_after = np.corrcoef(res["pred"], res["feat_1"])[0, 1]

    # Neutralized correlation should drop significantly close to zero
    assert abs(corr_after) < 0.15


def test_whitebox_neutralize_collinear_features_stability():
    """White-box: perfectly collinear features do not cause LinAlgError or division by zero."""
    np.random.seed(42)
    n = 100
    feat = np.random.randn(n)
    df = pd.DataFrame({
        "pred": np.random.uniform(0, 1, n),
        "feat_1": feat,
        "feat_2": feat * 2.0,  # Collinear
        "feat_const": np.ones(n),  # Constant
    })
    res = neutralize(df, ["pred"], extra_neutralizers=["feat_1", "feat_2", "feat_const"], proportion=0.5)
    assert not res["pred"].isna().any()
    assert np.all(res["pred"].values >= 0.0)
    assert np.all(res["pred"].values <= 1.0)


def test_whitebox_neutralize_nan_features_resilience():
    """White-box: feature columns containing NaNs must not corrupt predictions into NaNs."""
    np.random.seed(42)
    n = 100
    feat_with_nans = np.random.randn(n)
    feat_with_nans[10:20] = np.nan
    feat_with_nans[50:60] = np.nan

    df = pd.DataFrame({
        "pred": np.random.uniform(0, 1, n),
        "feat_nan": feat_with_nans,
        "feat_clean": np.random.randn(n),
    })
    res = neutralize(df, ["pred"], extra_neutralizers=["feat_nan", "feat_clean"], proportion=0.5)
    assert not res["pred"].isna().any(), "Predictions must contain ZERO NaNs even when features contain NaNs"
    assert np.all(res["pred"].values >= 0.0)
    assert np.all(res["pred"].values <= 1.0)
