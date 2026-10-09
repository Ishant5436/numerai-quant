"""
Unit and invariant tests for Gaussian Rank-Normal (Probit) Feature Neutralization.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from neutralize import gaussian_neutralize, rank_01


def test_gaussian_neutralize_uniform_bounds_and_no_nans():
    """Verify Gaussian neutralization outputs strictly uniform [0, 1] with 0 NaNs."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "pred": np.random.uniform(0, 1, n),
        "feat1": np.random.normal(0, 1, n),
        "feat2": np.random.normal(0, 1, n),
    })
    
    neut_df = gaussian_neutralize(df, columns=["pred"], extra_neutralizers=["feat1", "feat2"], proportion=0.50)
    assert isinstance(neut_df, pd.DataFrame)
    assert not neut_df["pred"].isna().any(), "Output cannot contain NaNs"
    assert neut_df["pred"].min() >= 0.0, "Values must be >= 0.0"
    assert neut_df["pred"].max() <= 1.0, "Values must be <= 1.0"


def test_gaussian_neutralize_reduces_factor_correlation():
    """Verify 100% Gaussian neutralization drives factor correlation to near zero."""
    np.random.seed(1337)
    n = 300
    risk_factor = np.random.normal(0, 1, n)
    # Intentionally correlated prediction
    raw_pred = 0.8 * risk_factor + 0.2 * np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "pred": rank_01(raw_pred),
        "risk_factor": risk_factor,
    })

    initial_corr, _ = spearmanr(df["pred"], df["risk_factor"])
    assert initial_corr > 0.60, "Pre-condition: Initial correlation must be strong"

    neut_df = gaussian_neutralize(df, columns=["pred"], extra_neutralizers=["risk_factor"], proportion=1.0)
    final_corr, _ = spearmanr(neut_df["pred"], df["risk_factor"])
    
    assert abs(final_corr) < 0.08, f"Final correlation should be near zero, got {final_corr}"
    assert abs(final_corr) < abs(initial_corr), "Correlation must be significantly reduced"


def test_gaussian_neutralize_zero_proportion_identity():
    """Verify proportion <= 0.0 leaves input dataframe unaltered."""
    df = pd.DataFrame({"pred": [0.2, 0.5, 0.8], "feat": [1.0, 2.0, 3.0]})
    res = gaussian_neutralize(df, columns=["pred"], extra_neutralizers=["feat"], proportion=0.0)
    assert np.allclose(res["pred"].values, df["pred"].values)
    assert len(res) == len(df)
