"""
Comprehensive Integration Test Suite for Numerai Signals v3 'Supernova'
Tests:
1. End-to-end multi-ticker pipeline with 8 alpha factors
2. Cross-factor orthogonality and non-collinearity verification
3. Full 5-model fleet execution with real parquet dataset in dry-run mode
4. Pipeline robustness under extreme price shocks and liquidity stress
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from signals.alpha_factors import SupernovaAlphaGenerator  # noqa: E402
from signals.signals_config import FACTOR_WEIGHTS, REPRESENTATIVE_TICKERS, SIGNALS_MODELS  # noqa: E402
from signals.signals_pipeline import (  # noqa: E402
    SupernovaSignalsPipeline,
    compute_fleet_signals,
    neutralize_and_format_signals,
    upload_fleet_submissions,
)


def test_signals_integration_8_factors_intermediate_integrity():
    """
    Integration Test 1: Verify all 8 alpha factors are extracted and populated
    without NaNs or infinite values across a multi-asset universe.
    """
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    assert len(FACTOR_WEIGHTS) == 8, f"Expected 8 factors, got {len(FACTOR_WEIGHTS)}"

    expected_factors = {
        "carhart_momentum_12_1m",
        "momentum_12m",
        "short_term_reversal_5d",
        "momentum_1m",
        "volatility_inverse",
        "downside_volatility_asymmetry",
        "trend_slope",
        "volume_shock",
    }
    assert set(FACTOR_WEIGHTS.keys()) == expected_factors

    records = []
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "XOM", "TSLA", "LLY", "TSM"]

    for i, ticker in enumerate(tickers):
        np.random.seed(100 + i)
        n_bars = 260
        # Realistic drift + diffusion
        rets = np.random.normal(0.0005, 0.015, size=n_bars)
        prices = 100.0 * np.exp(np.cumsum(rets))
        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.006, size=n_bars)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.006, size=n_bars)))
        volumes = np.random.lognormal(14.0, 0.4, size=n_bars)

        closes_s = pd.Series(prices)
        highs_s = pd.Series(highs)
        lows_s = pd.Series(lows)
        volumes_s = pd.Series(volumes)

        factors = gen.compute_factors_for_series(closes_s, highs_s, lows_s, volumes_s)
        factors["ticker"] = ticker
        records.append(factors)

    df_factors = pd.DataFrame(records).set_index("ticker")
    assert df_factors.shape == (10, 8), f"Expected 10x8 factor matrix, got {df_factors.shape}"
    assert not df_factors.isna().any().any(), "Found unexpected NaN values in factor matrix"

    combined = gen.combine_factors(df_factors)
    assert len(combined) == 10
    assert not combined.isna().any()
    assert np.all((combined >= 0.0) & (combined <= 1.0))
    # Normalized ranks should have mean 0.50
    np.testing.assert_almost_equal(combined.mean(), 0.50, decimal=2)


def test_signals_integration_factor_orthogonality():
    """
    Integration Test 2: Verify pairwise correlations between new factors and legacy factors.
    New factors must be orthogonal (not perfectly collinear, |corr| < 0.85).
    """
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    records = []
    n_assets = 35
    n_bars = 260

    for i in range(n_assets):
        np.random.seed(200 + i)
        # Mixture of market regimes: trenders, mean-reverters, high-volatility
        trend_drift = np.random.uniform(-0.001, 0.002)
        vol = np.random.uniform(0.01, 0.035)
        rets = np.random.normal(trend_drift, vol, size=n_bars)
        prices = 50.0 * np.exp(np.cumsum(rets))
        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, size=n_bars)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, size=n_bars)))
        volumes = np.random.lognormal(13.0, 0.5, size=n_bars)

        factors = gen.compute_factors_for_series(pd.Series(prices), pd.Series(highs), pd.Series(lows), pd.Series(volumes))
        records.append(factors)

    df_factors = pd.DataFrame(records)
    corr_matrix = df_factors.corr()

    # 1. Carhart 12-1m vs 1-Month Reversal should not be redundant
    carhart_vs_1m = abs(corr_matrix.loc["carhart_momentum_12_1m", "momentum_1m"])
    assert carhart_vs_1m < 0.85, f"Carhart and 1m momentum too correlated: {carhart_vs_1m:.3f}"

    # 2. Downside Volatility Asymmetry vs Inverse Volatility should provide unique variance
    asym_vs_invol = abs(corr_matrix.loc["downside_volatility_asymmetry", "volatility_inverse"])
    assert asym_vs_invol < 0.85, f"Asymmetry and inverse vol too correlated: {asym_vs_invol:.3f}"

    # 3. 5-day reversal vs 12m momentum should be largely independent
    rev5d_vs_12m = abs(corr_matrix.loc["short_term_reversal_5d", "momentum_12m"])
    assert rev5d_vs_12m < 0.85, f"5d reversal and 12m momentum too correlated: {rev5d_vs_12m:.3f}"


def test_signals_integration_e2e_ticker_pipeline(tmp_path):
    """
    Integration Test 3: End-to-end execution of SupernovaSignalsPipeline
    with the 8-factor engine generating an output submission CSV.
    """
    test_tickers = REPRESENTATIVE_TICKERS[:8]
    pipeline = SupernovaSignalsPipeline(tickers=test_tickers, use_live_data=False)

    out_csv = str(tmp_path / "integration_submission.csv")
    sub_df = pipeline.run_pipeline(output_filename=out_csv)
    assert len(sub_df) == len(test_tickers)

    # 1. Output file verification
    assert os.path.exists(out_csv), "Submission CSV was not created"
    assert os.path.getsize(out_csv) > 0, "Submission CSV is empty"

    # 2. Schema compliance verification
    loaded = pd.read_csv(out_csv)
    assert list(loaded.columns) == ["numerai_ticker", "signal"]
    assert len(loaded) == len(test_tickers)
    assert not loaded["signal"].isna().any(), "Signal column contains NaNs"
    assert np.all(loaded["signal"] >= 0.0) and np.all(loaded["signal"] <= 1.0)

    # 3. Strict rank ordering (no identical flat signals)
    assert len(loaded["signal"].unique()) == len(test_tickers), "Signals must be strictly ranked"


def test_signals_integration_fleet_dry_run(tmp_path, monkeypatch):
    """
    Integration Test 4: End-to-end 5-model fleet pipeline execution using real
    live.parquet data, checking neutralization, formatting, and file export.
    """
    parquet_path = os.path.join(BASE_DIR, "signals", "data", "signals_v3_live.parquet")
    if not os.path.exists(parquet_path):
        pytest.skip("signals_v3_live.parquet not found locally; skipping fleet test")

    live_df = pd.read_parquet(parquet_path)
    assert len(live_df) >= 5000, f"Expected full universe (>=5000), got {len(live_df)}"

    # 1. Factor calculation for 5 models
    signal_df = compute_fleet_signals(live_df)
    assert "s_flagship" in signal_df.columns
    assert "s_mom" in signal_df.columns
    assert "s_val" in signal_df.columns
    assert "s_vol" in signal_df.columns
    assert "s_alpha" in signal_df.columns

    # 2. QR Neutralization & Rank01 formatting
    formatted = neutralize_and_format_signals(signal_df)
    assert set(formatted.keys()) == set(SIGNALS_MODELS.keys())

    for model_name, m_df in formatted.items():
        assert len(m_df) == len(live_df)
        assert list(m_df.columns) == ["numerai_ticker", "signal"]
        assert not m_df["signal"].isna().any()
        assert m_df["signal"].min() >= 0.0
        assert m_df["signal"].max() <= 1.0
        np.testing.assert_almost_equal(m_df["signal"].mean(), 0.50, decimal=2)

    # 3. Strategy diversity: pairwise correlation across 5 models must be distinct (< 0.95)
    model_preds = pd.DataFrame({m: formatted[m]["signal"].values for m in formatted})
    corr = model_preds.corr()
    for col1 in corr.columns:
        for col2 in corr.columns:
            if col1 != col2:
                assert corr.loc[col1, col2] < 0.95, f"Models {col1} and {col2} are too correlated: {corr.loc[col1, col2]}"

    # 4. Dry-run export to temporary directory
    monkeypatch.setattr("signals.signals_pipeline.SIGNALS_DATA_DIR", str(tmp_path))
    results = upload_fleet_submissions(formatted, dry_run=True)
    assert len(results) == 5
    for m, status in results.items():
        assert status.startswith("LOCAL_SAVED:")
        saved_file = status.split("LOCAL_SAVED:")[1]
        assert os.path.exists(saved_file)


def test_signals_integration_extreme_stress_handling():
    """
    Integration Test 5: Verify resilience of 8-factor math under extreme market shocks:
    - Zero volume
    - 99% liquidation crash
    - 10x sudden acquisition spike
    """
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    dates = pd.date_range("2025-01-01", periods=100)

    # Case A: Complete liquidation drop (from 100 to 0.01)
    crash_prices = np.linspace(100.0, 0.01, 100)
    factors_crash = gen.compute_factors_for_series(
        closes=pd.Series(crash_prices, index=dates),
        highs=pd.Series(crash_prices * 1.05, index=dates),
        lows=pd.Series(crash_prices * 0.95, index=dates),
        volumes=pd.Series(np.full(100, 1000.0), index=dates),
    )
    for k, v in factors_crash.items():
        assert np.isfinite(v), f"Crash case factor {k} is not finite: {v}"

    # Case B: 10x buyout spike (from 10 to 100 on day 99)
    buyout_prices = np.full(100, 10.0)
    buyout_prices[-1] = 100.0
    factors_buyout = gen.compute_factors_for_series(
        closes=pd.Series(buyout_prices, index=dates),
        highs=pd.Series(buyout_prices * 1.01, index=dates),
        lows=pd.Series(buyout_prices * 0.99, index=dates),
        volumes=pd.Series(np.full(100, 5000.0), index=dates),
    )
    for k, v in factors_buyout.items():
        assert np.isfinite(v), f"Buyout case factor {k} is not finite: {v}"

    # Case C: Zero trading volume across entire history
    zero_vol = pd.Series(np.zeros(100), index=dates)
    normal_prices = pd.Series(np.linspace(50.0, 60.0, 100), index=dates)
    factors_zero_vol = gen.compute_factors_for_series(
        closes=normal_prices,
        highs=normal_prices * 1.01,
        lows=normal_prices * 0.99,
        volumes=zero_vol,
    )
    for k, v in factors_zero_vol.items():
        assert np.isfinite(v), f"Zero volume factor {k} is not finite: {v}"
    assert factors_zero_vol["volume_shock"] == 0.0
