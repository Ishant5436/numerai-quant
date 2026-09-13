"""
Unit & Invariant Test Suite for Numerai Signals v3 'Supernova' Pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signals.alpha_factors import SupernovaAlphaGenerator, rank_01
from signals.signals_config import FACTOR_WEIGHTS
from signals.signals_pipeline import SupernovaSignalsPipeline
import signals.live_data_fetcher as live_data_fetcher
import signals.signals_pipeline as signals_pipeline
from signals.live_data_fetcher import DataFetchError, fetch_ohlcv_history


def test_signals_alpha_factor_extraction():
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    dates = pd.date_range("2025-01-01", periods=100)
    closes = pd.Series(np.linspace(100.0, 150.0, 100), index=dates)
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = pd.Series(np.full(100, 1000000.0), index=dates)

    factors = gen.compute_factors_for_series(closes, highs, lows, volumes)
    assert factors["momentum_12m"] > 0.0  # Positive trend
    assert factors["trend_slope"] > 0.0
    assert factors["volatility_inverse"] > 0.0


def test_signals_rank_01_uniform_bounds():
    raw_vals = np.array([10.5, 3.2, 99.1, -4.0, 12.0, 0.0])
    ranked = rank_01(raw_vals)

    assert len(ranked) == len(raw_vals)
    assert np.all(ranked >= 0.0)
    assert np.all(ranked <= 1.0)
    # Monotonic preservation
    assert ranked[2] == np.max(ranked)  # 99.1 is highest
    assert ranked[3] == np.min(ranked)  # -4.0 is lowest


def test_signals_pipeline_end_to_end(tmp_path):
    # use_live_data=False: this test is about the pipeline's math/plumbing,
    # not about network reachability, so it stays on the deterministic
    # synthetic generator. Live fetching + caching is covered separately
    # below in the live_data_fetcher tests.
    pipeline = SupernovaSignalsPipeline(
        tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"], use_live_data=False
    )
    out_file = str(tmp_path / "test_signals_submission.csv")

    df = pipeline.run_pipeline(output_filename=out_file)
    assert os.path.exists(out_file)
    assert len(df) == 5
    assert "numerai_ticker" in df.columns
    assert "signal" in df.columns
    assert not df["signal"].isna().any()
    assert np.all((df["signal"] >= 0.0) & (df["signal"] <= 1.0))


def _fake_bars(n: int = 252) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, n), index=dates)
    return pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full(n, 1_000_000.0),
    }, index=dates)


def test_fetch_ohlcv_history_live_success_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(live_data_fetcher, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live_data_fetcher, "_fetch_live", lambda ticker, bars: _fake_bars(bars))

    df = fetch_ohlcv_history("AAPL", bars=252)

    assert len(df) == 252
    assert list(df.columns) == ["close", "high", "low", "volume"]
    assert not df.isna().any().any()

    cache_file = tmp_path / "AAPL.parquet"
    assert cache_file.exists()
    cached = pd.read_parquet(cache_file)
    assert len(cached) == 252


def test_fetch_ohlcv_history_falls_back_to_cache_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(live_data_fetcher, "CACHE_DIR", str(tmp_path))

    # Pre-seed the cache as if a previous successful run wrote it.
    seed = _fake_bars(252)
    seed.to_parquet(tmp_path / "MSFT.parquet")

    def _always_fails(ticker, bars):
        raise DataFetchError("simulated offline / rate-limited")

    monkeypatch.setattr(live_data_fetcher, "_fetch_live", _always_fails)

    df = fetch_ohlcv_history("MSFT", bars=252)

    assert len(df) == 252
    assert list(df.columns) == ["close", "high", "low", "volume"]


def test_fetch_ohlcv_history_raises_cleanly_when_no_cache_and_live_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(live_data_fetcher, "CACHE_DIR", str(tmp_path))  # empty dir, no cache file

    def _always_fails(ticker, bars):
        raise DataFetchError("simulated network failure")

    monkeypatch.setattr(live_data_fetcher, "_fetch_live", _always_fails)

    with pytest.raises(DataFetchError):
        fetch_ohlcv_history("ZZZZ_NONEXISTENT", bars=252)


def test_fetch_ohlcv_history_input_validation_is_a_controlled_assertion():
    with pytest.raises(AssertionError):
        fetch_ohlcv_history("", bars=252)
    with pytest.raises(AssertionError):
        fetch_ohlcv_history("AAPL", bars=10)  # below MIN_USABLE_BARS


def test_pipeline_wired_to_live_fetcher(tmp_path, monkeypatch):
    """The actual 'wire it in' check: use_live_data=True routes run_pipeline
    through fetch_ohlcv_history, not the synthetic generator."""
    calls = []

    def _fake_fetch(ticker, bars=252):
        calls.append((ticker, bars))
        return _fake_bars(bars)

    monkeypatch.setattr(signals_pipeline, "fetch_ohlcv_history", _fake_fetch)

    tickers = ["AAPL", "MSFT", "NVDA"]
    pipeline = SupernovaSignalsPipeline(tickers=tickers, use_live_data=True)
    out_file = str(tmp_path / "live_submission.csv")

    df = pipeline.run_pipeline(output_filename=out_file)

    assert calls == [(t, 252) for t in tickers]  # real 252-day history requested per ticker
    assert len(df) == 3
    assert not df["signal"].isna().any()
    assert np.all((df["signal"] >= 0.0) & (df["signal"] <= 1.0))


def test_pipeline_skips_failing_tickers_but_survives(tmp_path, monkeypatch):
    def _flaky_fetch(ticker, bars=252):
        if ticker == "BADTICKER":
            raise DataFetchError("delisted")
        return _fake_bars(bars)

    monkeypatch.setattr(signals_pipeline, "fetch_ohlcv_history", _flaky_fetch)

    pipeline = SupernovaSignalsPipeline(tickers=["AAPL", "BADTICKER", "MSFT"], use_live_data=True)
    df = pipeline.run_pipeline(output_filename=str(tmp_path / "partial.csv"))

    assert len(df) == 2  # BADTICKER skipped, the other two still produced signals
    assert set(df["numerai_ticker"]) == {"AAPL", "MSFT"}


def test_pipeline_raises_when_every_ticker_fails(tmp_path, monkeypatch):
    def _always_fails(ticker, bars=252):
        raise DataFetchError("simulated total outage")

    monkeypatch.setattr(signals_pipeline, "fetch_ohlcv_history", _always_fails)

    pipeline = SupernovaSignalsPipeline(tickers=["AAPL", "MSFT"], use_live_data=True)
    with pytest.raises(DataFetchError):
        pipeline.run_pipeline(output_filename=str(tmp_path / "should_not_be_written.csv"))


def test_short_term_reversal_5d_directionality():
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    dates = pd.date_range("2025-01-01", periods=60)
    # Series A: Flat for 55 days, then pumps 20% in last 5 days
    prices_pump = np.full(60, 100.0)
    prices_pump[55:] = np.linspace(100.0, 120.0, 5)
    closes_pump = pd.Series(prices_pump, index=dates)

    # Series B: Flat for 55 days, then dumps 20% in last 5 days
    prices_dump = np.full(60, 100.0)
    prices_dump[55:] = np.linspace(100.0, 80.0, 5)
    closes_dump = pd.Series(prices_dump, index=dates)

    volumes = pd.Series(np.full(60, 1_000_000.0), index=dates)

    factors_pump = gen.compute_factors_for_series(closes_pump, closes_pump * 1.01, closes_pump * 0.99, volumes)
    factors_dump = gen.compute_factors_for_series(closes_dump, closes_dump * 1.01, closes_dump * 0.99, volumes)

    # Reversal is negative return: a pump should yield negative reversal score (sell signal),
    # while a dump should yield positive reversal score (buy signal)
    assert "short_term_reversal_5d" in factors_pump
    assert "short_term_reversal_5d" in factors_dump
    assert factors_pump["short_term_reversal_5d"] < 0.0
    assert factors_dump["short_term_reversal_5d"] > 0.0


def test_carhart_momentum_12_1m_isolates_recent_drop():
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    dates = pd.date_range("2024-01-01", periods=260)
    # Stock rallied from 50 to 120 over 239 days, then pulled back to 100 in the last 21 days
    prices = np.full(260, 100.0)
    prices[:239] = np.linspace(50.0, 120.0, 239)
    prices[239:] = np.linspace(120.0, 100.0, 21)
    closes = pd.Series(prices, index=dates)
    volumes = pd.Series(np.full(260, 1_000_000.0), index=dates)

    factors = gen.compute_factors_for_series(closes, closes * 1.01, closes * 0.99, volumes)
    assert "carhart_momentum_12_1m" in factors
    # Between t-252 and t-21, price went from ~50 to 120 -> Strong positive Carhart momentum
    assert factors["carhart_momentum_12_1m"] > 0.5


def test_downside_volatility_asymmetry_convexity():
    gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
    dates = pd.date_range("2025-01-01", periods=60)

    # Stock A: Mostly flat with positive upside spikes (positive skew / upside convexity)
    np.random.seed(42)
    daily_rets_convex = np.zeros(60)
    daily_rets_convex[10] = 0.08
    daily_rets_convex[25] = 0.07
    daily_rets_convex[45] = 0.09
    prices_convex = 100.0 * np.cumprod(1.0 + daily_rets_convex)

    # Stock B: Mostly flat with downside crash spikes (downside liquidation)
    daily_rets_panic = np.zeros(60)
    daily_rets_panic[10] = -0.08
    daily_rets_panic[25] = -0.07
    daily_rets_panic[45] = -0.09
    prices_panic = 100.0 * np.cumprod(1.0 + daily_rets_panic)

    closes_convex = pd.Series(prices_convex, index=dates)
    closes_panic = pd.Series(prices_panic, index=dates)
    volumes = pd.Series(np.full(60, 1_000_000.0), index=dates)

    factors_convex = gen.compute_factors_for_series(closes_convex, closes_convex * 1.01, closes_convex * 0.99, volumes)
    factors_panic = gen.compute_factors_for_series(closes_panic, closes_panic * 1.01, closes_panic * 0.99, volumes)

    assert "downside_volatility_asymmetry" in factors_convex
    assert "downside_volatility_asymmetry" in factors_panic
    assert factors_convex["downside_volatility_asymmetry"] > 0.0
    assert factors_panic["downside_volatility_asymmetry"] < 0.0

