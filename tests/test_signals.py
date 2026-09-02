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
    pipeline = SupernovaSignalsPipeline(tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    out_file = str(tmp_path / "test_signals_submission.csv")

    df = pipeline.run_pipeline(output_filename=out_file)
    assert os.path.exists(out_file)
    assert len(df) == 5
    assert "numerai_ticker" in df.columns
    assert "signal" in df.columns
    assert not df["signal"].isna().any()
    assert np.all((df["signal"] >= 0.0) & (df["signal"] <= 1.0))
