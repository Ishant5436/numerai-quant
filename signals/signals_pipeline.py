"""
Autonomous Numerai Signals v3 'Supernova' Submission Pipeline
Computes multi-factor cross-sectional alpha, performs QR feature neutralization,
and formats official tournament submission files.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, "/Users/ishantpanchal/numerai-quant")
from signals.signals_config import (
    SIGNALS_DATA_DIR,
    REPRESENTATIVE_TICKERS,
    FACTOR_WEIGHTS,
    NEUTRALIZATION_PROPORTION
)
from signals.alpha_factors import SupernovaAlphaGenerator, rank_01
from signals.live_data_fetcher import fetch_ohlcv_history, DataFetchError
from neutralize import neutralize


class SupernovaSignalsPipeline:
    def __init__(self, tickers: list = None, use_live_data: bool = True):
        self.tickers = tickers or REPRESENTATIVE_TICKERS
        self.alpha_gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)
        self.use_live_data = use_live_data

    def generate_mock_market_history(self, ticker: str, bars: int = 260) -> pd.DataFrame:
        """
        Generates realistic geometric Brownian motion price & volume history for testing & benchmarking.
        """
        np.random.seed(abs(hash(ticker)) % (2**31))
        dt = 1.0 / 252.0
        mu = 0.10
        sigma = 0.25

        returns = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=bars)
        price_series = 100.0 * np.exp(np.cumsum(returns))

        highs = price_series * (1.0 + np.abs(np.random.normal(0, 0.008, size=bars)))
        lows = price_series * (1.0 - np.abs(np.random.normal(0, 0.008, size=bars)))
        volumes = np.random.lognormal(mean=14.0, sigma=0.5, size=bars)

        return pd.DataFrame({
            "close": price_series,
            "high": highs,
            "low": lows,
            "volume": volumes
        })

    def get_market_history(self, ticker: str, bars: int = 252) -> pd.DataFrame:
        """
        Real 252-day OHLCV history when self.use_live_data is True (live
        yfinance fetch, falling back to cached bars if offline -- see
        signals.live_data_fetcher). Synthetic GBM history otherwise, for
        deterministic offline testing/benchmarking.
        """
        if not self.use_live_data:
            return self.generate_mock_market_history(ticker, bars=bars)
        return fetch_ohlcv_history(ticker, bars=bars)

    def run_pipeline(self, output_filename: str = None) -> pd.DataFrame:
        """
        Executes end-to-end multi-factor signal extraction, rank combination,
        and QR factor neutralization.
        """
        assert len(self.tickers) > 0, "Ticker universe cannot be empty"

        factor_records = []
        skipped_tickers = []
        for ticker in self.tickers:
            try:
                history = self.get_market_history(ticker, bars=252)
            except DataFetchError as e:
                # One bad ticker (delisted, rate-limited, no cache yet)
                # should not abort the whole universe's run.
                skipped_tickers.append(ticker)
                continue
            factors = self.alpha_gen.compute_factors_for_series(
                closes=history["close"],
                highs=history["high"],
                lows=history["low"],
                volumes=history["volume"]
            )
            factors["ticker"] = ticker
            factor_records.append(factors)

        if not factor_records:
            raise DataFetchError(
                f"No usable market history for any of {len(self.tickers)} tickers "
                f"(skipped: {skipped_tickers})"
            )

        factor_df = pd.DataFrame(factor_records).set_index("ticker")

        # 1. Combine factors into raw composite alpha
        raw_signals = self.alpha_gen.combine_factors(factor_df)
        factor_df["raw_signal"] = raw_signals

        # 2. QR-Decomposition Linear Neutralization against market volatility and trend
        neutralized_df = neutralize(
            df=factor_df.reset_index(),
            columns=["raw_signal"],
            extra_neutralizers=["volatility_inverse", "trend_slope"],
            proportion=NEUTRALIZATION_PROPORTION
        )

        # 3. Format final submission dataframe
        submission_df = pd.DataFrame({
            "numerai_ticker": neutralized_df["ticker"],
            "signal": neutralized_df["raw_signal"]
        })

        if output_filename is None:
            today_str = datetime.now().strftime("%Y%m%d")
            output_filename = os.path.join(SIGNALS_DATA_DIR, f"signals_supernova_submission_{today_str}.csv")

        submission_df.to_csv(output_filename, index=False)

        # 4. Optional automated upload if Signals models are registered on connected account
        try:
            from numerapi import SignalsAPI
            auth = os.environ.get("NUMERAI_MCP_AUTH", "")
            public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
            secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")
            if "$" in auth and not (public_id and secret_key):
                public_id, secret_key = auth.split("$", 1)

            if public_id and secret_key:
                sapi = SignalsAPI(public_id=public_id, secret_key=secret_key)
                signals_models = sapi.get_models()
                if signals_models:
                    for s_name, s_id in signals_models.items():
                        print(f"Uploading Signals predictions to model '{s_name}' (ID: {s_id})...")
                        sub_id = sapi.upload_predictions(output_filename, model_id=s_id)
                        print(f"[SUCCESS] Signals model '{s_name}' submitted! Submission ID: {sub_id}")
                else:
                    print("[INFO] Zero Numerai Signals models registered on account. Signals saved locally for offline evaluation.")
        except Exception as upload_err:
            print(f"[NOTE] Signals automated upload skipped: {upload_err}")

        return submission_df


if __name__ == "__main__":
    pipeline = SupernovaSignalsPipeline()
    df = pipeline.run_pipeline()
    print("[SUCCESS] Numerai Signals Supernova pipeline completed!")
    print(f"Generated {len(df)} asset signals:")
    print(df.head(10))
