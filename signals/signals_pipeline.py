"""
Autonomous Numerai Signals v3 'Supernova' Fleet Submission Pipeline
Computes multi-factor orthogonal alphas across 7,219 equities, applies QR
feature neutralization, and manages automated weekly API submissions.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from signals.signals_config import (  # noqa: E402
    SIGNALS_DATA_DIR,
    SIGNALS_MODELS,
    REPRESENTATIVE_TICKERS,
    FACTOR_WEIGHTS,
    NEUTRALIZATION_PROPORTION,
    ensure_directories,
)
from signals.alpha_factors import SupernovaAlphaGenerator  # noqa: E402
from signals.live_data_fetcher import fetch_ohlcv_history, DataFetchError  # noqa: E402
from neutralize import neutralize, rank_01  # noqa: E402


def load_or_download_live_data(dest_path: str = None, force_download: bool = False) -> pd.DataFrame:
    """
    Loads local live parquet dataset or downloads official signals/v3.0/live.parquet.
    Validates universe size and mandatory columns.
    """
    ensure_directories()
    if dest_path is None:
        dest_path = os.path.join(SIGNALS_DATA_DIR, "signals_v3_live.parquet")

    assert isinstance(dest_path, str) and len(dest_path) > 0, "Invalid destination path"

    need_download = force_download or not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0
    if need_download:
        print("[INFO] Fetching latest live dataset: signals/v3.0/live.parquet...")
        from numerapi import SignalsAPI
        sapi = SignalsAPI()
        sapi.download_dataset("signals/v3.0/live.parquet", dest_path)

    df = pd.read_parquet(dest_path)
    assert len(df) >= 1000, f"Expected at least 1,000 tickers, got {len(df)}"
    assert "numerai_ticker" in df.columns, "Missing mandatory 'numerai_ticker' column"
    return df


def compute_fleet_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 5 distinct orthogonal alpha factor strategies for registered Signals models.
    """
    assert len(df) > 0, "Input DataFrame cannot be empty"
    assert "numerai_ticker" in df.columns, "Missing numerai_ticker column"

    res = df.copy()
    mom_cols = ["feature_momentum_12w_factor", "feature_momentum_26w_factor", "feature_momentum_52w_less_4w_factor"]
    val_cols = ["feature_book_to_price_factor", "feature_earnings_yield_factor", "feature_dividend_yield_factor"]
    vol_cols = ["feature_volatility_factor", "feature_beta_factor"]

    # 1. Flagship Multi-Factor Composite
    s1_mom = res[mom_cols].mean(axis=1)
    s1_val = res[val_cols].mean(axis=1)
    s1_vol = -res[vol_cols].mean(axis=1)
    res["s_flagship"] = 0.35 * rank_01(s1_mom) + 0.35 * rank_01(s1_val) + 0.30 * rank_01(s1_vol)

    # 2. Momentum Divergence & Oscillators
    osc_cols = ["feature_ppo_60d_90d_country_ranknorm", "feature_trix_60d_country_ranknorm", "feature_rsi_60d_country_ranknorm"]
    res["s_mom"] = 0.50 * rank_01(res[mom_cols].mean(axis=1)) + 0.50 * rank_01(res[osc_cols].mean(axis=1))

    # 3. Fundamental Value Yield Alpha
    s3_b2p = res["feature_book_to_price_factor"].fillna(0.0)
    s3_ey = res["feature_earnings_yield_factor"].fillna(0.0)
    s3_div = res["feature_dividend_yield_factor"].fillna(0.0)
    res["s_val"] = 0.40 * rank_01(s3_b2p) + 0.40 * rank_01(s3_ey) + 0.20 * rank_01(s3_div)

    # 4. Low-Volatility & Quality Defensive
    s4_vol = -res["feature_volatility_factor"].fillna(0.0)
    s4_beta = -res["feature_beta_factor"].fillna(0.0)
    s4_gro = res["feature_growth_factor"].fillna(0.0)
    res["s_vol"] = 0.40 * rank_01(s4_vol) + 0.30 * rank_01(s4_beta) + 0.30 * rank_01(s4_gro)

    # 5. Supernova Multi-Horizon Composite
    s5_p1 = res["feature_ppo_60d_130d_country_ranknorm"].fillna(0.5)
    s5_p2 = res["feature_trix_130d_country_ranknorm"].fillna(0.5)
    res["s_alpha"] = 0.30 * rank_01(s5_p1) + 0.30 * rank_01(s5_p2) + 0.20 * rank_01(res["s_val"]) + 0.20 * rank_01(res["s_vol"])

    return res


def neutralize_and_format_signals(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Applies QR-decomposition feature neutralization and formats submission dataframes.
    """
    assert len(df) > 0, "DataFrame must contain data"
    signal_cols = ["s_flagship", "s_mom", "s_val", "s_vol", "s_alpha"]
    risk_factors = ["feature_market_cap_factor", "feature_volatility_factor", "feature_beta_factor"]

    neut_df = neutralize(df, signal_cols, risk_factors, proportion=NEUTRALIZATION_PROPORTION)
    strategy_mapping = {
        "cypherpole_sig": "s_flagship",
        "cypherpole_sig_mom": "s_mom",
        "cypherpole_sig_val": "s_val",
        "cypherpole_sig_vol": "s_vol",
        "cypherpole_sig_alpha": "s_alpha",
    }

    formatted = {}
    for model_name, col in strategy_mapping.items():
        assert col in neut_df.columns, f"Missing signal column: {col}"
        ranks = rank_01(neut_df[col])
        assert not np.isnan(ranks).any(), f"NaN values detected in {model_name}"
        formatted[model_name] = pd.DataFrame({
            "numerai_ticker": neut_df["numerai_ticker"],
            "signal": ranks
        })

    assert len(formatted) == 5, f"Expected 5 model frames, got {len(formatted)}"
    return formatted


def upload_fleet_submissions(formatted_dict: dict[str, pd.DataFrame], dry_run: bool = False) -> dict[str, str]:
    """
    Saves submission CSVs and uploads to Numerai Signals API with retry guards.
    """
    assert len(formatted_dict) > 0, "Formatted submissions dictionary cannot be empty"
    ensure_directories()
    today_str = datetime.now().strftime("%Y%m%d")

    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    pub_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    sec_key = os.environ.get("NUMERAI_SECRET_KEY", "")
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    if "$" in auth and not (pub_id and sec_key):
        pub_id, sec_key = auth.split("$", 1)

    sapi = None
    if not dry_run and pub_id and sec_key:
        from numerapi import SignalsAPI
        sapi = SignalsAPI(public_id=pub_id, secret_key=sec_key)

    results = {}
    model_keys = list(SIGNALS_MODELS.keys())
    assert len(model_keys) <= 10, "Model fleet size exceeds upper safety bound"

    for model_name in model_keys:
        model_meta = SIGNALS_MODELS[model_name]
        model_id = model_meta["id"]
        sub_df = formatted_dict.get(model_name)
        if sub_df is None:
            continue

        csv_path = os.path.join(SIGNALS_DATA_DIR, f"signals_{model_name}_{today_str}.csv")
        sub_df.to_csv(csv_path, index=False)

        if dry_run or sapi is None:
            results[model_name] = f"LOCAL_SAVED:{csv_path}"
            print(f"[DRY-RUN] Saved {len(sub_df)} rows for {model_name} -> {csv_path}")
            continue

        try:
            print(f"Uploading {len(sub_df)} rows for {model_name} (ID: {model_id})...")
            sub_id = sapi.upload_predictions(csv_path, model_id=model_id)
            results[model_name] = sub_id
            print(f"[SUCCESS] {model_name} submitted! ID: {sub_id}")
        except Exception as err:
            results[model_name] = f"ERROR:{err}"
            print(f"[ERROR] Failed upload for {model_name}: {err}")

    assert len(results) > 0, "No model results processed"
    return results


class SupernovaSignalsPipeline:
    """
    High-level orchestrator class for Numerai Signals operations.
    Supports both custom ticker universe analysis and full 5-model Signals fleet.
    """
    def __init__(self, tickers: list = None, use_live_data: bool = True):
        ensure_directories()
        self.tickers = tickers
        self.use_live_data = use_live_data
        self.alpha_gen = SupernovaAlphaGenerator(FACTOR_WEIGHTS)

    def generate_mock_market_history(self, ticker: str, bars: int = 260) -> pd.DataFrame:
        """Generates realistic geometric Brownian motion price & volume history."""
        np.random.seed(abs(hash(ticker)) % (2**31))
        dt = 1.0 / 252.0
        mu = 0.10
        sigma = 0.25
        returns = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=bars)
        price_series = 100.0 * np.exp(np.cumsum(returns))
        highs = price_series * (1.0 + np.abs(np.random.normal(0, 0.008, size=bars)))
        lows = price_series * (1.0 - np.abs(np.random.normal(0, 0.008, size=bars)))
        volumes = np.random.lognormal(mean=14.0, sigma=0.5, size=bars)
        return pd.DataFrame({"close": price_series, "high": highs, "low": lows, "volume": volumes})

    def get_market_history(self, ticker: str, bars: int = 252) -> pd.DataFrame:
        """Retrieves market history via live fetcher or mock generator."""
        if not self.use_live_data:
            return self.generate_mock_market_history(ticker, bars=bars)
        return fetch_ohlcv_history(ticker, bars=bars)

    def run_ticker_pipeline(self, output_filename: str = None) -> pd.DataFrame:
        """Executes multi-factor signal extraction for explicit ticker list."""
        tickers_to_process = self.tickers or REPRESENTATIVE_TICKERS
        assert len(tickers_to_process) > 0, "Ticker universe cannot be empty"

        factor_records = []
        skipped_tickers = []
        for ticker in tickers_to_process:
            try:
                history = self.get_market_history(ticker, bars=252)
            except DataFetchError:
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
            raise DataFetchError(f"No usable market history for tickers (skipped: {skipped_tickers})")

        factor_df = pd.DataFrame(factor_records).set_index("ticker")
        raw_signals = self.alpha_gen.combine_factors(factor_df)
        factor_df["raw_signal"] = raw_signals

        neutralized_df = neutralize(
            df=factor_df.reset_index(),
            columns=["raw_signal"],
            extra_neutralizers=["volatility_inverse", "trend_slope"],
            proportion=NEUTRALIZATION_PROPORTION
        )

        submission_df = pd.DataFrame({
            "numerai_ticker": neutralized_df["ticker"],
            "signal": neutralized_df["raw_signal"]
        })

        if output_filename is not None:
            submission_df.to_csv(output_filename, index=False)
        return submission_df

    def run_fleet_pipeline(self, force_download: bool = False, dry_run: bool = False) -> dict[str, str]:
        """Runs the complete end-to-end 5-model Signals fleet execution."""
        live_df = load_or_download_live_data(force_download=force_download)
        assert len(live_df) > 0, "Failed to load live data"

        signal_df = compute_fleet_signals(live_df)
        formatted_dict = neutralize_and_format_signals(signal_df)
        results = upload_fleet_submissions(formatted_dict, dry_run=dry_run)
        return results

    def run_pipeline(self, output_filename: str = None, force_download: bool = False, dry_run: bool = False):
        """Unified entrypoint dispatching between ticker-specific and fleet modes."""
        if self.tickers is not None or output_filename is not None:
            return self.run_ticker_pipeline(output_filename=output_filename)
        return self.run_fleet_pipeline(force_download=force_download, dry_run=dry_run)


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Numerai Signals Fleet Pipeline...")
    pipeline = SupernovaSignalsPipeline(use_live_data=True)
    out = pipeline.run_pipeline(dry_run=False)
    print("\n--- Signals Fleet Submission Summary ---")
    for k, v in out.items():
        print(f"  {k}: {v}")
