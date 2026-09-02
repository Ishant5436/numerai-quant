"""
Live historical OHLCV fetcher for the Numerai Signals Supernova pipeline.

Fetches real daily bars via yfinance and caches them as parquet files under
signals/data/cache/. The cache serves two purposes: it avoids re-fetching
on every run, and it is the fallback data source when the live fetch fails
(offline, rate-limited, ticker delisted, etc.) -- fallback goes to the last
good cached bars, never to synthetic data.
"""

import logging
import os

import pandas as pd

from signals.signals_config import SIGNALS_DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(SIGNALS_DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["close", "high", "low", "volume"]
MIN_USABLE_BARS = 50  # matches SupernovaAlphaGenerator.compute_factors_for_series


class DataFetchError(RuntimeError):
    """Raised when live data cannot be fetched and no usable cache exists."""


def _cache_path(ticker: str) -> str:
    assert isinstance(ticker, str) and ticker, "ticker must be a non-empty string"
    safe = ticker.strip().upper().replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("Cache file for %s is unreadable (%s); ignoring it.", ticker, e)
        return None
    if df.empty or not set(REQUIRED_COLUMNS).issubset(df.columns):
        return None
    return df


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(ticker))
    except Exception as e:
        # Cache write failures must never break a live fetch that already
        # succeeded -- log and move on.
        logger.warning("Failed to write cache for %s: %s", ticker, e)


def _fetch_live(ticker: str, bars: int) -> pd.DataFrame:
    """Raises on any failure; never returns a partial/invalid frame."""
    import yfinance as yf

    # Generous calendar window so `bars` trading days survive weekends and
    # holidays: 1.6x plus a fixed floor comfortably covers 252 trading days.
    period_days = max(int(bars * 1.6), bars + 30)
    raw = yf.Ticker(ticker).history(period=f"{period_days}d", interval="1d", auto_adjust=True)

    if raw is None or raw.empty:
        raise DataFetchError(f"yfinance returned no data for {ticker!r}")

    raw = raw.rename(columns={"Close": "close", "High": "high", "Low": "low", "Volume": "volume"})
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataFetchError(f"yfinance response for {ticker!r} missing columns: {missing}")

    df = raw[REQUIRED_COLUMNS].dropna()
    if len(df) < MIN_USABLE_BARS:
        raise DataFetchError(
            f"yfinance returned only {len(df)} usable bars for {ticker!r} (need >= {MIN_USABLE_BARS})"
        )

    return df.tail(bars)


def fetch_ohlcv_history(ticker: str, bars: int = 252) -> pd.DataFrame:
    """
    Return the most recent `bars` daily OHLCV rows for `ticker`, oldest-first,
    with columns ["close", "high", "low", "volume"].

    Tries a live yfinance fetch first and refreshes the on-disk parquet cache
    on success. If the live fetch fails for any reason, falls back to the
    existing cache when one is present (logging that the data may be stale).
    Raises DataFetchError only when neither a live fetch nor a usable cache
    is available -- callers should not silently receive fabricated data.
    """
    assert isinstance(ticker, str) and ticker, "ticker must be a non-empty string"
    assert bars >= MIN_USABLE_BARS, f"bars must be >= {MIN_USABLE_BARS}"

    try:
        df = _fetch_live(ticker, bars)
        _save_cache(ticker, df)
        return df
    except Exception as live_error:
        cached = _load_cache(ticker)
        if cached is not None and len(cached) >= MIN_USABLE_BARS:
            logger.warning(
                "Live fetch failed for %s (%s); falling back to cached bars (%d rows).",
                ticker, live_error, len(cached),
            )
            return cached.tail(bars)
        raise DataFetchError(
            f"Live fetch failed for {ticker!r} and no usable cache exists: {live_error}"
        ) from live_error
