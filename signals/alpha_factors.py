"""
Multi-Factor Alpha Generators for Numerai Signals v3 'Supernova'
Computes cross-sectional alpha factors: Momentum, Short-Term Reversal,
Low-Volatility Anomaly, Trend Convergence, and Volume Shocks.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array strictly to uniform distribution [0.0, 1.0] with NaN safety."""
    arr = np.asarray(arr, dtype=np.float64)
    result = np.full_like(arr, 0.5)
    valid = ~np.isnan(arr)
    n_valid = valid.sum()
    if n_valid > 0:
        ranks = rankdata(arr[valid], method="average")
        result[valid] = (ranks - 0.5) / n_valid
    return result


class SupernovaAlphaGenerator:
    """
    Computes orthogonal multi-factor alpha scores for an equity price/volume history.
    """
    def __init__(self, factor_weights: dict):
        self.weights = factor_weights

    def compute_factors_for_series(self, closes: pd.Series, highs: pd.Series, lows: pd.Series, volumes: pd.Series) -> dict:
        assert len(closes) >= 50, "Need at least 50 bars of history"
        assert len(highs) == len(closes) and len(lows) == len(closes)

        # 1. 12-Month Momentum (or max available)
        lookback_12m = min(len(closes) - 1, 252)
        mom_12m = (closes.iloc[-1] - closes.iloc[-lookback_12m]) / max(closes.iloc[-lookback_12m], 1e-6)

        # 2. 1-Month Reversal (Mean reversion over 21 days)
        lookback_1m = min(len(closes) - 1, 21)
        mom_1m = (closes.iloc[-1] - closes.iloc[-lookback_1m]) / max(closes.iloc[-lookback_1m], 1e-6)

        # 3. Parkinson Realized Volatility Anomaly (Inverse Volatility)
        # Low volatility stocks consistently exhibit higher risk-adjusted forward returns
        log_hl = np.log(highs.tail(30) / np.maximum(lows.tail(30), 1e-6))
        parkinson_var = (1.0 / (4.0 * np.log(2.0))) * np.mean(log_hl ** 2)
        parkinson_vol = np.sqrt(max(parkinson_var, 1e-9))
        vol_inverse = 1.0 / (parkinson_vol + 1e-4)

        # 4. Trend Slope (Fast EMA 20 vs Slow EMA 50)
        ema_20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        trend_slope = (ema_20 - ema_50) / max(closes.iloc[-1], 1e-6)

        # 5. Volume Acceleration Shock
        vol_20ma = volumes.tail(20).mean()
        vol_shock = (volumes.iloc[-1] - vol_20ma) / max(vol_20ma, 1.0)

        return {
            "momentum_12m": float(mom_12m),
            "momentum_1m": float(mom_1m),
            "volatility_inverse": float(vol_inverse),
            "trend_slope": float(trend_slope),
            "volume_shock": float(vol_shock)
        }

    def combine_factors(self, factor_df: pd.DataFrame) -> pd.Series:
        """
        Cross-sectionally rank-normalizes individual factors and blends via optimal weights.
        """
        assert not factor_df.empty, "Factor DataFrame cannot be empty"

        composite_raw = pd.Series(0.0, index=factor_df.index)

        for factor_name, weight in self.weights.items():
            if factor_name in factor_df.columns:
                ranked = rank_01(factor_df[factor_name].values) - 0.5
                composite_raw += weight * ranked

        # Final uniform [0.0, 1.0] transformation
        return pd.Series(rank_01(composite_raw.values), index=factor_df.index)
