"""
Numerai Signals v3 'Supernova' Engine Configuration
Target Horizon: 60-Day Forward Equity Returns (Supernova Standard)
"""

import os

SIGNALS_DATA_DIR = "/Users/ishantpanchal/numerai-quant/signals/data"
SIGNALS_MODEL_DIR = "/Users/ishantpanchal/numerai-quant/signals/models"
os.makedirs(SIGNALS_DATA_DIR, exist_ok=True)
os.makedirs(SIGNALS_MODEL_DIR, exist_ok=True)

# Universe of representative global liquid equities for multi-factor alpha computation
REPRESENTATIVE_TICKERS = [
    # US Mega-Cap Tech & Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "QCOM",
    # US Financials & Industrials
    "JPM", "V", "MA", "BAC", "CAT", "GE", "UNH", "LLY", "JNJ", "PG",
    # Global Semi & Hardware
    "ASML", "TSM", "ARM", "INTC", "TXN",
    # Energy & Commodities
    "XOM", "CVX", "COP", "SLB", "EOG"
]

# Factor Weights for Multi-Factor Supernova Alpha
FACTOR_WEIGHTS = {
    "momentum_12m": 0.30,      # 12-month residual price momentum
    "momentum_1m": -0.10,      # 1-month short-term reversal (mean reversion)
    "volatility_inverse": 0.25,# Low-volatility anomaly (Parkinson-derived)
    "trend_slope": 0.20,       # EMA 50 / 200 Golden Cross slope
    "volume_shock": 0.15       # Institutional abnormal volume acceleration
}

NEUTRALIZATION_PROPORTION = 0.35  # Project out 35% of broad market beta
