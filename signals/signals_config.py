"""
Numerai Signals v3 'Supernova' Engine Configuration
Target Horizon: 60-Day Forward Equity Returns (Supernova Standard)
"""

import os

SIGNALS_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_DATA_DIR = os.path.join(SIGNALS_DIR, "data")
SIGNALS_MODEL_DIR = os.path.join(SIGNALS_DIR, "models")


def ensure_directories() -> None:
    """Ensure data and model directories exist without top-level import side effects."""
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
    "carhart_momentum_12_1m": 0.25,        # 12-month trend excluding recent 1-month reversal
    "momentum_12m": 0.10,                  # 12-month residual price momentum
    "short_term_reversal_5d": 0.15,        # 5-day weekly liquidity reversal
    "momentum_1m": -0.05,                  # 1-month short-term mean reversion
    "volatility_inverse": 0.15,            # Low-volatility anomaly (Parkinson-derived)
    "downside_volatility_asymmetry": 0.15, # Positive return skew / upside convexity
    "trend_slope": 0.10,                   # EMA 20 / 50 trend slope
    "volume_shock": 0.05                   # Institutional abnormal volume acceleration
}

# Registered Numerai Signals Models
SIGNALS_MODELS = {
    "cypherpole_sig": {
        "id": "15c9929e-d4d7-46ff-b677-d906f292f465",
        "description": "Flagship Multi-Factor Composite",
        "strategy": "flagship"
    },
    "cypherpole_sig_mom": {
        "id": "a028d72b-7b66-4f87-add2-2e167180d31e",
        "description": "Momentum & Technical Oscillator Divergence",
        "strategy": "momentum"
    },
    "cypherpole_sig_val": {
        "id": "f4a0cf86-4749-4773-a027-5b9621ee70a5",
        "description": "Fundamental Value Yield Alpha",
        "strategy": "value"
    },
    "cypherpole_sig_vol": {
        "id": "0c9787d6-4550-4a0e-8058-f8d68719aa3b",
        "description": "Low-Volatility & Quality Defensive Alpha",
        "strategy": "low_vol"
    },
    "cypherpole_sig_alpha": {
        "id": "0907cb71-355b-4b8e-a0a4-58a1cbf3f87d",
        "description": "Supernova Multi-Horizon Composite",
        "strategy": "supernova"
    },
}

NEUTRALIZATION_PROPORTION = 0.35  # Project out 35% of broad market risk factors
