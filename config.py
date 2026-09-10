"""
Numerai Quant Configuration (Alpha Ensemble Standard - v5.0)
"""
import os

FEATURE_SET = "medium"  # 705 features for deep multi-factor alpha
FNCV3_SET = "fncv3_features"  # 400 official risk factor features for FNCv3 neutralization

ENSEMBLE_TARGETS = [
    "target",            # Benchmark Cyrus (core alpha)
    "target_cyrusd_60",   # 60-Day Ender Core Benchmark
    "target_agnes_60",   # 60-Day Orthogonal Residual Returns
    "target_victor_60",  # 60-Day Volatility-Adjusted Returns
    "target_jeremy_60",  # 60-Day Value/Quality Momentum Returns
    "target_xerxes_60"   # 60-Day Tail-Risk Defense Returns
]

# 60-Day Multi-Target Quintet for 95th+ Percentile Out-of-Sample Performance
ENSEMBLE_TARGETS_60D = [
    "target_cyrusd_60",   # 60-Day Ender Core Benchmark
    "target_agnes_60",    # 60-Day Orthogonal Residual Alpha
    "target_victor_60",   # 60-Day Volatility-Adjusted Returns
    "target_jeremy_60",   # 60-Day Quality/Value Momentum
    "target_xerxes_60"    # 60-Day Tail-Risk Defense
]

NEUTRALIZATION_PROPORTION = 0.35

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_60D_DIR = os.path.join(MODEL_DIR, "ensemble_60d")
ORTHO_60D_DIR = os.path.join(MODEL_DIR, "orthogonal_fleet_60d")
FEATURES_JSON = os.path.join(BASE_DIR, "features.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_60D_DIR, exist_ok=True)
os.makedirs(ORTHO_60D_DIR, exist_ok=True)
TARGET_COL = "target"
MODEL_PATH = os.path.join(MODEL_DIR, "lgb_target.pkl")

# Complete 25-Strategy 60-Day Fleet Architecture Mapping
# Format: strat_id -> (target_name, feature_group_key, default_neutralization_proportion)
FLEET_STRATEGY_MAP_60D = {
    1: ("5_target_quintet", "all_medium", 0.35),
    2: ("target_jeremy_60", "fundamental", 0.35),
    3: ("target_victor_60", "momentum", 0.40),
    4: ("target_xerxes_60", "macro", 0.45),
    5: ("target_delta_60", "constitution", 0.50),
    6: ("target_cyrusd_60", "all_medium", 0.30),
    7: ("target_jeremy_60", "quality_defensive", 0.35),
    8: ("target_caroline_60", "trend_velocity", 0.40),
    9: ("target_bravo_60", "value_capital", 0.35),
    10: ("target_xerxes_60", "macro_tail", 0.45),
    11: ("target_alpha_60", "alpha_conviction", 0.30),
    12: ("target_victor_60", "volatility_defensive", 0.40),
    13: ("target_claudia_60", "risk_parity", 0.35),
    14: ("target_teager2b_60", "all_medium", 0.25),
    15: ("target_sam_60", "macro_hedged", 0.50),
    16: ("target_bravo_60", "fundamental_value", 0.35),
    17: ("target_charlie_60", "low_beta_defensive", 0.40),
    18: ("target_delta_60", "residual_alpha", 0.50),
    19: ("target_echo_60", "mean_reversion", 0.30),
    20: ("target_ralph_60", "factor_momentum", 0.35),
    21: ("target_rowan_60", "high_sharpe_quality", 0.30),
    22: ("target_sam_60", "macro_tail_liquidity", 0.45),
    23: ("target_tyler_60", "earnings_quality", 0.35),
    24: ("target_waldo_60", "sentiment_divergence", 0.40),
    25: ("target_victor_60", "vol_adjusted_alpha", 0.35),
}

# Optimized LightGBM Hyperparameters for ARM64 M5 Pro
LGB_PARAMS = {
    "n_estimators": 450,
    "learning_rate": 0.02,
    "max_depth": 5,
    "num_leaves": 31,
    "colsample_bytree": 0.1,  # Feature subsampling for extreme speed & variance reduction across 705 features
    "subsample": 0.8,
    "n_jobs": -1,
    "random_state": 42,
    "importance_type": "gain"
}
