"""
Numerai Quant Configuration (Alpha Ensemble Standard - v5.0)
"""
import os

FEATURE_SET = "medium"  # 705 features for deep multi-factor alpha
ENSEMBLE_TARGETS = [
    "target",            # Benchmark Cyrus (core alpha)
    "target_cyrusd_60",   # 60-Day Ender Core Benchmark
    "target_agnes_60",   # 60-Day Orthogonal Residual Returns
    "target_victor_60",  # 60-Day Volatility-Adjusted Returns
    "target_jeremy_60",  # 60-Day Value/Quality Momentum Returns
    "target_xerxes_60"   # 60-Day Tail-Risk Defense Returns
]
NEUTRALIZATION_PROPORTION = 0.25

MODEL_DIR = "/Users/ishantpanchal/numerai-quant/models"
FEATURES_JSON = "/Users/ishantpanchal/numerai-quant/features.json"
DATA_DIR = "/Users/ishantpanchal/numerai-quant/data"

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
