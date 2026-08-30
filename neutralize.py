"""
High-Performance Feature Neutralization Module
Orthogonalizes predictions against raw risk factors/features to maximize
Feature Neutral Correlation (FNC) and Meta Model Contribution (MMC).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_01(series_or_array) -> np.ndarray:
    """Normalize array strictly to uniform distribution [0.0, 1.0] with NaN safety."""
    arr = np.asarray(series_or_array, dtype=np.float64)
    result = np.full_like(arr, 0.5)  # NaN entries get neutral midpoint
    valid = ~np.isnan(arr)
    n_valid = valid.sum()
    if n_valid > 0:
        ranks = rankdata(arr[valid], method="average")
        result[valid] = (ranks - 0.5) / n_valid
    return result


def neutralize(
    df: pd.DataFrame,
    columns: list,
    extra_neutralizers: list = None,
    proportion: float = 0.25
) -> pd.DataFrame:
    """
    Linearly project predictions away from feature space per era/group.
    """
    if proportion <= 0.0:
        return df

    neutralizers = extra_neutralizers if extra_neutralizers is not None else []
    neutralizers_matrix = df[neutralizers].values if neutralizers else None

    df_neutralized = df.copy()

    for col in columns:
        col_values = df[col].values
        # Rank to standard Gaussian / uniform before projection
        transformed = rank_01(col_values) - 0.5

        if neutralizers_matrix is not None and neutralizers_matrix.shape[1] > 0:
            # Linear least squares projection
            # preds_neutral = preds - proportion * X @ (X^T X)^-1 X^T @ preds
            x = neutralizers_matrix - neutralizers_matrix.mean(axis=0)
            norm_x = x / (np.linalg.norm(x, axis=0) + 1e-8)
            projection = norm_x @ (norm_x.T @ transformed)
            neutralized_values = transformed - proportion * projection
        else:
            neutralized_values = transformed

        df_neutralized[col] = rank_01(neutralized_values)

    return df_neutralized
