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
    Numerically Stable QR-Decomposition Linear Feature Neutralization.
    Projects prediction vectors away from the subspace spanned by risk factor matrix X.
    Invariant: For orthonormal Q (from X = QR), Projection = Q @ (Q.T @ P), O(N*K) time, 0 matrix inversion instability.
    """
    if proportion <= 0.0:
        return df

    neutralizers = extra_neutralizers if extra_neutralizers is not None else []
    neutralizers_matrix = df[neutralizers].values if neutralizers else None

    df_neutralized = df.copy()

    if neutralizers_matrix is not None and neutralizers_matrix.shape[1] > 0:
        # 1. Clean and zero-center feature matrix
        X = np.nan_to_num(neutralizers_matrix, nan=0.5)
        X = X - X.mean(axis=0)

        # 2. Economy-size QR decomposition: X = Q @ R
        # Q is an N x K matrix with orthonormal columns: Q^T @ Q = I_K
        Q, _ = np.linalg.qr(X, mode="reduced")

        for col in columns:
            col_values = df[col].values
            # Zero-centered rank uniform [-0.5, 0.5]
            P = rank_01(col_values) - 0.5

            # Orthogonal projection: Proj_X(P) = Q @ (Q^T @ P)
            projection = Q @ (Q.T @ P)
            neutralized_values = P - proportion * projection
            df_neutralized[col] = rank_01(neutralized_values)
    else:
        for col in columns:
            df_neutralized[col] = rank_01(df[col].values)

    return df_neutralized

