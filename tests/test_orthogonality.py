import numpy as np
from chimera.orthogonality_filter import TriHurdleFilter

def test_tri_hurdle_accepts_good_alpha():
    np.random.seed(42)
    n_eras = 30
    rows_per_era = 200
    total_rows = n_eras * rows_per_era
    
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)
    
    # Base features: pure random
    f0 = np.random.randn(total_rows).astype(np.float32)
    f1 = np.random.randn(total_rows).astype(np.float32)
    feature_matrix = np.column_stack([f0, f1])

    # Target
    target = np.random.randn(total_rows).astype(np.float32)

    # Signal with strong positive correlation to target across all eras, but orthogonal to f0, f1
    signal = target + 0.3 * np.random.randn(total_rows).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=0.8, max_factor_corr=0.15, min_positive_era_ratio=0.60)
    result = filter_gate.evaluate(signal, target, eras, feature_matrix)

    assert result.passed, f"Expected pass, got: {result.failure_reason}"
    assert result.sharpe > 0.8
    assert result.max_factor_corr < 0.15
    assert result.positive_era_ratio >= 0.60

def test_tri_hurdle_rejects_collinear_signal():
    np.random.seed(42)
    n_eras = 20
    rows_per_era = 100
    total_rows = n_eras * rows_per_era
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)

    f0 = np.random.randn(total_rows).astype(np.float32)
    feature_matrix = f0.reshape(-1, 1)

    # Signal collinear with feature 0 (rho ~ 0.95)
    signal = f0 + 0.05 * np.random.randn(total_rows).astype(np.float32)
    # Target has high correlation with signal so it passes Sharpe, but fails Orthogonality
    target = signal + 0.1 * np.random.randn(total_rows).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=0.5, max_factor_corr=0.12, min_positive_era_ratio=0.50)
    result = filter_gate.evaluate(signal, target, eras, feature_matrix)

    assert not result.passed
    assert "Orthogonality failure" in result.failure_reason

def test_tri_hurdle_rejects_fluke_stability():
    np.random.seed(42)
    n_eras = 20
    rows_per_era = 100
    total_rows = n_eras * rows_per_era
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)

    feature_matrix = np.random.randn(total_rows, 2).astype(np.float32)
    target = np.random.randn(total_rows).astype(np.float32)
    
    # Random noise signal
    signal = np.random.randn(total_rows).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=1.0, max_factor_corr=0.12, min_positive_era_ratio=0.65)
    result = filter_gate.evaluate(signal, target, eras, feature_matrix)

    assert not result.passed
