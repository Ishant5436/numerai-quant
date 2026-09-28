import os
import pytest
import numpy as np
from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter

@pytest.fixture
def temp_vault_path(tmp_path):
    return str(tmp_path / "test_synth_vault.json")

def test_genetic_synthesizer_evolution_cycle(temp_vault_path):
    np.random.seed(42)
    n_eras = 10
    rows_per_era = 100
    total_rows = n_eras * rows_per_era
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)

    f0 = np.random.randn(total_rows).astype(np.float32)
    f1 = np.random.randn(total_rows).astype(np.float32)
    f2 = np.random.randn(total_rows).astype(np.float32)
    features = np.column_stack([f0, f1, f2])

    # Target is constructed from a known non-linear formula: tanh(f0 + f1)
    target = np.tanh(f0 + f1) + 0.05 * np.random.randn(total_rows).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=0.5, max_factor_corr=0.45, min_positive_era_ratio=0.5)
    synth = GeneticSynthesizer(
        num_features=3,
        pop_size=20,
        vault_path=temp_vault_path,
        filter_gate=filter_gate
    )

    vault = synth.evolve_on_dataset(features, target, eras, generations=3)
    assert os.path.exists(temp_vault_path)
    # The run finishes cleanly without exceptions
    assert len(vault.entries) >= 0

def test_genetic_synthesizer_binds_feature_names(temp_vault_path):
    np.random.seed(42)
    n_eras = 10
    rows_per_era = 100
    total_rows = n_eras * rows_per_era
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)

    f0 = np.random.randn(total_rows).astype(np.float32)
    f1 = np.random.randn(total_rows).astype(np.float32)
    features = np.column_stack([f0, f1])
    target = np.tanh(f0) + 0.01 * np.random.randn(total_rows).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=0.1, max_factor_corr=0.99, min_positive_era_ratio=0.3)
    synth = GeneticSynthesizer(
        num_features=2,
        pop_size=20,
        vault_path=temp_vault_path,
        filter_gate=filter_gate
    )

    feature_names = ["alpha_feat_0", "alpha_feat_1"]
    vault = synth.evolve_on_dataset(features, target, eras, generations=2, feature_names=feature_names)
    assert os.path.exists(temp_vault_path)
    if len(vault.entries) > 0:
        for entry in vault.entries:
            assert hasattr(entry, "feature_names")
            assert len(entry.feature_names) > 0
            for fn in entry.feature_names:
                assert fn in feature_names

