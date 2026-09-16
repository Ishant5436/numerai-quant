import os
import pytest
import numpy as np
import pandas as pd
from config import CHIMERA_VAULT_PATH
from chimera.alpha_vault import AlphaVault, AlphaEntry
from chimera.genetic_synthesizer import GeneticSynthesizer
from chimera.orthogonality_filter import TriHurdleFilter

def test_chimera_config_and_vault_integration(tmp_path):
    assert CHIMERA_VAULT_PATH.endswith("alpha_vault.json")
    
    # Test isolated vault lifecycle in temp path
    test_vault_file = str(tmp_path / "test_alpha_vault.json")
    vault = AlphaVault.load(test_vault_file)
    assert len(vault.entries) == 0

    # Create synthetic dataset with learnable signal
    np.random.seed(101)
    n_eras = 15
    rows_per_era = 80
    total = n_eras * rows_per_era
    eras = np.repeat([f"era_{i:03d}" for i in range(n_eras)], rows_per_era)
    
    f0 = np.random.randn(total).astype(np.float32)
    f1 = np.random.randn(total).astype(np.float32)
    features = np.column_stack([f0, f1])
    target = np.tanh(f0 - f1) + 0.05 * np.random.randn(total).astype(np.float32)

    filter_gate = TriHurdleFilter(min_sharpe=0.5, max_factor_corr=0.45, min_positive_era_ratio=0.50)
    synth = GeneticSynthesizer(
        num_features=2,
        pop_size=16,
        tournament_size=2,
        max_depth=3,
        vault_path=test_vault_file,
        filter_gate=filter_gate
    )

    vault = synth.evolve_on_dataset(features, target, eras, generations=2)
    assert os.path.exists(test_vault_file)

    # Test augment_dataframe
    df = pd.DataFrame({"feat_0": f0[:50], "feat_1": f1[:50]})
    aug_df = vault.augment_dataframe(df, feature_cols=["feat_0", "feat_1"])
    assert len(aug_df) == 50
    assert not aug_df.isna().any().any()
