import os
import pytest
import numpy as np
import pandas as pd
from chimera.alpha_vault import AlphaVault, AlphaEntry
from chimera.ast_generator import ASTGenerator

@pytest.fixture
def temp_vault_path(tmp_path):
    return str(tmp_path / "alpha_vault.json")

def test_alpha_vault_save_load(temp_vault_path):
    vault = AlphaVault(vault_path=temp_vault_path)
    assert len(vault.entries) == 0

    gen = ASTGenerator(num_features=5, max_depth=3)
    tree = gen.random_tree()
    
    entry = AlphaEntry(
        name="chimera_alpha_01",
        formula=tree.to_formula(),
        instructions=[(ins.op, ins.out_reg, ins.in_reg1, ins.in_reg2, ins.feat_idx, ins.imm_val) for ins in tree.compile_to_bytecode()],
        sharpe=1.25,
        mean_corr=0.035,
        max_factor_corr=0.08,
        positive_era_ratio=0.72
    )
    vault.add_entry(entry)
    vault.save()

    assert os.path.exists(temp_vault_path)

    loaded_vault = AlphaVault.load(temp_vault_path)
    assert len(loaded_vault.entries) == 1
    loaded_entry = loaded_vault.entries[0]
    assert loaded_entry.name == "chimera_alpha_01"
    assert loaded_entry.sharpe == 1.25
    assert loaded_entry.formula == entry.formula

def test_augment_features(temp_vault_path):
    vault = AlphaVault(vault_path=temp_vault_path)
    gen = ASTGenerator(num_features=2, max_depth=2)
    tree = gen.random_tree()
    entry = AlphaEntry(
        name="chimera_alpha_01",
        formula=tree.to_formula(),
        instructions=[(ins.op, ins.out_reg, ins.in_reg1, ins.in_reg2, ins.feat_idx, ins.imm_val) for ins in tree.compile_to_bytecode()],
        sharpe=1.1,
        mean_corr=0.02,
        max_factor_corr=0.05,
        positive_era_ratio=0.68
    )
    vault.add_entry(entry)

    df = pd.DataFrame({
        "feature_0": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "feature_1": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "target": np.array([0.5, 0.75, 0.25, 0.5], dtype=np.float32)
    })

    feature_cols = ["feature_0", "feature_1"]
    augmented_df = vault.augment_dataframe(df, feature_cols=feature_cols)

    assert "chimera_alpha_01" in augmented_df.columns
    assert len(augmented_df) == 4
    assert not augmented_df["chimera_alpha_01"].isna().any()
