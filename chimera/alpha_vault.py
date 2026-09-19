import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
import pandas as pd
import numpy as np

from chimera.evaluator import ChimeraEngine, ChimeraInstruction

@dataclass
class AlphaEntry:
    name: str
    formula: str
    instructions: List[Tuple[int, int, int, int, int, float]]
    sharpe: float
    mean_corr: float
    max_factor_corr: float
    positive_era_ratio: float

class AlphaVault:
    def __init__(self, vault_path: str = "data/alpha_vault.json"):
        self.vault_path = vault_path
        self.entries: List[AlphaEntry] = []

    def add_entry(self, entry: AlphaEntry):
        assert isinstance(entry, AlphaEntry), "Entry must be an AlphaEntry instance"
        # Prevent duplicates
        for existing in self.entries:
            if existing.formula == entry.formula:
                return
        self.entries.append(entry)

    def save(self):
        import tempfile
        abs_path = os.path.abspath(self.vault_path)
        dir_name = os.path.dirname(abs_path)
        os.makedirs(dir_name, exist_ok=True)
        data = {
            "version": "1.0",
            "entries": [asdict(e) for e in self.entries]
        }
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        os.replace(temp_name, abs_path)

    @classmethod
    def load(cls, vault_path: str = "data/alpha_vault.json") -> "AlphaVault":
        vault = cls(vault_path=vault_path)
        if not os.path.exists(vault_path):
            return vault
        try:
            with open(vault_path, "r") as f:
                data = json.load(f)
            for item in data.get("entries", []):
                entry = AlphaEntry(
                    name=item["name"],
                    formula=item["formula"],
                    instructions=[tuple(ins) for ins in item["instructions"]],
                    sharpe=float(item["sharpe"]),
                    mean_corr=float(item["mean_corr"]),
                    max_factor_corr=float(item["max_factor_corr"]),
                    positive_era_ratio=float(item["positive_era_ratio"])
                )
                vault.entries.append(entry)
        except Exception as err:
            print(f"Warning: Failed to load alpha vault from {vault_path}: {err}")
        return vault

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        if len(self.entries) == 0:
            return df

        assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
        assert len(feature_cols) > 0, "feature_cols cannot be empty"

        n_rows = len(df)
        if n_rows == 0:
            return df

        features = np.ascontiguousarray(df[feature_cols].values, dtype=np.float32)
        engine = ChimeraEngine(capacity_rows=max(1000, n_rows))

        try:
            for entry in self.entries:
                c_instrs = [
                    ChimeraInstruction(
                        op=ins[0],
                        out_reg=ins[1],
                        in_reg1=ins[2],
                        in_reg2=ins[3],
                        feat_idx=ins[4],
                        imm_val=ins[5]
                    ) for ins in entry.instructions
                ]
                col_data = engine.execute(c_instrs, features)
                df[entry.name] = col_data
        finally:
            engine.close()

        return df
