import os
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
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
    feature_names: List[str] = field(default_factory=list)

class AlphaVault:
    def __init__(self, vault_path: str = "data/alpha_vault.json"):
        self.vault_path = vault_path
        self.entries: List[AlphaEntry] = []

    def add_entry(self, entry: AlphaEntry):
        assert isinstance(entry, AlphaEntry), "Entry must be an AlphaEntry instance"
        assert len(entry.instructions) > 0, "Instructions cannot be empty"
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
                    positive_era_ratio=float(item["positive_era_ratio"]),
                    feature_names=item.get("feature_names", [])
                )
                vault.entries.append(entry)
        except Exception as err:
            print(f"Warning: Failed to load alpha vault from {vault_path}: {err}")
        return vault

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
        if len(self.entries) == 0 or len(df) == 0:
            return df
        assert df.shape[0] > 0, "DataFrame rows must be > 0"

        n_rows = len(df)
        engine = ChimeraEngine(capacity_rows=max(1000, n_rows))
        new_cols = {}

        try:
            fallback_features = None
            if feature_cols is not None and len(feature_cols) > 0:
                available_cols = [c for c in feature_cols if c in df.columns]
                if len(available_cols) == len(feature_cols):
                    fallback_features = np.ascontiguousarray(df[feature_cols].values, dtype=np.float32)

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

                if getattr(entry, "feature_names", None) and len(entry.feature_names) > 0:
                    missing = [c for c in entry.feature_names if c not in df.columns]
                    if missing:
                        continue
                    feat_matrix = np.ascontiguousarray(df[entry.feature_names].values, dtype=np.float32)
                    col_data = engine.execute(c_instrs, feat_matrix)
                elif fallback_features is not None:
                    col_data = engine.execute(c_instrs, fallback_features)
                else:
                    continue

                new_cols[entry.name] = col_data
        finally:
            engine.close()

        if not new_cols:
            return df

        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        assert len(new_cols_df) == len(df), "Augmented column rows must match original dataframe"
        return pd.concat([df, new_cols_df], axis=1)

    def compute_composite_alpha(
        self,
        df: pd.DataFrame,
        top_k: int = 25,
        feature_cols: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
        assert top_k > 0, "top_k must be > 0"
        if len(self.entries) == 0 or len(df) == 0:
            return None

        sorted_entries = sorted(self.entries, key=lambda e: e.sharpe, reverse=True)[:top_k]
        sub_vault = AlphaVault()
        sub_vault.entries = sorted_entries
        augmented = sub_vault.augment_dataframe(df, feature_cols=feature_cols)

        alpha_col_names = [e.name for e in sorted_entries if e.name in augmented.columns]
        if not alpha_col_names:
            return None

        ranked_alphas = []
        for col in alpha_col_names:
            series = augmented[col].values
            if np.std(series) > 1e-7:
                ranked = pd.Series(series).rank(pct=True).values.astype(np.float32)
                ranked_alphas.append(ranked)

        if not ranked_alphas:
            return None

        composite = np.mean(ranked_alphas, axis=0)
        final_rank = pd.Series(composite).rank(pct=True).values.astype(np.float32)
        assert len(final_rank) == len(df), "Composite length must match dataframe length"
        return final_rank

