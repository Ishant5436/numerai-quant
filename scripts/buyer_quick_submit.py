"""
Turnkey Staker Prediction Ingestion & Auto-Submission Pipeline.
Allows Numerbay trial users and subscribers to validate, optionally blend,
and submit cypherpole predictions to Numerai in a single CLI command.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""
import os
import sys
import ast
import argparse
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def validate_prediction_file(filepath: str) -> Tuple[bool, str]:
    """Validates CSV format, columns, value bounds, and absence of NaNs."""
    assert isinstance(filepath, str) and len(filepath) > 0, "filepath must be non-empty string"
    if not os.path.exists(filepath):
        return False, "File does not exist"

    try:
        df = pd.read_csv(filepath)
    except Exception as err:
        return False, f"Failed to parse CSV: {err}"

    required_cols = {"id", "prediction"}
    if not required_cols.issubset(df.columns):
        return False, f"Missing required columns: {required_cols - set(df.columns)}"

    if df["prediction"].isna().any():
        return False, "Prediction column contains NaNs"

    p_min = df["prediction"].min()
    p_max = df["prediction"].max()
    if p_min < 0.0 or p_max > 1.0:
        return False, f"Values out of bounds [0.0, 1.0]: [{p_min}, {p_max}]"

    assert len(df) > 0, "Dataframe must not be empty"
    return True, f"VALID ({len(df):,} rows)"


def blend_predictions(
    df_user: pd.DataFrame, df_cypher: pd.DataFrame, cypher_weight: float = 0.25
) -> pd.DataFrame:
    """Convexly blends user predictions with cypherpole predictions via rank percentile."""
    assert isinstance(df_user, pd.DataFrame) and isinstance(df_cypher, pd.DataFrame), "DataFrames required"
    assert 0.0 <= cypher_weight <= 1.0, "cypher_weight must be in [0.0, 1.0]"

    merged = pd.merge(df_user[["id", "prediction"]], df_cypher[["id", "prediction"]], on="id", suffixes=("_user", "_cypher"))
    w_user = 1.0 - cypher_weight
    blended_raw = w_user * merged["prediction_user"] + cypher_weight * merged["prediction_cypher"]

    # Rank percentile normalization to uniform [0.0, 1.0]
    ranks = blended_raw.rank(pct=True).values
    blended_df = pd.DataFrame({"id": merged["id"], "prediction": ranks})

    assert len(blended_df) == len(merged), "Row count must match"
    assert not blended_df["prediction"].isna().any(), "Blended predictions cannot contain NaNs"
    return blended_df


def submit_to_numerai(filepath: str, model_name: str) -> Dict[str, Any]:
    """Submits validated predictions to Numerai tournament slot."""
    assert isinstance(filepath, str) and os.path.exists(filepath), "File must exist"
    assert isinstance(model_name, str) and len(model_name) > 0, "model_name required"

    from numerapi import NumerAPI
    napi = NumerAPI()
    models = napi.get_models()
    model_id = models.get(model_name)
    if not model_id:
        return {"status": "ERROR", "message": f"Model {model_name} not found in account"}

    try:
        sub_id = napi.upload_predictions(filepath, model_id=model_id)
        assert sub_id is not None, "Submission ID must be returned"
        return {"status": "SUCCESS", "submission_id": sub_id, "model": model_name}
    except Exception as err:
        return {"status": "ERROR", "message": str(err)}


def audit_safety_invariants_local() -> bool:
    """Mechanically verifies Power of 10 invariants: functions <= 60 lines, asserts >= 2."""
    filepath = os.path.abspath(__file__)
    assert os.path.exists(filepath), "Source file must exist"
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)

    passed = True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines = (node.end_lineno - node.lineno) + 1
            assert lines <= 60, f"Function {node.name} exceeds 60 lines: {lines}"
            assert_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))
            assert (
                assert_count >= 2
            ), f"Function {node.name} has fewer than 2 asserts: {assert_count}"

    assert passed is True, "Audit must return true"
    return passed


def main():
    parser = argparse.ArgumentParser(description="Turnkey Numerai Submission for Cypherpole Buyers")
    parser.add_argument("--file", type=str, required=True, help="Path to cypherpole predictions CSV")
    parser.add_argument("--model", type=str, required=True, help="Target model slot on Numerai")
    parser.add_argument("--blend", type=str, default="", help="Optional user predictions CSV to blend")
    parser.add_argument("--weight", type=float, default=0.25, help="Cypherpole blend weight (default 0.25)")
    args = parser.parse_args()

    assert len(args.file) > 0, "file arg required"
    assert len(args.model) > 0, "model arg required"

    valid, msg = validate_prediction_file(args.file)
    if not valid:
        print(f"[ERROR] Prediction file invalid: {msg}")
        sys.exit(1)

    target_file = args.file
    if args.blend and os.path.exists(args.blend):
        print(f"[*] Blending {args.blend} with {args.file} (Weight: {args.weight:.0%})...")
        df_u = pd.read_csv(args.blend)
        df_c = pd.read_csv(args.file)
        df_b = blend_predictions(df_u, df_c, cypher_weight=args.weight)
        target_file = "predictions_blended.csv"
        df_b.to_csv(target_file, index=False)

    print(f"[*] Submitting {target_file} to Numerai slot '{args.model}'...")
    res = submit_to_numerai(target_file, args.model)
    print(f"[RESULT] {res}")


if __name__ == "__main__":
    main()
