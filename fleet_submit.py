#!/usr/bin/env python3
"""
Numerai Multi-Model Fleet Autonomous Submission Engine (Tri-Ensemble Stacking Architecture)
Blends LightGBM + XGBoost + CatBoost predictions across 5 orthogonal feature groups:
1. Strategy 1 (Flagship / 'cypherpole') -> Core Tri-Ensemble (705 features, 25% Neutralized)
2. Strategy 2 (Fundamental / 'fund')    -> Fundamental Tri-Ensemble (186 features, 35% Neutralized)
3. Strategy 3 (Momentum / 'mom')        -> Momentum Tri-Ensemble (133 features, 40% Neutralized)
4. Strategy 4 (Macro / 'macro')         -> Macro Regime Tri-Ensemble (278 features, 45% Neutralized)
5. Strategy 5 (Residual / 'res')        -> Constitution Residual Tri-Ensemble (155 features, 50% Neutralized)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI
from config import FEATURES_JSON, DATA_DIR
from neutralize import neutralize, rank_01

load_dotenv(os.path.expanduser("~/.env"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRI_DIR = os.path.join(BASE_DIR, "models", "tri_ensemble_fleet")
ORTHO_DIR = os.path.join(BASE_DIR, "models", "orthogonal_fleet")


def load_feature_groups() -> dict:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    medium_set = set(meta["feature_sets"]["medium"])

    def get_subset(keys):
        combined = set()
        for k in keys:
            combined.update(meta["feature_sets"].get(k, []))
        return sorted(list(combined.intersection(medium_set)))

    groups = {
        "all_medium": meta["feature_sets"]["medium"],
        "fundamental": get_subset(["intelligence", "charisma", "wisdom"]),
        "momentum": get_subset(["strength", "dexterity", "agility"]),
        "macro": get_subset(["serenity", "sunshine", "midnight"]),
        "constitution": get_subset(["constitution", "dexterity"]),
        "quality_defensive": get_subset(["serenity", "wisdom", "intelligence"]),
        "trend_velocity": get_subset(["agility", "strength", "sunshine"]),
        "value_capital": get_subset(["charisma", "wisdom", "constitution"]),
        "macro_tail": get_subset(["midnight", "rain", "serenity"]),
        "alpha_conviction": get_subset(["intelligence", "strength", "wisdom"]),
        "volatility_defensive": get_subset(["serenity", "constitution", "rain"]),
        "risk_parity": get_subset(["sunshine", "intelligence", "dexterity"]),
        "macro_hedged": get_subset(["midnight", "agility", "rain"])
    }
    return groups


def resolve_strategy_config(model_name: str, idx: int) -> tuple[int, str, float]:
    """
    Deterministically maps a model name and index to its orthogonal strategy specification:
    Returns (strat_id, feature_group_key, neutralization_proportion).
    Guarantees zero unhandled states, zero uninitialized variables, and clean testability.
    Phase 1: Explicit keyword matches in model name take absolute priority.
    Phase 2: Fallback to modulo slot routing for generic/unbranded model names.
    """
    name_lower = (model_name or "").lower()

    # 1. Explicit keyword matching in model name
    if "xerxes" in name_lower:
        return (4, "macro", 0.45)
    elif "macro_tail" in name_lower or "tail" in name_lower or "sam" in name_lower:
        return (10, "macro_tail", 0.45)
    elif "fund" in name_lower or "jeremy" in name_lower:
        return (2, "fundamental", 0.35)
    elif "mom" in name_lower or "victor" in name_lower:
        return (3, "momentum", 0.40)
    elif "res" in name_lower or "delta" in name_lower:
        return (5, "constitution", 0.50)
    elif "cyrus" in name_lower:
        return (6, "all_medium", 0.30)
    elif "qual" in name_lower or "def" in name_lower:
        return (7, "quality_defensive", 0.35)
    elif "vel" in name_lower or "trend" in name_lower:
        return (8, "trend_velocity", 0.40)
    elif "val" in name_lower or "cap" in name_lower:
        return (9, "value_capital", 0.35)
    elif "macro_hedged" in name_lower or "hedged" in name_lower:
        return (15, "macro_hedged", 0.50)
    elif "macro" in name_lower:
        return (4, "macro", 0.45)
    elif "alpha" in name_lower:
        return (11, "alpha_conviction", 0.30)
    elif "vol" in name_lower:
        return (12, "volatility_defensive", 0.40)
    elif "sharpe" in name_lower:
        return (13, "risk_parity", 0.35)
    elif "deep" in name_lower:
        return (14, "all_medium", 0.25)

    # 2. Modulo slot fallback for generic model names
    slot = idx % 15
    slot_map = {
        0: (1, "all_medium", 0.25),
        1: (2, "fundamental", 0.35),
        2: (3, "momentum", 0.40),
        3: (4, "macro", 0.45),
        4: (5, "constitution", 0.50),
        5: (6, "all_medium", 0.30),
        6: (7, "quality_defensive", 0.35),
        7: (8, "trend_velocity", 0.40),
        8: (9, "value_capital", 0.35),
        9: (10, "macro_tail", 0.45),
        10: (11, "alpha_conviction", 0.30),
        11: (12, "volatility_defensive", 0.40),
        12: (13, "risk_parity", 0.35),
        13: (14, "all_medium", 0.25),
        14: (15, "macro_hedged", 0.50),
    }
    return slot_map.get(slot, (1, "all_medium", 0.25))


def generate_tri_ensemble_prediction(live_df: pd.DataFrame, strat_id: int, feature_subset: list, neut_proportion: float, neutralizer_feats: list, allow_mock_fallback: bool = False) -> np.ndarray:
    lgb_path = os.path.join(TRI_DIR, f"lgb_strat_{strat_id}.pkl")
    xgb_path = os.path.join(TRI_DIR, f"xgb_strat_{strat_id}.pkl")
    cb_path = os.path.join(TRI_DIR, f"cb_strat_{strat_id}.pkl")

    if os.path.exists(lgb_path) and os.path.exists(xgb_path) and os.path.exists(cb_path):
        m_lgb = joblib.load(lgb_path)
        m_xgb = joblib.load(xgb_path)
        m_cb = joblib.load(cb_path)

        p_lgb = rank_01(m_lgb.predict(live_df[feature_subset]))
        p_xgb = rank_01(m_xgb.predict(live_df[feature_subset]))
        p_cb = rank_01(m_cb.predict(live_df[feature_subset]))

        raw_pred = 0.40 * p_lgb + 0.30 * p_xgb + 0.30 * p_cb
    else:
        # Fallback to single LightGBM
        single_path = os.path.join(ORTHO_DIR, f"lgb_strat_{strat_id}.pkl")
        if os.path.exists(single_path):
            model = joblib.load(single_path)
            raw_pred = model.predict(live_df[feature_subset])
        elif allow_mock_fallback:
            # Explicitly restricted to synthetic test environments where weights are gitignored
            raw_pred = np.mean(live_df[feature_subset].values, axis=1)
        else:
            raise FileNotFoundError(
                f"Production Error: No model weights found for strategy {strat_id} at {TRI_DIR} or {single_path}. "
                "Refusing to degrade to untrained feature averages during live competition submission."
            )

    live_copy = live_df.copy()
    live_copy["pred"] = rank_01(raw_pred)
    live_copy = neutralize(live_copy, ["pred"], extra_neutralizers=neutralizer_feats, proportion=neut_proportion)
    return rank_01(live_copy["pred"].values)


def main():
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")

    if "$" in auth and not (public_id and secret_key):
        public_id, secret_key = auth.split("$", 1)

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round = napi.get_current_round()
    models = napi.get_models()

    print(f"=== Numerai Fleet Autonomous Tri-Ensemble Submitter: Round {current_round} ===")
    print(f"Connected Account Models ({len(models)}): {models}")

    groups = load_feature_groups()
    live_path = os.path.join(DATA_DIR, "live.parquet")

    print("\nDownloading active live.parquet dataset...")
    napi.download_dataset("v5.0/live.parquet", live_path)
    live_df = pd.read_parquet(live_path, columns=groups["all_medium"])
    print(f"Live market universe loaded: {len(live_df)} assets")
    neutralizer_feats = groups["all_medium"][:60]

    for idx, (model_name, model_id) in enumerate(models.items()):
        print(f"\n--- Processing Model [{idx+1}/{len(models)}]: '{model_name}' (ID: {model_id}) ---")
        preds_path = os.path.join(DATA_DIR, f"predictions_{model_name}_round_{current_round}.csv")

        strat_id, group_key, neut_prop = resolve_strategy_config(model_name, idx)
        print(f"Applying Strategy {strat_id} ('{group_key}', {len(groups[group_key])} features, {neut_prop*100:.0f}% Neutralized)...")
        preds = generate_tri_ensemble_prediction(live_df, strat_id, groups[group_key], neut_prop, neutralizer_feats)

        sub_df = pd.DataFrame({"id": live_df.index, "prediction": preds})
        sub_df.to_csv(preds_path, index=False)
        print(f"Saved {len(sub_df)} predictions -> {preds_path}")

        print(f"Uploading submission to Numerai (Model ID: {model_id})...")
        sub_id = napi.upload_predictions(preds_path, model_id=model_id)
        print(f"[SUCCESS]  Successfully submitted '{model_name}' to Round {current_round}! Submission ID: {sub_id}")

    print(f"\n[COMPLETE]  Fleet submission complete across all {len(models)} models for Round {current_round}!")


if __name__ == "__main__":
    main()
