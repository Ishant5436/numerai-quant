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
os.makedirs(DATA_DIR, exist_ok=True)
TRI_DIR = "/Users/ishantpanchal/numerai-quant/models/tri_ensemble_fleet"
ORTHO_DIR = "/Users/ishantpanchal/numerai-quant/models/orthogonal_fleet"


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
        "constitution": get_subset(["constitution", "dexterity"])
    }
    return groups


def generate_tri_ensemble_prediction(live_df: pd.DataFrame, strat_id: int, feature_subset: list, neut_proportion: float, neutralizer_feats: list) -> np.ndarray:
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
        model = joblib.load(os.path.join(ORTHO_DIR, f"lgb_strat_{strat_id}.pkl"))
        raw_pred = model.predict(live_df[feature_subset])

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

        name_lower = model_name.lower()
        if "fund" in name_lower or "jeremy" in name_lower or (idx % 5 == 1):
            print("Applying Strategy 2: Fundamental Tri-Ensemble (186 features, 35% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, 2, groups["fundamental"], 0.35, neutralizer_feats)
        elif "mom" in name_lower or "victor" in name_lower or (idx % 5 == 2):
            print("Applying Strategy 3: Momentum Tri-Ensemble (133 features, 40% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, 3, groups["momentum"], 0.40, neutralizer_feats)
        elif "macro" in name_lower or "xerxes" in name_lower or (idx % 5 == 3):
            print("Applying Strategy 4: Macro Regime Tri-Ensemble (278 features, 45% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, 4, groups["macro"], 0.45, neutralizer_feats)
        elif "res" in name_lower or "delta" in name_lower or (idx % 5 == 4):
            print("Applying Strategy 5: Constitution Residual Tri-Ensemble (155 features, 50% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, 5, groups["constitution"], 0.50, neutralizer_feats)
        else:
            print("Applying Strategy 1: Core Tri-Ensemble Flagship (705 features, 25% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, 1, groups["all_medium"], 0.25, neutralizer_feats)

        sub_df = pd.DataFrame({"id": live_df.index, "prediction": preds})
        sub_df.to_csv(preds_path, index=False)
        print(f"Saved {len(sub_df)} predictions -> {preds_path}")

        print(f"Uploading submission to Numerai (Model ID: {model_id})...")
        sub_id = napi.upload_predictions(preds_path, model_id=model_id)
        print(f"[SUCCESS]  Successfully submitted '{model_name}' to Round {current_round}! Submission ID: {sub_id}")

    print(f"\n[COMPLETE]  Fleet submission complete across all {len(models)} models for Round {current_round}!")


if __name__ == "__main__":
    main()
