#!/usr/bin/env python3
"""
Numerai Automated Alpha Ensemble Live Inference & Submission Engine
- Loads ~/.env credentials
- Downloads active round live.parquet
- Runs inference across all multi-target LightGBM models
- Blends predictions via rank-averaging
- Applies 25% feature neutralization against risk factors
- Exports predictions.csv
- Uploads directly to Numerai tournament via API
"""

import os
import glob
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI
from config import (
    FEATURE_SET,
    MODEL_DIR,
    FEATURES_JSON,
    DATA_DIR,
    NEUTRALIZATION_PROPORTION
)
from neutralize import neutralize, rank_01

# Load credentials from ~/.env
load_dotenv(os.path.expanduser("~/.env"))
os.makedirs(DATA_DIR, exist_ok=True)


def get_feature_list() -> list:
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    return meta["feature_sets"][FEATURE_SET]


def main():
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")
    model_id = os.environ.get("NUMERAI_MODEL_ID", "")

    if "$" in auth and not (public_id and secret_key):
        public_id, secret_key = auth.split("$", 1)

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round = napi.get_current_round()
    print(f"=== Numerai Alpha Ensemble Pipeline: Round {current_round} ===")

    if not model_id:
        models = napi.get_models()
        if models:
            model_name = list(models.keys())[0]
            model_id = models[model_name]
            print(f"Using account model: {model_name} ({model_id})")

    features = get_feature_list()
    live_path = os.path.join(DATA_DIR, "live.parquet")
    preds_path = os.path.join(DATA_DIR, f"predictions_round_{current_round}.csv")

    print("Downloading active live.parquet dataset...")
    napi.download_dataset("v5.0/live.parquet", live_path)

    print("Loading live feature matrix...")
    live_df = pd.read_parquet(live_path, columns=features)

    # Check for ensemble models
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "lgb_*.pkl")))
    if not model_files:
        raise FileNotFoundError("No trained models found in MODEL_DIR. Run train_ensemble.py first.")

    print(f"Running inference across {len(model_files)} specialized models...")
    predictions = []
    for mf in model_files:
        target_name = os.path.basename(mf).replace("lgb_", "").replace(".pkl", "")
        print(f"  • Inferring from model: {target_name}")
        model = joblib.load(mf)
        raw_pred = model.predict(live_df[features])
        ranked_pred = rank_01(raw_pred)
        predictions.append(ranked_pred)

    # Rank-average predictions across ensemble
    blended_preds = pd.DataFrame(predictions).T.mean(axis=1).values
    live_df["prediction_raw"] = rank_01(blended_preds)

    # Apply 25% feature neutralization
    print(f"Applying {int(NEUTRALIZATION_PROPORTION*100)}% feature neutralization against risk factors...")
    neutralizer_feats = features[:50]
    live_df = neutralize(
        live_df,
        columns=["prediction_raw"],
        extra_neutralizers=neutralizer_feats,
        proportion=NEUTRALIZATION_PROPORTION
    )
    final_preds = rank_01(live_df["prediction_raw"].values)

    # Format submission dataframe
    submission_df = pd.DataFrame({
        "id": live_df.index,
        "prediction": final_preds
    })
    submission_df.to_csv(preds_path, index=False)
    print(f"✅ Generated {len(submission_df)} neutralized ensemble predictions -> {preds_path}")

    # Submit
    if public_id and secret_key:
        print(f"Uploading Alpha Ensemble submission to Numerai (Model ID: {model_id})...")
        submission_id = napi.upload_predictions(preds_path, model_id=model_id)
        print(f"🚀 Successfully submitted Alpha Ensemble to Round {current_round}! Submission ID: {submission_id}")
    else:
        print("\n⚠️ API keys not found in environment.")


if __name__ == "__main__":
    main()
