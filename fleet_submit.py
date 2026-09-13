#!/usr/bin/env python3
"""
Numerai Multi-Model Fleet Autonomous Submission Engine (Hybrid 15-Model Quantitative Fleet)
Architecture:
1. Strategies 1-5 (Flagship Tier): Tri-Ensemble Stacking (LightGBM + XGBoost + CatBoost 40/30/30 blend across
   5 orthogonal feature groups with 25% - 50% linear QR feature neutralization).
2. Strategies 6-15 (Specialist Tier): Orthogonal Factor Specialists (Dedicated LightGBM models targeting
   uncorrelated factor sub-regimes: Quality Defensive, Trend Velocity, Value Capital, Macro Tail,
   Alpha Conviction, Volatility Defensive, Risk Parity, Macro Hedged).
"""

import os
import time
import json
import joblib
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI
from config import (
    DATA_DIR,
    EXPLICIT_MODEL_ROUTING,
    FEATURES_JSON,
    FLEET_STRATEGY_MAP_60D,
    FLAGSHIP_ANCHOR_WEIGHT,
    ORTHO_60D_DIR,
)
from neutralize import neutralize, rank_01

load_dotenv(os.path.expanduser("~/.env"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRI_DIR = os.path.join(BASE_DIR, "models", "tri_ensemble_fleet")
ORTHO_DIR = os.path.join(BASE_DIR, "models", "orthogonal_fleet")
MODEL_60D_DIR = os.path.join(BASE_DIR, "models", "ensemble_60d")


def robust_api_call(func, *args, max_retries: int = 5, base_delay: float = 2.0, description: str = "API Operation", **kwargs):
    """
    Execute a NumerAPI or network operation with exponential backoff.
    Mitigates HTTP 429 (rate limits), 5xx server errors, connection resets, and transient timeouts.
    Fails fast immediately on permanent authentication errors (401/403).
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if "unauthorized" in err_msg or "401" in err_msg or "forbidden" in err_msg or "403" in err_msg or "invalid authentication" in err_msg:
                print(f"[FATAL] Permanent auth error in {description}: {e}. Aborting retries immediately.")
                raise e
            if attempt == max_retries:
                break
            sleep_time = base_delay * (2 ** (attempt - 1))
            print(f"[RETRY] {description} failed (Attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    raise RuntimeError(f"{description} failed permanently after {max_retries} attempts: {last_err}")


def safe_upload_predictions(napi: NumerAPI, file_path: str, model_id: str, timeout=(10, 600)) -> str:
    """
    Robust upload wrapper that explicitly verifies AWS S3 HTTP PUT status code (2xx)
    before issuing the create_submission GraphQL mutation.
    Fixes the upstream numerapi vulnerability where failed PUT uploads silently proceed to create_submission.
    """
    upload_auth = napi._upload_auth(
        "submission_upload_auth", file_path, napi.tournament_id, model_id
    )
    headers = {"x_compute_id": os.getenv("NUMERAI_COMPUTE_ID")}
    with open(file_path, "rb") as file:
        put_resp = requests.put(
            upload_auth["url"], data=file.read(), headers=headers, timeout=timeout
        )
    put_resp.raise_for_status()

    create_query = """
        mutation($filename: String!
                 $tournament: Int!
                 $modelId: String
                 $triggerId: String,
                 $dataDatestamp: Int) {
            create_submission(filename: $filename
                              tournament: $tournament
                              modelId: $modelId
                              triggerId: $triggerId
                              source: "numerapi"
                              dataDatestamp: $dataDatestamp) {
                id
            }
        }
    """
    arguments = {
        "filename": upload_auth["filename"],
        "tournament": napi.tournament_id,
        "modelId": model_id,
        "triggerId": os.getenv("TRIGGER_ID", None),
        "dataDatestamp": None,
    }
    create = napi.raw_query(create_query, arguments, authorization=True)
    submission_id = create["data"]["create_submission"]["id"]
    return submission_id



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
        "fncv3_features": meta["feature_sets"]["fncv3_features"],
        "fundamental": get_subset(["intelligence", "charisma", "wisdom"]),
        "momentum": get_subset(["strength", "dexterity", "agility"]),
        "macro": get_subset(["serenity", "sunshine", "midnight"]),
        "constitution": get_subset(["constitution"]),
        "quality_defensive": get_subset(["serenity", "wisdom", "intelligence"]),
        "trend_velocity": get_subset(["agility", "strength", "sunshine"]),
        "value_capital": get_subset(["charisma", "wisdom", "constitution"]),
        "macro_tail": get_subset(["midnight", "serenity"]),
        "alpha_conviction": get_subset(["intelligence", "strength", "wisdom"]),
        "volatility_defensive": get_subset(["serenity", "dexterity", "wisdom"]),
        "risk_parity": get_subset(["sunshine", "intelligence", "dexterity"]),
        "macro_hedged": get_subset(["midnight", "agility", "serenity"]),
        # Expansion groups for strategies 16-30 (guaranteed pairwise Jaccard < 0.85):
        "fundamental_value": get_subset(["charisma", "dexterity"]),
        "low_beta_defensive": get_subset(["serenity", "constitution"]),
        "residual_alpha": get_subset(["constitution", "agility"]),
        "mean_reversion": get_subset(["dexterity", "agility"]),
        "factor_momentum": get_subset(["strength", "agility"]),
        "high_sharpe_quality": get_subset(["intelligence", "wisdom"]),
        "macro_tail_liquidity": get_subset(["midnight", "strength"]),
        "earnings_quality": get_subset(["charisma", "sunshine"]),
        "sentiment_divergence": get_subset(["midnight", "serenity", "charisma"]),
        "vol_adjusted_alpha": get_subset(["serenity", "strength"]),
        "orthogonal_risk_parity": get_subset(["sunshine", "agility"]),
        "residual_spread": get_subset(["dexterity", "wisdom"]),
        "growth_trend": get_subset(["agility", "charisma"]),
    }
    return groups


def resolve_strategy_config(model_name: str, idx: int) -> tuple[int, str, float]:
    """
    Deterministically maps a model name and index to its orthogonal strategy specification:
    Returns (strat_id, feature_group_key, neutralization_proportion).
    Guarantees zero unhandled states, zero uninitialized variables, and clean testability.
    Phase 0: Explicit model dictionary lookup takes highest priority (exact identity match).
    Phase 1: Explicit keyword matches in model name take secondary priority.
    Phase 2: Fallback to modulo 30 slot routing for generic/unbranded model names.
    """
    name_lower = (model_name or "").lower().strip()

    # 0. Deterministic explicit model dictionary lookup takes absolute priority
    if name_lower in EXPLICIT_MODEL_ROUTING:
        strat_id = EXPLICIT_MODEL_ROUTING[name_lower]
        if strat_id in FLEET_STRATEGY_MAP_60D:
            _, feat_key, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
            return (strat_id, feat_key, neut_prop)

    # 1. Explicit keyword matching for 30-model fleet
    if "supernova" in name_lower:
        return (30, "all_medium", 0.35)
    elif "ender" in name_lower:
        return (29, "all_medium", 0.25)
    elif "caroline" in name_lower or "caro" in name_lower:
        return (28, "growth_trend", 0.30)
    elif "agnes" in name_lower:
        return (27, "residual_spread", 0.45)
    elif "claudia" in name_lower:
        return (26, "orthogonal_risk_parity", 0.35)
    elif "victor" in name_lower:
        return (25, "vol_adjusted_alpha", 0.35)
    elif "waldo" in name_lower:
        return (24, "sentiment_divergence", 0.40)
    elif "tyler" in name_lower:
        return (23, "earnings_quality", 0.35)
    elif "sam" in name_lower:
        return (22, "macro_tail_liquidity", 0.45)
    elif "rowan" in name_lower:
        return (21, "high_sharpe_quality", 0.30)
    elif "ralph" in name_lower:
        return (20, "factor_momentum", 0.35)
    elif "echo" in name_lower:
        return (19, "mean_reversion", 0.30)
    elif "delta" in name_lower:
        return (18, "residual_alpha", 0.50)
    elif "charlie" in name_lower:
        return (17, "low_beta_defensive", 0.40)
    elif "bravo" in name_lower:
        return (16, "fundamental_value", 0.35)
    elif "macro_hedged" in name_lower or "hedged" in name_lower:
        return (15, "macro_hedged", 0.50)
    elif "cyrus" in name_lower:
        return (6, "all_medium", 0.30)
    elif "deep" in name_lower:
        return (14, "all_medium", 0.25)
    elif "sharpe" in name_lower:
        return (13, "risk_parity", 0.35)
    elif "vol" in name_lower:
        return (12, "volatility_defensive", 0.40)
    elif "fund" in name_lower or "jeremy" in name_lower:
        return (2, "fundamental", 0.35)
    elif "alpha" in name_lower:
        return (11, "alpha_conviction", 0.30)
    elif "xerxes" in name_lower:
        return (4, "macro", 0.45)
    elif "macro_tail" in name_lower or "tail" in name_lower:
        return (10, "macro_tail", 0.45)
    elif "val" in name_lower or "cap" in name_lower:
        return (9, "value_capital", 0.35)
    elif "vel" in name_lower or "trend" in name_lower:
        return (8, "trend_velocity", 0.40)
    elif "qual" in name_lower or "def" in name_lower:
        return (7, "quality_defensive", 0.35)
    elif "res" in name_lower:
        return (5, "constitution", 0.50)
    elif "macro" in name_lower:
        return (4, "macro", 0.45)
    elif "mom" in name_lower:
        return (3, "momentum", 0.40)

    # 2. Modulo 30 slot fallback for generic model names
    slot = idx % 30
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
        15: (16, "fundamental_value", 0.35),
        16: (17, "low_beta_defensive", 0.40),
        17: (18, "residual_alpha", 0.50),
        18: (19, "mean_reversion", 0.30),
        19: (20, "factor_momentum", 0.35),
        20: (21, "high_sharpe_quality", 0.30),
        21: (22, "macro_tail_liquidity", 0.45),
        22: (23, "earnings_quality", 0.35),
        23: (24, "sentiment_divergence", 0.40),
        24: (25, "vol_adjusted_alpha", 0.35),
        25: (26, "orthogonal_risk_parity", 0.35),
        26: (27, "residual_spread", 0.45),
        27: (28, "growth_trend", 0.30),
        28: (29, "all_medium", 0.25),
        29: (30, "all_medium", 0.35),
    }
    return slot_map.get(slot, (1, "all_medium", 0.25))


def get_flagship_quintet_raw_prediction(live_df: pd.DataFrame, feature_subset: list = None) -> np.ndarray | None:
    """Compute raw ensemble predictions from 5-target 60-day Flagship models."""
    targets_60d = [
        "target_cyrusd_60", "target_agnes_60", "target_victor_60",
        "target_jeremy_60", "target_xerxes_60"
    ]
    models_60d_paths = [os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl") for t in targets_60d]
    if not all(os.path.exists(p) for p in models_60d_paths):
        return None
    try:
        preds = []
        for p in models_60d_paths:
            m = joblib.load(p)
            feats_to_use = [f for f in m.feature_name_ if f in live_df.columns] if hasattr(m, "feature_name_") else feature_subset
            if not feats_to_use or len(feats_to_use) < 10:
                feats_to_use = [c for c in live_df.columns if c.startswith("feature_")]
            preds.append(rank_01(m.predict(live_df[feats_to_use])))
        return np.mean(preds, axis=0)
    except Exception:
        return None


def generate_tri_ensemble_prediction(
    live_df: pd.DataFrame,
    strat_id: int,
    feature_subset: list,
    neut_proportion: float,
    neutralizer_feats: list,
    allow_mock_fallback: bool = False,
    anchor_weight: float = None,
    quintet_raw: np.ndarray = None,
) -> np.ndarray:
    if anchor_weight is None:
        anchor_weight = FLAGSHIP_ANCHOR_WEIGHT if strat_id > 1 else 0.0

    # 1. 60-Day Multi-Target Quintet (Flagship Strategy 1 Priority)
    if strat_id == 1:
        targets_60d = [
            "target_cyrusd_60", "target_agnes_60", "target_victor_60",
            "target_jeremy_60", "target_xerxes_60"
        ]
        models_60d_paths = [os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl") for t in targets_60d]
        if all(os.path.exists(p) for p in models_60d_paths):
            try:
                preds_60d = []
                for p in models_60d_paths:
                    m = joblib.load(p)
                    preds_60d.append(rank_01(m.predict(live_df[feature_subset])))
                raw_pred = np.mean(preds_60d, axis=0)
                live_copy = live_df.copy()
                live_copy["pred"] = rank_01(raw_pred)
                live_copy = neutralize(live_copy, ["pred"], extra_neutralizers=neutralizer_feats, proportion=neut_proportion)
                return rank_01(live_copy["pred"].values)
            except Exception as e:
                if allow_mock_fallback:
                    raw_pred = np.mean(live_df[feature_subset].values, axis=1)
                    live_copy = live_df.copy()
                    live_copy["pred"] = rank_01(raw_pred)
                    live_copy = neutralize(live_copy, ["pred"], extra_neutralizers=neutralizer_feats, proportion=neut_proportion)
                    return rank_01(live_copy["pred"].values)
                raise e

    # 2. 60-Day Dedicated Orthogonal Fleet Priority (Strategies 2..25)
    ortho_60d_path = os.path.join(ORTHO_60D_DIR, f"lgb_strat_{strat_id}.pkl")
    if os.path.exists(ortho_60d_path):
        try:
            model = joblib.load(ortho_60d_path)
            raw_pred = model.predict(live_df[feature_subset])

            # Apply Flagship Anchored Blending if anchor_weight > 0
            if anchor_weight > 0.0:
                if quintet_raw is None:
                    quintet_raw = get_flagship_quintet_raw_prediction(live_df, feature_subset)
                if quintet_raw is not None and len(quintet_raw) == len(raw_pred):
                    raw_pred = (1.0 - anchor_weight) * rank_01(raw_pred) + anchor_weight * rank_01(quintet_raw)

            live_copy = live_df.copy()
            live_copy["pred"] = rank_01(raw_pred)
            live_copy = neutralize(live_copy, ["pred"], extra_neutralizers=neutralizer_feats, proportion=neut_proportion)
            return rank_01(live_copy["pred"].values)
        except Exception as e:
            if not allow_mock_fallback:
                raise e

    # 3. Existing Tri-Ensemble Fleet (LightGBM + XGBoost + CatBoost)
    lgb_path = os.path.join(TRI_DIR, f"lgb_strat_{strat_id}.pkl")
    xgb_path = os.path.join(TRI_DIR, f"xgb_strat_{strat_id}.pkl")
    cb_path = os.path.join(TRI_DIR, f"cb_strat_{strat_id}.pkl")

    if os.path.exists(lgb_path) and os.path.exists(xgb_path) and os.path.exists(cb_path):
        try:
            m_lgb = joblib.load(lgb_path)
            m_xgb = joblib.load(xgb_path)
            m_cb = joblib.load(cb_path)

            p_lgb = rank_01(m_lgb.predict(live_df[feature_subset]))
            p_xgb = rank_01(m_xgb.predict(live_df[feature_subset]))
            p_cb = rank_01(m_cb.predict(live_df[feature_subset]))

            raw_pred = 0.40 * p_lgb + 0.30 * p_xgb + 0.30 * p_cb
        except Exception as e:
            if allow_mock_fallback:
                raw_pred = np.mean(live_df[feature_subset].values, axis=1)
            else:
                raise e
    else:
        # Fallback to single LightGBM
        single_path = os.path.join(ORTHO_DIR, f"lgb_strat_{strat_id}.pkl")
        if os.path.exists(single_path):
            try:
                model = joblib.load(single_path)
                raw_pred = model.predict(live_df[feature_subset])
            except Exception as e:
                if allow_mock_fallback:
                    raw_pred = np.mean(live_df[feature_subset].values, axis=1)
                else:
                    raise e
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
    current_round = robust_api_call(napi.get_current_round, description="Fetch current round")
    models = robust_api_call(napi.get_models, description="Fetch account models")

    print(f"=== Numerai Fleet Autonomous Tri-Ensemble Submitter: Round {current_round} ===")
    print(f"Connected Account Models ({len(models)}): {models}")

    groups = load_feature_groups()
    os.makedirs(DATA_DIR, exist_ok=True)
    live_path = os.path.join(DATA_DIR, "live.parquet")

    print("\nDownloading active live.parquet dataset...")
    robust_api_call(napi.download_dataset, "v5.0/live.parquet", live_path, description="Download live.parquet")
    live_df = pd.read_parquet(live_path, columns=groups["all_medium"])
    
    # Defensive data audit: verify NaNs and impute with neutral median if ever present
    nan_count = int(live_df.isna().sum().sum())
    if nan_count > 0:
        print(f"[WARN] Detected {nan_count} NaNs in live feature universe. Applying neutral rank imputation (0.5)...")
        live_df = live_df.fillna(0.5)
    else:
        print("Feature universe integrity verified: 0 NaNs across all medium features.")
    
    print(f"Live market universe loaded: {len(live_df)} assets")
    neutralizer_feats = groups.get("fncv3_features", groups["all_medium"])
    print(f"Loaded {len(neutralizer_feats)} canonical FNCv3 risk factors for orthogonal neutralization.")

    # Precompute Flagship Quintet prediction once for efficient anchored blending across the fleet
    print("[INIT] Precomputing Flagship Quintet prediction for anchored blending...")
    quintet_raw = get_flagship_quintet_raw_prediction(live_df, groups["all_medium"])
    if quintet_raw is not None:
        print(f"[OK] Flagship Quintet precomputed for {len(quintet_raw)} assets (Anchor weight: {FLAGSHIP_ANCHOR_WEIGHT*100:.0f}%).")

    failed_models = []
    success_models = []

    # Checkpoint tracking: skip models that already successfully submitted in this round
    checkpoint_file = os.path.join(DATA_DIR, f"completed_submissions_round_{current_round}.json")
    completed_models = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                completed_models = set(json.load(f))
        except Exception:
            completed_models = set()

    for idx, (model_name, model_id) in enumerate(models.items()):
        if model_name in completed_models:
            print(f"\n[SKIP] Model [{idx+1}/{len(models)}]: '{model_name}' already successfully submitted for Round {current_round}.")
            success_models.append(model_name)
            continue

        print(f"\n--- Processing Model [{idx+1}/{len(models)}]: '{model_name}' (ID: {model_id}) ---")
        try:
            preds_path = os.path.join(DATA_DIR, f"predictions_{model_name}_round_{current_round}.csv")

            strat_id, group_key, neut_prop = resolve_strategy_config(model_name, idx)
            print(f"Applying Strategy {strat_id} ('{group_key}', {len(groups[group_key])} features, {neut_prop*100:.0f}% Neutralized)...")
            preds = generate_tri_ensemble_prediction(live_df, strat_id, groups[group_key], neut_prop, neutralizer_feats, quintet_raw=quintet_raw)

            sub_df = pd.DataFrame({"id": live_df.index, "prediction": preds})
            sub_df.to_csv(preds_path, index=False)
            print(f"Saved {len(sub_df)} predictions -> {preds_path}")

            # Robust upload with per-model retries and explicit AWS S3 HTTP PUT 2xx verification
            sub_id = None
            for upload_attempt in range(1, 4):
                try:
                    print(f"Uploading submission to Numerai (Model ID: {model_id}, Attempt {upload_attempt}/3)...")
                    sub_id = safe_upload_predictions(napi, preds_path, model_id=model_id)
                    if sub_id:
                        break
                except Exception as upload_err:
                    print(f"[WARN] Upload attempt {upload_attempt}/3 for '{model_name}' encountered: {upload_err}")
                    if upload_attempt < 3:
                        time.sleep(3.0 * upload_attempt)
                    else:
                        raise upload_err

            if not sub_id:
                raise RuntimeError(f"Upload to Numerai returned empty submission ID for model '{model_name}'")

            # Verify submission confirmation directly in Numerai submission registry
            try:
                subs = napi.submission_ids(model_id=model_id)
                if any(s.get("id") == sub_id for s in subs):
                    print(f"[VERIFIED] Submission ID {sub_id} confirmed in Numerai submission registry for '{model_name}'.")
                else:
                    print(f"[NOTE] Submission ID {sub_id} accepted; registry indexing pending (eventual consistency).")
            except Exception as v_err:
                print(f"[WARN] Non-blocking registry check failed for '{model_name}': {v_err}")

            print(f"[SUCCESS] Successfully submitted '{model_name}' to Round {current_round}! Submission ID: {sub_id}")
            success_models.append(model_name)
            completed_models.add(model_name)
            try:
                with open(checkpoint_file, "w") as f:
                    json.dump(list(completed_models), f)
            except Exception:
                pass
        except Exception as err:
            print(f"[ERROR] Failed processing '{model_name}': {err}")
            failed_models.append((model_name, str(err)))

    print(f"\n[COMPLETE] Fleet submission complete. Succeeded: {len(success_models)}/{len(models)} | Failed: {len(failed_models)}")

    # Autonomous Numerbay Zero-Capital Marketplace Syndication (Post-Submission Hook)
    try:
        from numerbay_publisher import NumerbayPublisher
        nb_pub = NumerbayPublisher()
        if nb_pub.is_configured:
            print("\n[NUMERBAY] Syndicating predictions to Numerbay marketplace...")
            listings = nb_pub.get_listings()
            for item in listings:
                p_name = item.get("name")
                if not p_name:
                    continue
                csv_file = os.path.join(DATA_DIR, f"predictions_{p_name}_round_{current_round}.csv")
                if os.path.exists(csv_file):
                    nb_res = nb_pub.publish_predictions(p_name, csv_file)
                    print(f"  [NUMERBAY] {p_name}: {nb_res.get('status')}")
    except Exception as nb_err:
        print(f"[WARN] Non-blocking Numerbay syndication bypassed: {nb_err}")

    if failed_models:
        print(f"[FAILURES] Failed models: {failed_models}")
        raise RuntimeError(f"Fleet submission completed with {len(failed_models)} failed model(s): {failed_models}")


if __name__ == "__main__":
    main()

