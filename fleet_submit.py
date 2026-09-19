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
    STRATEGY_ANCHOR_WEIGHTS,
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
    assert max_retries >= 1, f"max_retries must be at least 1, got {max_retries}"
    assert base_delay > 0, f"base_delay must be positive, got {base_delay}"

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
    assert os.path.exists(file_path), f"Prediction file does not exist: {file_path}"
    assert model_id, "model_id must be non-empty"

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
    if isinstance(create, dict) and "errors" in create and create["errors"]:
        raise RuntimeError(f"Numerai GraphQL error: {create['errors']}")
    data = create.get("data") if isinstance(create, dict) else {}
    sub = data.get("create_submission") if isinstance(data, dict) else {}
    submission_id = sub.get("id") if isinstance(sub, dict) else None
    if not submission_id:
        raise RuntimeError(f"Numerai create_submission returned empty submission id: {create}")
    return submission_id


def load_feature_groups() -> dict:
    """Build the named feature-group universe used to route each fleet strategy."""
    with open(FEATURES_JSON) as f:
        meta = json.load(f)
    medium_set = set(meta["feature_sets"]["medium"])
    assert medium_set, "medium feature set must be non-empty"

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
    assert groups["all_medium"], "all_medium feature group must be non-empty"
    assert groups["fncv3_features"], "fncv3_features group must be non-empty"
    return groups


# Ordered (keywords, (strat_id, feature_group_key, neutralization_proportion)) routing table.
# Order is a correctness property: more specific keywords (e.g. "macro_hedged") must be
# checked before the general ones they contain (e.g. "macro"), exactly matching the
# original if/elif priority chain this table replaces.
_KEYWORD_STRATEGY_ROUTING: tuple[tuple[tuple[str, ...], tuple[int, str, float]], ...] = (
    (("supernova",), (30, "all_medium", 0.35)),
    (("ender",), (29, "all_medium", 0.25)),
    (("caroline", "caro"), (28, "growth_trend", 0.30)),
    (("agnes",), (27, "residual_spread", 0.45)),
    (("claudia",), (26, "orthogonal_risk_parity", 0.35)),
    (("victor",), (25, "vol_adjusted_alpha", 0.35)),
    (("waldo",), (24, "sentiment_divergence", 0.40)),
    (("tyler",), (23, "earnings_quality", 0.35)),
    (("sam",), (22, "macro_tail_liquidity", 0.45)),
    (("rowan",), (21, "high_sharpe_quality", 0.25)),
    (("ralph",), (20, "factor_momentum", 0.35)),
    (("echo",), (19, "mean_reversion", 0.25)),
    (("delta",), (18, "residual_alpha", 0.50)),
    (("charlie",), (17, "low_beta_defensive", 0.40)),
    (("bravo",), (16, "fundamental_value", 0.35)),
    (("macro_hedged", "hedged"), (15, "macro_hedged", 0.50)),
    (("cyrus",), (6, "all_medium", 0.30)),
    (("deep",), (14, "all_medium", 0.25)),
    (("sharpe",), (13, "risk_parity", 0.35)),
    (("vol",), (12, "volatility_defensive", 0.40)),
    (("fund", "jeremy"), (2, "fundamental", 0.35)),
    (("alpha",), (11, "alpha_conviction", 0.30)),
    (("xerxes",), (4, "macro", 0.45)),
    (("macro_tail", "tail"), (10, "macro_tail", 0.45)),
    (("val", "cap"), (9, "value_capital", 0.35)),
    (("vel", "trend"), (8, "trend_velocity", 0.40)),
    (("qual", "def"), (7, "quality_defensive", 0.25)),
    (("res",), (5, "constitution", 0.25)),
    (("macro",), (4, "macro", 0.45)),
    (("mom",), (3, "momentum", 0.40)),
)

# Modulo-30 fallback slot routing for generic/unbranded model names.
_MODULO_SLOT_STRATEGY_MAP: dict[int, tuple[int, str, float]] = {
    0: (1, "all_medium", 0.25),
    1: (2, "fundamental", 0.35),
    2: (3, "momentum", 0.40),
    3: (4, "macro", 0.45),
    4: (5, "constitution", 0.25),
    5: (6, "all_medium", 0.30),
    6: (7, "quality_defensive", 0.25),
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
    18: (19, "mean_reversion", 0.25),
    19: (20, "factor_momentum", 0.35),
    20: (21, "high_sharpe_quality", 0.25),
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


def resolve_strategy_config(model_name: str, idx: int) -> tuple[int, str, float]:
    """
    Deterministically maps a model name and index to its orthogonal strategy specification:
    Returns (strat_id, feature_group_key, neutralization_proportion).
    Phase 0: Explicit model dictionary lookup takes highest priority (exact identity match).
    Phase 1: Ordered keyword matches in model name take secondary priority.
    Phase 2: Fallback to modulo 30 slot routing for generic/unbranded model names.
    """
    assert idx >= 0, f"idx must be non-negative, got {idx}"
    name_lower = (model_name or "").lower().strip()

    result = None
    if name_lower in EXPLICIT_MODEL_ROUTING:
        strat_id = EXPLICIT_MODEL_ROUTING[name_lower]
        if strat_id in FLEET_STRATEGY_MAP_60D:
            _, feat_key, neut_prop = FLEET_STRATEGY_MAP_60D[strat_id]
            result = (strat_id, feat_key, neut_prop)

    if result is None:
        for keywords, mapped in _KEYWORD_STRATEGY_ROUTING:
            if any(kw in name_lower for kw in keywords):
                result = mapped
                break

    if result is None:
        slot = idx % 30
        result = _MODULO_SLOT_STRATEGY_MAP.get(slot, (1, "all_medium", 0.25))

    assert 1 <= result[0] <= 30, f"resolved strat_id must be in [1, 30], got {result[0]}"
    assert isinstance(result[1], str) and result[1], "resolved feature group key must be a non-empty string"
    assert 0.0 <= result[2] <= 1.0, f"neutralization proportion must be in [0,1], got {result[2]}"
    return result


def get_flagship_quintet_raw_prediction(live_df: pd.DataFrame, feature_subset: list = None) -> np.ndarray | None:
    """Compute raw ensemble predictions from 5-target 60-day Flagship models."""
    assert not live_df.empty, "live_df must be non-empty"

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
            feat_names = None
            if hasattr(m, "feature_name_"):
                feat_names = m.feature_name_
            elif hasattr(m, "feature_name") and callable(m.feature_name):
                feat_names = m.feature_name()
            elif hasattr(m, "booster_") and hasattr(m.booster_, "feature_name"):
                feat_names = m.booster_.feature_name()
            elif feature_subset:
                feat_names = feature_subset

            feats_to_use = [f for f in feat_names if f in live_df.columns] if feat_names else []
            if not feats_to_use or len(feats_to_use) < 10:
                feats_to_use = [c for c in live_df.columns if c.startswith("feature_")]
            preds.append(rank_01(m.predict(live_df[feats_to_use])))
        quintet = np.mean(preds, axis=0)
        assert len(quintet) == len(live_df), "quintet prediction length must match live_df length"
        return quintet
    except Exception:
        return None


def _finish_prediction(
    live_df: pd.DataFrame, raw_pred: np.ndarray, neut_proportion: float, neutralizer_feats: list
) -> np.ndarray:
    """Shared final stage for every prediction tier: rank, neutralize, re-rank."""
    assert len(raw_pred) == len(live_df), "raw_pred length must match live_df length"
    assert 0.0 <= neut_proportion <= 1.0, f"neut_proportion must be in [0,1], got {neut_proportion}"

    live_copy = live_df.copy()
    live_copy["pred"] = rank_01(raw_pred)
    live_copy = neutralize(live_copy, ["pred"], extra_neutralizers=neutralizer_feats, proportion=neut_proportion)
    result = rank_01(live_copy["pred"].values)
    assert len(result) == len(live_df), "finished prediction length must match live_df length"
    return result


def _tier1_flagship_quintet(
    live_df: pd.DataFrame, strat_id: int, feature_subset: list, allow_mock_fallback: bool
) -> np.ndarray | None:
    """Strategy 1 only: raw 60-day multi-target quintet prediction.
    Returns None (fall through to tier 2) if strat_id != 1 or quintet weights are missing."""
    if strat_id != 1:
        return None

    targets_60d = [
        "target_cyrusd_60", "target_agnes_60", "target_victor_60",
        "target_jeremy_60", "target_xerxes_60"
    ]
    models_60d_paths = [os.path.join(MODEL_60D_DIR, f"lgb_{t}.pkl") for t in targets_60d]
    if not all(os.path.exists(p) for p in models_60d_paths):
        return None

    try:
        preds_60d = [rank_01(joblib.load(p).predict(live_df[feature_subset])) for p in models_60d_paths]
        return np.mean(preds_60d, axis=0)
    except Exception:
        if allow_mock_fallback:
            return np.mean(live_df[feature_subset].values, axis=1)
        raise


def _tier2_dedicated_ortho(
    live_df: pd.DataFrame,
    strat_id: int,
    feature_subset: list,
    anchor_weight: float,
    quintet_raw: np.ndarray | None,
    allow_mock_fallback: bool,
) -> np.ndarray | None:
    """Dedicated 60-day orthogonal specialist model, with optional Flagship Anchored
    Blending. Applies to any strat_id with a matching weights file on disk (in
    practice strategies 2-25). Returns None to fall through to tier 3 -- either no
    weights file exists, or one failed to load and mock fallback is allowed."""
    ortho_60d_path = os.path.join(ORTHO_60D_DIR, f"lgb_strat_{strat_id}.pkl")
    if not os.path.exists(ortho_60d_path):
        return None

    try:
        model = joblib.load(ortho_60d_path)
        raw_pred = model.predict(live_df[feature_subset])

        if anchor_weight > 0.0:
            resolved_quintet = quintet_raw
            if resolved_quintet is None:
                resolved_quintet = get_flagship_quintet_raw_prediction(live_df, None)
            if resolved_quintet is not None and len(resolved_quintet) == len(raw_pred):
                raw_pred = (1.0 - anchor_weight) * rank_01(raw_pred) + anchor_weight * rank_01(resolved_quintet)
        return raw_pred
    except Exception:
        if allow_mock_fallback:
            return None
        raise


def _tier3_tri_ensemble_or_single(
    live_df: pd.DataFrame, strat_id: int, feature_subset: list, allow_mock_fallback: bool
) -> np.ndarray:
    """Legacy tri-ensemble (LightGBM+XGBoost+CatBoost 40/30/30) tier, falling back to
    a single orthogonal model, falling back to a feature-average mock only when
    explicitly allowed. Raises FileNotFoundError in production if no weights exist
    anywhere and mock fallback is disabled -- this tier never returns None."""
    lgb_path = os.path.join(TRI_DIR, f"lgb_strat_{strat_id}.pkl")
    xgb_path = os.path.join(TRI_DIR, f"xgb_strat_{strat_id}.pkl")
    cb_path = os.path.join(TRI_DIR, f"cb_strat_{strat_id}.pkl")

    if os.path.exists(lgb_path) and os.path.exists(xgb_path) and os.path.exists(cb_path):
        try:
            p_lgb = rank_01(joblib.load(lgb_path).predict(live_df[feature_subset]))
            p_xgb = rank_01(joblib.load(xgb_path).predict(live_df[feature_subset]))
            p_cb = rank_01(joblib.load(cb_path).predict(live_df[feature_subset]))
            return 0.40 * p_lgb + 0.30 * p_xgb + 0.30 * p_cb
        except Exception:
            if allow_mock_fallback:
                return np.mean(live_df[feature_subset].values, axis=1)
            raise

    single_path = os.path.join(ORTHO_DIR, f"lgb_strat_{strat_id}.pkl")
    if os.path.exists(single_path):
        try:
            return joblib.load(single_path).predict(live_df[feature_subset])
        except Exception:
            if allow_mock_fallback:
                return np.mean(live_df[feature_subset].values, axis=1)
            raise

    if allow_mock_fallback:
        # Explicitly restricted to synthetic test environments where weights are gitignored
        return np.mean(live_df[feature_subset].values, axis=1)

    raise FileNotFoundError(
        f"Production Error: No model weights found for strategy {strat_id} at {TRI_DIR} or {single_path}. "
        "Refusing to degrade to untrained feature averages during live competition submission."
    )


def generate_tri_ensemble_prediction(
    live_df: pd.DataFrame,
    strat_id: int,
    feature_subset: list,
    neut_proportion: float,
    neutralizer_feats: list,
    allow_mock_fallback: bool = False,
    anchor_weight: float | None = None,
    quintet_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the first applicable prediction tier (Flagship Quintet ->
    dedicated orthogonal specialist -> legacy tri-ensemble/single-model fallback),
    then finish with rank -> neutralize -> re-rank."""
    assert strat_id >= 1, f"strat_id must be positive, got {strat_id}"
    assert len(feature_subset) > 0, f"feature_subset must be non-empty for strategy {strat_id}"
    assert not live_df.empty, "live_df must be non-empty"

    if anchor_weight is None:
        anchor_weight = STRATEGY_ANCHOR_WEIGHTS.get(strat_id, FLAGSHIP_ANCHOR_WEIGHT if strat_id > 1 else 0.0)

    raw_pred = _tier1_flagship_quintet(live_df, strat_id, feature_subset, allow_mock_fallback)
    if raw_pred is None:
        raw_pred = _tier2_dedicated_ortho(
            live_df, strat_id, feature_subset, anchor_weight, quintet_raw, allow_mock_fallback
        )
    if raw_pred is None:
        raw_pred = _tier3_tri_ensemble_or_single(live_df, strat_id, feature_subset, allow_mock_fallback)

    assert raw_pred is not None, f"no prediction tier produced output for strategy {strat_id}"
    assert len(raw_pred) == len(live_df), (
        f"prediction length {len(raw_pred)} != live_df length {len(live_df)} for strategy {strat_id}"
    )
    return _finish_prediction(live_df, raw_pred, neut_proportion, neutralizer_feats)


def _init_session() -> tuple[NumerAPI, int, dict]:
    """Authenticate with Numerai and fetch the current round + registered models."""
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")
    if not (public_id and secret_key):
        if "$" in auth:
            parts = auth.split("$", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                public_id, secret_key = parts[0], parts[1]
    if not (public_id and secret_key):
        raise ValueError("Numerai credentials missing: set NUMERAI_PUBLIC_ID/SECRET_KEY or NUMERAI_MCP_AUTH")

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round = robust_api_call(napi.get_current_round, description="Fetch current round")
    models = robust_api_call(napi.get_models, description="Fetch account models")
    assert models, "No registered models found on this Numerai account"

    print(f"=== Numerai Fleet Autonomous Tri-Ensemble Submitter: Round {current_round} ===")
    print(f"Connected Account Models ({len(models)}): {models}")
    return napi, current_round, models


def _load_live_universe(napi: NumerAPI, groups: dict) -> pd.DataFrame:
    """Download the live dataset and defensively impute any NaNs with neutral rank 0.5."""
    live_path = os.path.join(DATA_DIR, "live.parquet")
    print("\nDownloading active live.parquet dataset...")
    robust_api_call(napi.download_dataset, "v5.0/live.parquet", live_path, description="Download live.parquet")
    live_df = pd.read_parquet(live_path, columns=groups["all_medium"])
    assert not live_df.empty, "Downloaded live universe is empty"

    nan_count = int(live_df.isna().sum().sum())
    if nan_count > 0:
        print(f"[WARN] Detected {nan_count} NaNs in live feature universe. Applying neutral rank imputation (0.5)...")
        live_df = live_df.fillna(0.5)
    else:
        print("Feature universe integrity verified: 0 NaNs across all medium features.")
    assert int(live_df.isna().sum().sum()) == 0, "NaNs remain in live universe after imputation"

    print(f"Live market universe loaded: {len(live_df)} assets")
    return live_df


def _precompute_quintet(live_df: pd.DataFrame, groups: dict) -> np.ndarray | None:
    """Precompute the Flagship Quintet prediction once, shared across the fleet's anchored blending."""
    print("[INIT] Precomputing Flagship Quintet prediction for anchored blending...")
    quintet_raw = get_flagship_quintet_raw_prediction(live_df, groups["all_medium"])
    if quintet_raw is not None:
        assert len(quintet_raw) == len(live_df), "quintet prediction length must match live universe size"
        print(f"[OK] Flagship Quintet precomputed for {len(quintet_raw)} assets (Anchor weight: {FLAGSHIP_ANCHOR_WEIGHT*100:.0f}%).")
    return quintet_raw


def _load_checkpoint(checkpoint_file: str) -> set:
    """Load the set of model names already successfully submitted this round, if any."""
    if not os.path.exists(checkpoint_file):
        return set()
    try:
        import fcntl
        with open(checkpoint_file, "r") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
                return set(json.load(f))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        return set()


def _save_checkpoint(checkpoint_file: str, completed_models: set) -> None:
    """Best-effort persistence of submission progress; failure here must never abort a run."""
    try:
        import tempfile
        import fcntl
        dir_name = os.path.dirname(os.path.abspath(checkpoint_file))
        lock_file = checkpoint_file + ".lock"
        with open(lock_file, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tf:
                    json.dump(list(completed_models), tf)
                    temp_name = tf.name
                os.replace(temp_name, checkpoint_file)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except Exception:
        pass


def _upload_with_retries(napi: NumerAPI, preds_path: str, model_id: str, model_name: str, max_attempts: int = 3) -> str:
    """Upload one model's predictions with per-attempt backoff. Raises on exhausted retries."""
    assert max_attempts >= 1, "max_attempts must be at least 1"

    sub_id = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Uploading submission to Numerai (Model ID: {model_id}, Attempt {attempt}/{max_attempts})...")
            sub_id = safe_upload_predictions(napi, preds_path, model_id=model_id)
            if sub_id:
                break
        except Exception as upload_err:
            print(f"[WARN] Upload attempt {attempt}/{max_attempts} for '{model_name}' encountered: {upload_err}")
            if attempt < max_attempts:
                time.sleep(3.0 * attempt)
            else:
                raise
    if not sub_id:
        raise RuntimeError(f"Upload to Numerai returned empty submission ID for model '{model_name}'")
    return sub_id


def _verify_submission(napi: NumerAPI, model_id: str, sub_id: str, model_name: str) -> None:
    """Best-effort, non-blocking confirmation that the submission is registered."""
    try:
        subs = napi.submission_ids(model_id=model_id)
        if any(s.get("id") == sub_id for s in subs):
            print(f"[VERIFIED] Submission ID {sub_id} confirmed in Numerai submission registry for '{model_name}'.")
        else:
            print(f"[NOTE] Submission ID {sub_id} accepted; registry indexing pending (eventual consistency).")
    except Exception as v_err:
        print(f"[WARN] Non-blocking registry check failed for '{model_name}': {v_err}")


def _submit_one_model(
    napi: NumerAPI,
    model_name: str,
    model_id: str,
    idx: int,
    current_round: int,
    live_df: pd.DataFrame,
    groups: dict,
    neutralizer_feats: list,
    quintet_raw: np.ndarray | None,
) -> str:
    """Generate, save, and upload one model's predictions. Returns the submission ID; raises on any failure."""
    preds_path = os.path.join(DATA_DIR, f"predictions_{model_name}_round_{current_round}.csv")
    strat_id, group_key, neut_prop = resolve_strategy_config(model_name, idx)
    print(f"Applying Strategy {strat_id} ('{group_key}', {len(groups[group_key])} features, {neut_prop*100:.0f}% Neutralized)...")

    preds = generate_tri_ensemble_prediction(
        live_df, strat_id, groups[group_key], neut_prop, neutralizer_feats, quintet_raw=quintet_raw
    )
    assert len(preds) == len(live_df), f"prediction count mismatch for '{model_name}'"

    sub_df = pd.DataFrame({"id": live_df.index, "prediction": preds})
    sub_df.to_csv(preds_path, index=False)
    print(f"Saved {len(sub_df)} predictions -> {preds_path}")

    sub_id = _upload_with_retries(napi, preds_path, model_id, model_name)
    _verify_submission(napi, model_id, sub_id, model_name)
    print(f"[SUCCESS] Successfully submitted '{model_name}' to Round {current_round}! Submission ID: {sub_id}")
    return sub_id


def _syndicate_to_numerbay(current_round: int) -> None:
    """Best-effort, non-blocking publish of fresh predictions to Numerbay subscribers."""
    try:
        from numerbay_publisher import NumerbayPublisher

        nb_pub = NumerbayPublisher()
        if not nb_pub.is_configured:
            return

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


def main():
    napi, current_round, models = _init_session()
    groups = load_feature_groups()
    os.makedirs(DATA_DIR, exist_ok=True)

    live_df = _load_live_universe(napi, groups)
    neutralizer_feats = groups.get("fncv3_features", groups["all_medium"])
    print(f"Loaded {len(neutralizer_feats)} canonical FNCv3 risk factors for orthogonal neutralization.")

    quintet_raw = _precompute_quintet(live_df, groups)

    checkpoint_file = os.path.join(DATA_DIR, f"completed_submissions_round_{current_round}.json")
    completed_models = _load_checkpoint(checkpoint_file)

    failed_models = []
    success_models = []
    for idx, (model_name, model_id) in enumerate(models.items()):
        if model_name in completed_models:
            print(f"\n[SKIP] Model [{idx+1}/{len(models)}]: '{model_name}' already successfully submitted for Round {current_round}.")
            success_models.append(model_name)
            continue

        print(f"\n--- Processing Model [{idx+1}/{len(models)}]: '{model_name}' (ID: {model_id}) ---")
        try:
            _submit_one_model(napi, model_name, model_id, idx, current_round, live_df, groups, neutralizer_feats, quintet_raw)
            success_models.append(model_name)
            completed_models.add(model_name)
            _save_checkpoint(checkpoint_file, completed_models)
        except Exception as err:
            print(f"[ERROR] Failed processing '{model_name}': {err}")
            failed_models.append((model_name, str(err)))

    print(f"\n[COMPLETE] Fleet submission complete. Succeeded: {len(success_models)}/{len(models)} | Failed: {len(failed_models)}")
    _syndicate_to_numerbay(current_round)

    assert len(success_models) + len(failed_models) == len(models), "processed model count must equal total registered models"

    if failed_models:
        print(f"[FAILURES] Failed models: {failed_models}")
        raise RuntimeError(f"Fleet submission completed with {len(failed_models)} failed model(s): {failed_models}")


if __name__ == "__main__":
    main()
