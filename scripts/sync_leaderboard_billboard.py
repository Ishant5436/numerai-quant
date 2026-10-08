"""
Numerai Official Leaderboard Billboard Synchronizer.
Transforms the public tournament model page (numer.ai/cypherpole) into an inbound
billboard by setting official bio and link fields to the Numerbay storefront.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def build_billboard_specs(model_name: str, model_id: str) -> Dict[str, Any]:
    """Generates official model bio and link payloads for tournament leaderboard."""
    assert isinstance(model_name, str) and len(model_name) > 0, "Invalid model_name"
    assert isinstance(model_id, str) and len(model_id) > 0, "Invalid model_id"

    bio = (
        "Flagship LightGBM quintet ensemble with linear QR feature neutralization. "
        "Top 1.5% out-of-sample corj60 (+0.0607, 84.2% win rate). "
        "Autonomous prediction syndication on Numerbay. "
        "Use coupon code CYPHER100 for 100% off the trial round."
    )
    link_text = "Numerbay Predictions (Free Trial: CYPHER100)"
    link_url = f"https://numerbay.ai/explore/numerai-predictions?term={model_name}"

    if "hedged" in model_name:
        bio = "Macro-Hedged Factor Specialist. 50% linear QR neutralization against systematic drawdowns. Available on Numerbay."
    elif "vel" in model_name:
        bio = "Trend Velocity Factor Specialist. Momentum alpha breakout capture. Available on Numerbay."

    return {
        "model_name": model_name,
        "model_id": model_id,
        "bio": bio,
        "link_text": link_text,
        "link_url": link_url,
    }


def push_billboard_update(napi: Any, specs: Dict[str, Any]) -> Dict[str, Any]:
    """Executes GraphQL mutations to update model bio and link on Numerai tournament."""
    assert hasattr(napi, "set_bio") and hasattr(napi, "set_link"), "Invalid NumerAPI instance"
    assert isinstance(specs, dict) and "model_id" in specs, "Invalid specs payload"

    m_id = specs["model_id"]
    bio_ok = False
    link_ok = False
    try:
        bio_ok = bool(napi.set_bio(m_id, specs["bio"]))
        link_ok = bool(napi.set_link(m_id, specs["link_text"], specs["link_url"]))
    except Exception as err:
        return {"model": specs["model_name"], "status": "ERROR", "error": str(err)}

    assert isinstance(bio_ok, bool) and isinstance(link_ok, bool), "Status invariants violated"
    return {
        "model": specs["model_name"],
        "status": "SUCCESS" if (bio_ok and link_ok) else "PARTIAL",
        "bio_updated": bio_ok,
        "link_updated": link_ok,
    }


def sync_all_leaderboard_billboards(dry_run: bool = False) -> List[Dict[str, Any]]:
    """Synchronizes leaderboard marketing specs across managed tournament models."""
    assert isinstance(dry_run, bool), "dry_run must be boolean"
    from numerapi import NumerAPI

    napi = NumerAPI()
    models = napi.get_models()
    assert isinstance(models, dict), "get_models must return a dict"

    targets = ["cypherpole", "cypherpole_hedged", "cypherpole_vel"]
    results = []
    for t_name in targets:
        m_id = models.get(t_name)
        if not m_id:
            results.append({"model": t_name, "status": "NOT_FOUND"})
            continue
        specs = build_billboard_specs(t_name, m_id)
        if dry_run:
            results.append({"model": t_name, "status": "DRY_RUN", "specs": specs})
        else:
            results.append(push_billboard_update(napi, specs))

    assert len(results) == len(targets), "All target models must be evaluated"
    return results


if __name__ == "__main__":
    print("Synchronizing Numerai Leaderboard Inbound Billboards...")
    res = sync_all_leaderboard_billboards(dry_run=False)
    for r in res:
        print(f" - {r['model']}: {r['status']}")
