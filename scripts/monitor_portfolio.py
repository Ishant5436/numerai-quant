#!/usr/bin/env python3
"""
Automated Portfolio Telemetry & Zero-Capital Monitoring Engine.
Monitors:
1. Numerbay.ai Sales, Orders, and Revenue Telemetry.
2. Numerai Tournament Fleet Standings, Round Deadlines, and Daily Score Updates.
3. Hackathon & Grant Pipelines (CTC Fall, Somnia, KeeperHub, Optimism, WEEX).
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

ENV_PATH = os.path.expanduser("~/.env")
load_dotenv(ENV_PATH)
load_dotenv(os.path.join(BASE_DIR, ".env"))


def check_numerbay_telemetry() -> dict:
    """Checks active listings, subscriber orders, and NMR revenue on Numerbay."""
    user = os.getenv("NUMERBAY_USERNAME", "")
    pwd = os.getenv("NUMERBAY_PASSWORD", "")
    if not (user and pwd):
        return {"status": "UNCONFIGURED", "sales": 0, "nmr_earned": 0.0}

    try:
        from numerbay import NumerBay
        nb = NumerBay(username=user, password=pwd)
        listings = nb.get_my_listings() or []
        sales = nb.get_my_sales(active_only=False) or []
        orders = nb.get_my_orders() or []
        
        nmr_earned = sum(float(s.get("price", 0.0)) for s in sales[:100])
        return {
            "status": "ONLINE",
            "active_listings": len(listings),
            "listings": [{"name": l.get("name"), "id": l.get("id"), "sku": l.get("sku")} for l in listings[:10]],
            "total_sales_count": len(sales),
            "total_orders_count": len(orders),
            "nmr_revenue_earned": nmr_earned,
            "latest_sale": sales[0] if sales else None
        }
    except Exception as err:
        return {"status": "ERROR", "error": str(err)}


def check_numerai_tournament_status() -> dict:
    """Checks current active round, deadlines, and 30-model submission confirmation."""
    pub_id = os.getenv("NUMERAI_PUBLIC_ID", "")
    sec_key = os.getenv("NUMERAI_SECRET_KEY", "")
    if not (pub_id and sec_key):
        return {"status": "UNCONFIGURED"}

    try:
        from numerapi import NumerAPI, SignalsAPI
        napi = NumerAPI(pub_id, sec_key)
        sapi = SignalsAPI(pub_id, sec_key)

        curr_round = napi.get_current_round()
        is_open = napi.check_round_open()

        round_info = napi.raw_query("""
            query ($round: Int!) {
              rounds(tournament: 8, number: $round) {
                number
                openTime
                closeTime
                resolveTime
              }
            }
        """, {"round": curr_round}).get("data", {}).get("rounds", [{}])[0]

        classic_models = napi.get_models()
        signals_models = sapi.get_models()

        return {
            "status": "ONLINE",
            "current_round": curr_round,
            "is_round_open": is_open,
            "close_time": round_info.get("closeTime"),
            "resolve_time": round_info.get("resolveTime"),
            "classic_model_count": len(classic_models),
            "signals_model_count": len(signals_models),
            "total_fleet_size": len(classic_models) + len(signals_models)
        }
    except Exception as err:
        return {"status": "ERROR", "error": str(err)}


def check_hackathon_pipeline_countdown() -> dict:
    """Computes exact time remaining and status for all active funding pipelines."""
    now = datetime.now(timezone.utc)
    deadlines = {
        "buidl_ctc_fall_2026": {
            "name": "BUIDL CTC Fall 2026",
            "prize_pool": "$15,000 USD",
            "target": "agent-keeper-mcp (BUIDL #48196)",
            "deadline_utc": "2026-09-14T03:59:00Z",
            "status": "SUBMITTED_FINAL_HOURS"
        },
        "somnia_dreamdex": {
            "name": "Somnia × DreamDEX Hackathon",
            "prize_pool": "$5,000 USD",
            "target": "somnia-dreamdex-agent",
            "deadline_utc": "2026-09-11T23:59:00Z",
            "status": "IN_JUDGING"
        },
        "keeperhub_mcp_bounty": {
            "name": "KeeperHub MCP Integration",
            "prize_pool": "$5,000 + $1,000 PR",
            "target": "PR #2188",
            "deadline_utc": "2026-09-18T23:59:00Z",
            "status": "SUBMITTED"
        },
        "optimism_foundation_grant": {
            "name": "Optimism Foundation Grant",
            "prize_pool": "$15,000 in OP ($5,000 M1)",
            "target": "op-sec-proxy (Thread #10845)",
            "deadline_utc": "2026-09-30T23:59:00Z",
            "status": "COUNCIL_REVIEW"
        },
        "weex_ai_wars_ii": {
            "name": "WEEX AI Wars II",
            "prize_pool": "$200,000 USDT",
            "target": "alpha-engine (BUIDL #48230)",
            "deadline_utc": "2026-10-12T23:59:00Z",
            "status": "ACTIVE_SUBMISSION"
        }
    }

    report = {}
    for key, item in deadlines.items():
        dt = datetime.fromisoformat(item["deadline_utc"].replace("Z", "+00:00"))
        remaining = dt - now
        hours_remaining = round(remaining.total_seconds() / 3600.0, 1)
        report[key] = {
            "name": item["name"],
            "prize_pool": item["prize_pool"],
            "status": item["status"],
            "hours_remaining": hours_remaining if hours_remaining > 0 else 0.0,
            "is_past_deadline": hours_remaining <= 0
        }
    return report


def run_monitoring_pulse() -> dict:
    """Executes full diagnostic sweep and writes report to disk."""
    assert os.path.exists(BASE_DIR), "Base directory missing"
    
    pulse = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "numerbay": check_numerbay_telemetry(),
        "numerai": check_numerai_tournament_status(),
        "hackathons": check_hackathon_pipeline_countdown()
    }

    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    report_file = os.path.join(data_dir, "portfolio_telemetry.json")

    with open(report_file, "w") as f:
        json.dump(pulse, f, indent=2)

    return pulse


if __name__ == "__main__":
    res = run_monitoring_pulse()
    print(json.dumps(res, indent=2))
