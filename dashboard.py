#!/usr/bin/env python3
"""
Numerai 5-Strategy Quantitative Fleet Live Dashboard
Dynamically reads metrics.json generated from out-of-sample audits.
Runs an ultra-fast, local web dashboard on http://127.0.0.1:8501
"""

import os
import json
from dotenv import load_dotenv
from numerapi import NumerAPI
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route
import uvicorn

load_dotenv(os.path.expanduser("~/.env"))
METRICS_JSON = "/Users/ishantpanchal/numerai-quant/metrics.json"

DEFAULT_STRATEGIES = [
    {
        "id": 1,
        "name": "Core Alpha Ensemble",
        "target": "target (4-Target Blend: Cyrus, Agnes, Victor, Jeremy)",
        "neut": "25% Linear Feature Neutralization",
        "corr": "+0.0212",
        "sharpe": "0.927",
        "ann_sharpe": "3.212",
        "dd": "25.32%",
        "badge": "Flagship",
        "color": "#3b82f6"
    },
    {
        "id": 2,
        "name": "Tail-Risk Volatility Specialist",
        "target": "victor_20 + xerxes_20 + delta_20",
        "neut": "50% Heavy Feature Neutralization",
        "corr": "+0.0213",
        "sharpe": "0.998",
        "ann_sharpe": "3.458",
        "dd": "15.89%",
        "badge": "High MMC",
        "color": "#10b981"
    },
    {
        "id": 3,
        "name": "Quality Momentum Specialist",
        "target": "target_jeremy_20 + target_agnes_20",
        "neut": "35% Balanced Feature Neutralization",
        "corr": "+0.0163",
        "sharpe": "0.783",
        "ann_sharpe": "2.713",
        "dd": "29.49%",
        "badge": "Orthogonal",
        "color": "#8b5cf6"
    },
    {
        "id": 4,
        "name": "Pure Tail-Risk Specialist",
        "target": "target_xerxes_20 (Tail Volatility)",
        "neut": "40% Factor Neutralization",
        "corr": "+0.0214",
        "sharpe": "0.991",
        "ann_sharpe": "3.433",
        "dd": "12.42%",
        "badge": "Min Drawdown",
        "color": "#f59e0b"
    },
    {
        "id": 5,
        "name": "Pure Residual Specialist",
        "target": "target_delta_20 (Uncorrelated Residuals)",
        "neut": "40% Factor Neutralization",
        "corr": "+0.0174",
        "sharpe": "0.883",
        "ann_sharpe": "3.060",
        "dd": "14.68%",
        "badge": "Residual Alpha",
        "color": "#ec4899"
    }
]

DEFAULT_CORR_MATRIX = [
    [1.000, 0.922, 0.913, 0.871, 0.808],
    [0.922, 1.000, 0.793, 0.913, 0.871],
    [0.913, 0.793, 1.000, 0.714, 0.797],
    [0.871, 0.913, 0.714, 1.000, 0.726],
    [0.808, 0.871, 0.797, 0.726, 1.000]
]


def load_dynamic_metrics():
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r") as f:
                data = json.load(f)
                return data.get("strategies", DEFAULT_STRATEGIES), data.get("correlation_matrix", DEFAULT_CORR_MATRIX), data.get("updated_at", "Live")
        except Exception:
            pass
    return DEFAULT_STRATEGIES, DEFAULT_CORR_MATRIX, "Initial Baseline"


def render_html(account_info: dict, current_round: int) -> str:
    strategies, corr_matrix, last_updated = load_dynamic_metrics()

    cards_html = ""
    for s in strategies:
        cards_html += f"""
        <div class="strategy-card" style="border-top: 4px solid {s.get('color', '#3b82f6')};">
            <div class="card-header">
                <span class="badge" style="background-color: {s.get('color', '#3b82f6')}22; color: {s.get('color', '#3b82f6')};">{s.get('badge', 'Strategy')}</span>
                <span class="neut-tag">{s.get('neut', '')}</span>
            </div>
            <h3>Strategy {s.get('id', '')}: {s.get('name', '')}</h3>
            <p class="target-desc"><b>Target:</b> {s.get('target', '')}</p>
            <div class="metric-grid">
                <div class="metric"><span class="m-val" style="color: #10b981;">{s.get('corr', '')}</span><span class="m-lbl">Mean Corr</span></div>
                <div class="metric"><span class="m-val">{s.get('sharpe', '')}</span><span class="m-lbl">Per-Era Sharpe</span></div>
                <div class="metric"><span class="m-val">{s.get('ann_sharpe', '')}</span><span class="m-lbl">Ann. Sharpe</span></div>
                <div class="metric"><span class="m-val" style="color: #f59e0b;">{s.get('dd', '')}</span><span class="m-lbl">Max Drawdown</span></div>
            </div>
        </div>
        """

    matrix_rows = ""
    strat_names = ["Strat 1 (Core)", "Strat 2 (Vol)", "Strat 3 (Quality)", "Strat 4 (Tail)", "Strat 5 (Residual)"]
    for i, row in enumerate(corr_matrix):
        matrix_rows += f"<tr><td class='row-hdr'>{strat_names[i]}</td>"
        for j, val in enumerate(row):
            num_val = float(val)
            bg = "#1e293b" if i == j else ("#334155" if num_val > 0.85 else "#0f172a")
            matrix_rows += f"<td style='background: {bg}; text-align: center;'>{num_val:.3f}</td>"
        matrix_rows += "</tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Numerai 5-Strategy Quant Fleet Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ background: #0b0f19; color: #f8fafc; padding: 32px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #1e293b; padding-bottom: 20px; }}
        .header h1 {{ font-size: 24px; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .status-pill {{ background: #064e3b; color: #34d399; padding: 6px 14px; border-radius: 9999px; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; gap: 8px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 32px; }}
        .strategy-card {{ background: #131d2f; border-radius: 12px; padding: 20px; border: 1px solid #1e293b; }}
        .card-header {{ display: flex; justify-content: space-between; margin-bottom: 12px; }}
        .badge {{ padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
        .neut-tag {{ font-size: 11px; color: #94a3b8; }}
        .strategy-card h3 {{ font-size: 16px; margin-bottom: 6px; }}
        .target-desc {{ font-size: 12px; color: #64748b; margin-bottom: 16px; min-height: 32px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; background: #0b1120; padding: 12px; border-radius: 8px; text-align: center; }}
        .m-val {{ font-size: 15px; font-weight: 700; display: block; }}
        .m-lbl {{ font-size: 10px; color: #64748b; text-transform: uppercase; }}
        .section-title {{ font-size: 18px; font-weight: 600; margin: 28px 0 16px 0; color: #94a3b8; }}
        .matrix-table {{ width: 100%; border-collapse: collapse; background: #131d2f; border-radius: 8px; overflow: hidden; border: 1px solid #1e293b; font-size: 13px; }}
        .matrix-table th, .matrix-table td {{ padding: 12px; border: 1px solid #1e293b; }}
        .matrix-table th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-align: center; }}
        .row-hdr {{ font-weight: 600; color: #cbd5e1; background: #0f172a; text-align: left; }}
        .footer {{ margin-top: 32px; font-size: 12px; color: #475569; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ Numerai 5-Strategy Quant Fleet Dashboard</h1>
            <p style="color: #64748b; font-size: 13px; margin-top: 4px;">v5.0 Dataset • 705 Medium Features • Last Audited: {last_updated}</p>
        </div>
        <div>
            <div class="status-pill">● Round {current_round} Live • Account: {account_info.get('username', 'cypherpole')}</div>
        </div>
    </div>

    <div class="section-title">🏆 5-Strategy Fleet Topology & Dynamic Performance Metrics</div>
    <div class="grid">
        {cards_html}
    </div>

    <div class="section-title">🔗 Pairwise Spearman Correlation Matrix (Orthogonality Audit)</div>
    <table class="matrix-table">
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Strat 1 (Core)</th>
                <th>Strat 2 (Vol)</th>
                <th>Strat 3 (Quality)</th>
                <th>Strat 4 (Tail)</th>
                <th>Strat 5 (Residual)</th>
            </tr>
        </thead>
        <tbody>
            {matrix_rows}
        </tbody>
    </table>

    <div class="footer">
        Automated Quantitative Alpha Pipeline • Apple Silicon Optimized (M5 Pro ARM64) • Model ID: {account_info.get('models', {}).get('cypherpole', 'Active')}
    </div>
</body>
</html>"""


async def homepage(request):
    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    if "$" in auth and not (public_id and secret_key):
        public_id, secret_key = auth.split("$", 1)

    try:
        napi = NumerAPI(public_id=public_id, secret_key=secret_key)
        account = napi.get_account()
        current_round = napi.get_current_round()
        models = napi.get_models()
        account["models"] = models
    except Exception:
        account = {"username": "cypherpole", "models": {"cypherpole": "f85a2798-a8a3-4510-a8d4-0140da05f649"}}
        current_round = 1344

    return HTMLResponse(render_html(account, current_round))


app = Starlette(routes=[Route("/", homepage)])

if __name__ == "__main__":
    print("🚀 Starting Numerai Fleet Dashboard on http://127.0.0.1:8501 ...")
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
