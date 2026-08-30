#!/usr/bin/env python3
"""
Numerai MCP Server
Provides MCP tools for:
- numerai_status: Current round, active deadlines, account models, and submissions.
- numerai_submit: Run automated prediction pipeline and upload submission to Numerai.
"""

import os
import subprocess
from typing import Any, Dict, Optional
from numerapi import NumerAPI
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "numerai-mcp",
    instructions="Numerai quant tournament server for model tracking, round monitoring, prediction generation, and automated submissions."
)


def get_napi() -> NumerAPI:
    auth = os.environ.get("NUMERAI_MCP_AUTH", "")
    pub = os.environ.get("NUMERAI_PUBLIC_ID", "")
    sec = os.environ.get("NUMERAI_SECRET_KEY", "")

    if "$" in auth and not (pub and sec):
        pub, sec = auth.split("$", 1)

    return NumerAPI(public_id=pub or None, secret_key=sec or None)


@mcp.tool()
def numerai_status() -> Dict[str, Any]:
    """
    Get the current active tournament round number, submission deadline status, and connected models.
    """
    try:
        napi = get_napi()
        round_num = napi.get_current_round()
        models = {}
        authenticated = False
        try:
            models = napi.get_models()
            authenticated = bool(models)
        except Exception:
            pass

        return {
            "success": True,
            "current_round": round_num,
            "authenticated": authenticated,
            "models": models
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def numerai_submit(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Run automated prediction pipeline on active live data and upload submission to Numerai.

    Args:
        model_name: Optional target model name (defaults to active account model).
    """
    try:
        cmd = [
            "/Users/ishantpanchal/numerai-quant/venv/bin/python",
            "/Users/ishantpanchal/numerai-quant/fleet_submit.py"
        ]
        env = os.environ.copy()
        if model_name:
            env["NUMERAI_MODEL_NAME"] = model_name

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
