#!/usr/bin/env bash
# ==============================================================================
# Numerai Signals Daily Supernova Alpha Runner
# Runs Monday-Friday at market close (19:00 IST / 13:30 UTC)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/data/signals_cron.log"
PYTHON_BIN="$QUANT_DIR/venv/bin/python"

mkdir -p "$SCRIPT_DIR/data"

# Atomic concurrency guard: POSIX mkdir is atomic (no TOCTOU window)
LOCK_DIR="/tmp/numerai_daily_signals.lockdir"
PID_FILE="$LOCK_DIR/pid"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another signals runner process (PID $EXISTING_PID) is active. Exiting." >> "$LOG_FILE"
            exit 0
        fi
    fi
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo "$$" > "$PID_FILE"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

echo "=== [$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting Numerai Daily Signals Runner ===" >> "$LOG_FILE"

cd "$QUANT_DIR"

if [ -f "$QUANT_DIR/.env" ]; then
    set -a
    source "$QUANT_DIR/.env"
    set +a
fi

export NUMERAI_PUBLIC_ID="${NUMERAI_PUBLIC_ID:-}"
export NUMERAI_SECRET_KEY="${NUMERAI_SECRET_KEY:-}"
export NUMERAI_MODEL_ID="${NUMERAI_MODEL_ID:-}"

if [ -f "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/signals_pipeline.py" >> "$LOG_FILE" 2>&1
else
    python3 "$SCRIPT_DIR/signals_pipeline.py" >> "$LOG_FILE" 2>&1
fi

echo "=== [$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Finished Numerai Daily Signals Runner ===" >> "$LOG_FILE"
