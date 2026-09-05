#!/bin/bash
# Autonomous Weekly Numerai Fleet Submitter & Fail-Safe Watchdog
set -euo pipefail

LOG_FILE="/Users/ishantpanchal/numerai-quant/logs/fleet_submit.log"
PYTHON_BIN="/Users/ishantpanchal/numerai-quant/venv/bin/python"
SCRIPT_PATH="/Users/ishantpanchal/numerai-quant/fleet_submit.py"
SIGNALS_PATH="/Users/ishantpanchal/numerai-quant/signals/signals_pipeline.py"
LOCK_DIR="/tmp/numerai_fleet_submit.lockdir"
PID_FILE="$LOCK_DIR/pid"

cd /Users/ishantpanchal/numerai-quant

# Atomic concurrency guard: POSIX mkdir is atomic (no TOCTOU window)
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another submission process (PID $EXISTING_PID) is active. Exiting." >> "$LOG_FILE"
            exit 0
        fi
    fi
    # Clean up stale lock if process died unexpectedly
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo "$$" > "$PID_FILE"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

echo "========================================================" >> "$LOG_FILE"
echo "🕒 [$(date '+%Y-%m-%d %H:%M:%S')] Launching Numerai Fleet Submission..." >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"

MAX_RETRIES=3
RETRY_DELAY=60
SUCCESS=false

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "▶ Attempt $attempt of $MAX_RETRIES..." >> "$LOG_FILE"
    echo "▶ Generating Signals v3 Supernova predictions..." >> "$LOG_FILE"
    $PYTHON_BIN "$SIGNALS_PATH" >> "$LOG_FILE" 2>&1 || true

    if $PYTHON_BIN $SCRIPT_PATH >> "$LOG_FILE" 2>&1; then
        echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] Fleet submission completed successfully!" >> "$LOG_FILE"
        SUCCESS=true
        osascript -e 'display notification "Numerai weekly fleet predictions submitted successfully!" with title "Numerai Quant Autopilot" sound name "Glass"' 2>/dev/null || true
        break
    else
        echo "⚠️ [$(date '+%Y-%m-%d %H:%M:%S')] Attempt $attempt failed. Waiting $RETRY_DELAY seconds..." >> "$LOG_FILE"
        sleep $RETRY_DELAY
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "❌ [$(date '+%Y-%m-%d %H:%M:%S')] All $MAX_RETRIES attempts exhausted." >> "$LOG_FILE"
    osascript -e 'display notification "Numerai weekly fleet submission failed! Check logs." with title "Numerai Quant Alert" sound name "Basso"' 2>/dev/null || true
    exit 1
fi
