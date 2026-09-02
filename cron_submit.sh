#!/bin/bash
# Autonomous Weekly Numerai Fleet Submitter & Fail-Safe Watchdog
set -euo pipefail

LOG_FILE="/Users/ishantpanchal/numerai-quant/logs/fleet_submit.log"
PYTHON_BIN="/Users/ishantpanchal/numerai-quant/venv/bin/python"
SCRIPT_PATH="/Users/ishantpanchal/numerai-quant/fleet_submit.py"

echo "========================================================" >> "$LOG_FILE"
echo "🕒 [$(date '+%Y-%m-%d %H:%M:%S')] Launching Numerai Fleet Submission..." >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"

MAX_RETRIES=3
RETRY_DELAY=60
SUCCESS=false

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "▶ Attempt $attempt of $MAX_RETRIES..." >> "$LOG_FILE"
    echo "▶ Generating Signals v3 Supernova predictions..." >> "$LOG_FILE"
    $PYTHON_BIN signals/signals_pipeline.py >> "$LOG_FILE" 2>&1 || true

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
