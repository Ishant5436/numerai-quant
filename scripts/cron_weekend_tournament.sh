#!/bin/bash
set -euo pipefail

# Directory setup
REPO_DIR="/Users/ishantpanchal/numerai-quant"
LOG_DIR="$REPO_DIR/data/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/weekend_cron_${TIMESTAMP}.log"

echo "=== Numerai Weekend Tournament Cron Started: $(date) ===" >> "$LOG_FILE"

# Load user environment
if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
fi

# Ensure Python venv is used
PYTHON="$REPO_DIR/venv/bin/python"

cd "$REPO_DIR"
"$PYTHON" scripts/autonomous_tournament_worker.py >> "$LOG_FILE" 2>&1

echo "=== Numerai Weekend Tournament Cron Finished: $(date) ===" >> "$LOG_FILE"
