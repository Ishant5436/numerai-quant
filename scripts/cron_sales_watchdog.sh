#!/bin/bash
# Autonomous Numerbay Sales & Revenue Watchdog Runner
# Dispatches background polling, atomic lock handling, and native alerts.
set -euo pipefail

LOCK_DIR="/tmp/numerbay_sales_watchdog.lockdir"
PID_FILE="$LOCK_DIR/pid"

# Atomic concurrency guard
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another watchdog process (PID $EXISTING_PID) is active. Exiting."
            exit 0
        fi
    fi
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo "$$" > "$PID_FILE"
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sales_telemetry.log"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
DAEMON_SCRIPT="$SCRIPT_DIR/scripts/sales_telemetry_daemon.py"

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "--------------------------------------------------------" >> "$LOG_FILE"
echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Running Numerbay Sales Telemetry Check..." >> "$LOG_FILE"

if [ -f "$PYTHON_BIN" ] && [ -f "$DAEMON_SCRIPT" ]; then
    "$PYTHON_BIN" "$DAEMON_SCRIPT" >> "$LOG_FILE" 2>&1
    echo "[OK] [$(date '+%Y-%m-%d %H:%M:%S')] Telemetry check finished cleanly." >> "$LOG_FILE"
else
    echo "[ERROR] [$(date '+%Y-%m-%d %H:%M:%S')] Missing python or daemon script!" >> "$LOG_FILE"
    exit 1
fi
