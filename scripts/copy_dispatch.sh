#!/usr/bin/env bash
# Quick clipboard dispatcher for Numerai community channels.
# Usage: ./scripts/copy_dispatch.sh [discord|forum|reddit|staker]

set -euo pipefail

CHANNEL="${1:-discord}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_FILE="${BASE_DIR}/data/dispatches/dispatch_${CHANNEL}.txt"

if [ ! -f "${TARGET_FILE}" ]; then
    echo "[!] Dispatch file not found: ${TARGET_FILE}"
    echo "    Generating dispatches now..."
    "${BASE_DIR}/venv/bin/python" "${BASE_DIR}/scripts/target_audience_dispatcher.py"
fi

if [ -f "${TARGET_FILE}" ]; then
    cat "${TARGET_FILE}" | pbcopy
    echo "============================================================"
    echo " [OK] Copied '${CHANNEL}' announcement to macOS clipboard!"
    echo "============================================================"
    echo " Destination:"
    case "${CHANNEL}" in
        discord)
            echo "   -> Numerai Discord: #numerbay channel (https://discord.gg/numerai)"
            ;;
        forum)
            echo "   -> Numerai Forum: [Updates] NumerBay thread (https://forum.numer.ai/t/updates-numerbay-the-community-marketplace/3844)"
            ;;
        reddit)
            echo "   -> Reddit: r/numerai (https://www.reddit.com/r/numerai/)"
            ;;
        staker)
            echo "   -> Direct Staker Discussion / DM"
            ;;
    esac
    echo " Simply press Cmd+V to paste."
    echo "============================================================"
else
    echo "[ERROR] Failed to locate dispatch file for channel: ${CHANNEL}"
    exit 1
fi
