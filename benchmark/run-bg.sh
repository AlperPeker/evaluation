#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export NTFY_TOPIC="${NTFY_TOPIC:-thesis-benchmark}"

mkdir -p "$(pwd)/logs"
LOG="$(pwd)/logs/run-$(date +%Y%m%d-%H%M%S).log"

nohup python3 -u run.py >"$LOG" 2>&1 </dev/null &
PID=$!
disown

echo "run.py started in background"
echo "  PID: $PID"
echo "  log: $LOG"
echo
echo "Follow live:   tail -f \"$LOG\""
echo "Check alive:   ps -p $PID"
echo "Stop it:       kill $PID"
