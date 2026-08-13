#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROM_PIDFILE=/tmp/prometheus-stargz.pid
NE_PIDFILE=/tmp/node-exporter.pid
PORT=9090

# --- node_exporter ---
if ! command -v node_exporter &>/dev/null; then
    echo "node_exporter not found — run setup-prometheus.sh first"
    exit 1
fi

if [[ -f "$NE_PIDFILE" ]] && kill -0 "$(cat "$NE_PIDFILE")" 2>/dev/null; then
    echo "node_exporter already running (pid $(cat "$NE_PIDFILE"))"
else
    node_exporter &>/tmp/node-exporter.log &
    echo $! > "$NE_PIDFILE"
    echo "node_exporter started (pid $!)"
fi

# --- prometheus ---
if ! command -v prometheus &>/dev/null; then
    echo "prometheus not found — run setup-prometheus.sh first"
    exit 1
fi

if [[ -f "$PROM_PIDFILE" ]] && kill -0 "$(cat "$PROM_PIDFILE")" 2>/dev/null; then
    echo "prometheus already running (pid $(cat "$PROM_PIDFILE")) at http://localhost:${PORT}"
    exit 0
fi

prometheus \
    --config.file="${SCRIPT_DIR}/prometheus.yml" \
    --storage.tsdb.path=/tmp/prometheus-stargz-data \
    --web.listen-address="0.0.0.0:${PORT}" \
    --log.level=warn \
    &>/tmp/prometheus-stargz.log &

echo $! > "$PROM_PIDFILE"
echo "prometheus started (pid $!) at http://localhost:${PORT}"
