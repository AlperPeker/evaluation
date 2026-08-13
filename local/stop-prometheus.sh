#!/usr/bin/env bash
set -euo pipefail

stop_pid() {
    local name=$1
    local pidfile=$2
    if [[ ! -f "$pidfile" ]]; then
        echo "$name: no pidfile found, may not be running"
        return
    fi
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        rm -f "$pidfile"
        echo "$name stopped (pid $pid)"
    else
        echo "$name: process $pid not running, cleaning up pidfile"
        rm -f "$pidfile"
    fi
}

stop_pid "prometheus" /tmp/prometheus-stargz.pid
stop_pid "node_exporter" /tmp/node-exporter.pid
