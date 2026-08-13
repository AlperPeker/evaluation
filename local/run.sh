#!/bin/bash
set -euox pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="localhost:5000/library/experiment1-2dfs-stargz:latest--0.0.0.0"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 2DFS + STARGZ: ${IMAGE} ==="

time sudo ctr-remote images rpull --plain-http --use-containerd-labels "${IMAGE}" >/dev/null
time sudo ctr-remote run --rm --snapshotter=stargz "${IMAGE}" run-2dfs-stargz python3 /main.py
