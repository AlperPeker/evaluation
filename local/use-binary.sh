#!/bin/bash
set -euox pipefail

OUT_DIR="/home/garchu/workspace/2dfs/stargz-snapshotter/out"

sudo systemctl stop stargz-snapshotter
sudo cp "${OUT_DIR}/containerd-stargz-grpc" /usr/local/bin/containerd-stargz-grpc
sudo cp "${OUT_DIR}/ctr-remote" /usr/local/bin/ctr-remote
sudo systemctl start stargz-snapshotter
