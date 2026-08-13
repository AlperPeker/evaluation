#!/bin/bash
set -euox pipefail

STARGZ_ROOT="/var/lib/containerd-stargz-grpc"
IMAGE="localhost:5000/library/experiment1-2dfs-stargz:latest--0.0.0.0"

sudo ctr -n default images rm "${IMAGE}" 2>/dev/null || true

# Unmount FUSE mounts while stargz-snapshotter is still running
grep "${STARGZ_ROOT}/snapshotter/snapshots" /proc/mounts | awk '{print $2}' | xargs -r sudo umount

sudo systemctl stop stargz-snapshotter
sudo rm -rf "${STARGZ_ROOT:?}/snapshotter"
sudo rm -rf "${STARGZ_ROOT:?}/stargz"
sudo systemctl start stargz-snapshotter
