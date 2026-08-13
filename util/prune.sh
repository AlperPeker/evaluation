#!/bin/bash
set -euox pipefail

STARGZ_ROOT="/var/lib/containerd-stargz-grpc"

# --- stargz snapshotter ---
sudo systemctl stop stargz-snapshotter

# --- docker ---
sudo docker ps -q | xargs -r sudo docker stop 2>/dev/null || true
sudo docker system prune -af --volumes 2>/dev/null || true

# --- nerdctl / containerd ---
sudo nerdctl ps -q | xargs -r sudo nerdctl stop 2>/dev/null || true
sudo nerdctl system prune -af --volumes
sudo ctr content rm $(sudo ctr content ls -q) 2>/dev/null || true
sudo ctr snapshots --snapshotter overlayfs ls | awk 'NR>1 {print $1}' | xargs -r sudo ctr snapshots --snapshotter overlayfs rm 2>/dev/null || true
sudo ctr snapshots --snapshotter stargz ls | awk 'NR>1 {print $1}' | xargs -r sudo ctr snapshots --snapshotter stargz rm 2>/dev/null || true

# --- stargz on-disk cache ---
sudo umount -l "${STARGZ_ROOT}"/snapshotter/snapshots/*/fs 2>/dev/null || true
sudo rm -rf "${STARGZ_ROOT:?}/snapshotter" "${STARGZ_ROOT:?}/stargz"

# --- buildkit cache ---
sudo systemctl stop buildkit 2>/dev/null || true
buildctl prune --all 2>/dev/null || true
sudo ctr -n buildkit content ls | awk 'NR>1 {print $1}' | xargs -r sudo ctr -n buildkit content rm 2>/dev/null || true
sudo rm -rf /var/lib/buildkit/*

# --- 2dfs builder cache ---
for dir in ~/.2dfs /root/.2dfs; do
    sudo rm -rf "${dir:?}/blobs" "${dir:?}/uncompressed-keys" "${dir:?}/index"
done

# --- restart ---
sudo systemctl start buildkit
sudo systemctl start stargz-snapshotter
sudo systemctl restart containerd
