#!/bin/bash
set -euox pipefail

REPO_URL="https://github.com/mitrafsnap/stargz-snapshotter.git"
REPO_DIR="/tmp/stargz-snapshotter"

BIN_DIR="/usr/local/bin"
SOCKET_DIR="/run/containerd-stargz-grpc"

GRPC_BIN="containerd-stargz-grpc"
CTR_REMOTE_BIN="ctr-remote"

SERVICE="stargz-snapshotter.service"

if [[ $EUID -ne 0 ]]; then
  echo "run as root"
  exit 1
fi

rm -rf "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

make

systemctl stop containerd || true
systemctl stop "$SERVICE" || true

install -m 755 "./out/${GRPC_BIN}" "${BIN_DIR}/${GRPC_BIN}"
install -m 755 "./out/${CTR_REMOTE_BIN}" "${BIN_DIR}/${CTR_REMOTE_BIN}"

rm -f "${SOCKET_DIR}/${GRPC_BIN}.sock"

systemctl daemon-reload
systemctl start "$SERVICE"
systemctl start containerd

ctr-remote --help | head -n 3
systemctl status "$SERVICE" --no-pager

echo "snapshotter implementation replaced"