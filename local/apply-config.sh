#!/bin/bash
set -euox pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo systemctl stop stargz-snapshotter

echo "--- applying stargz-config.toml ---"
cat "${SCRIPT_DIR}/stargz-config.toml"
sudo cp "${SCRIPT_DIR}/stargz-config.toml" /etc/containerd-stargz-grpc/config.toml

sudo systemctl start stargz-snapshotter
