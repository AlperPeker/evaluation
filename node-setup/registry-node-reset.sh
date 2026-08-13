#!/bin/bash
set -euox pipefail

REGISTRY_DATA="/mydata/2dfs-registry-data"

sudo docker stop 2dfs-registry || true
sudo docker rm 2dfs-registry || true
sudo rm -rf "$REGISTRY_DATA"

WORKDIR="$HOME/2dfs-registry"
cd "$WORKDIR"
sudo git pull

sudo docker build -t 2dfs/registry:latest .

sudo mkdir -p "$REGISTRY_DATA"

sudo docker run -d \
  --name 2dfs-registry \
  -p 5000:5000 \
  -v "$REGISTRY_DATA":/var/lib/registry \
  2dfs/registry:latest
