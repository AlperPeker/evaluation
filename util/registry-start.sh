#!/bin/bash
set -euox pipefail

REGISTRY_SRC="/home/garchu/2dfs-registry"
IMAGE_NAME="tdfs-registry-1"
VOLUME_NAME="2dfs-registry-data"

# Build the registry image
sudo docker build -t "$IMAGE_NAME" "$REGISTRY_SRC"

# Remove existing container if any
EXISTING=$(sudo docker ps -a --format '{{.Names}}' | grep tdfs-registry || true)
if [ -n "$EXISTING" ]; then
  sudo docker rm -f "$EXISTING"
fi

# Run the registry with a named volume
sudo docker run -d \
  --name tdfs-registry-1 \
  -e OTEL_TRACES_EXPORTER=none \
  -v "$VOLUME_NAME":/var/lib/registry \
  -p 5000:5000 \
  "$IMAGE_NAME"

echo "Registry running at localhost:5000"
