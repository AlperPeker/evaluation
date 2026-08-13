#!/bin/bash
set -euox pipefail

VOLUME_NAME="2dfs-registry-data"

# Stop and remove registry container
EXISTING=$(sudo docker ps -a --format '{{.Names}}' | grep tdfs-registry || true)
if [ -n "$EXISTING" ]; then
  sudo docker rm -f "$EXISTING"
fi

# Remove the registry data volume
sudo docker volume rm "$VOLUME_NAME" 2>/dev/null || true

# Prune any anonymous/orphan volumes
sudo docker volume prune -f

echo "Registry stopped and volumes cleared. Run registry-start.sh to start fresh."
