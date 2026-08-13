#!/bin/bash
set -euox pipefail

# -------------------------------------------------------------------
# Root check
# -------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "❌ Please run as root"
  exit 1
fi

echo "▶ Tearing down registry node setup..."

# -------------------------------------------------------------------
# Step 1: Stop and remove the 2dfs-registry container
# -------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^2dfs-registry$'; then
  docker stop 2dfs-registry || true
  docker rm   2dfs-registry || true
fi

# -------------------------------------------------------------------
# Step 2: Remove the registry docker volume
# -------------------------------------------------------------------
docker volume rm 2dfs-registry-data 2>/dev/null || true

# -------------------------------------------------------------------
# Step 3: Remove the built image
# -------------------------------------------------------------------
docker rmi 2dfs/registry:latest 2>/dev/null || true

# -------------------------------------------------------------------
# Step 4: Stop and disable Docker
# -------------------------------------------------------------------
if systemctl is-active --quiet docker 2>/dev/null; then
  systemctl stop docker || true
fi
if systemctl is-enabled --quiet docker 2>/dev/null; then
  systemctl disable docker || true
fi
systemctl stop docker.socket 2>/dev/null || true

# -------------------------------------------------------------------
# Step 5: Uninstall Docker packages
# -------------------------------------------------------------------
apt-get remove -y --purge \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin \
  2>/dev/null || true

apt-get autoremove -y 2>/dev/null || true

# -------------------------------------------------------------------
# Step 6: Remove Docker apt repository and GPG key
# -------------------------------------------------------------------
rm -f /etc/apt/sources.list.d/docker.sources
rm -f /etc/apt/keyrings/docker.asc
apt-get update 2>/dev/null || true

# -------------------------------------------------------------------
# Step 7: Remove Docker data directories
# -------------------------------------------------------------------
rm -rf /var/lib/docker
rm -rf /var/lib/containerd
rm -rf /etc/docker
rm -rf /run/docker
rm -rf /run/containerd

# -------------------------------------------------------------------
# Step 8: Remove cloned 2dfs-registry repo
# -------------------------------------------------------------------
# Setup script uses $HOME which when run as root is /root
rm -rf /root/2dfs-registry

echo "✅ Registry node teardown complete"
