#!/bin/bash
set -euox pipefail

# docker
sudo apt update
sudo apt install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable Docker at boot and start it
sudo systemctl enable docker
sudo systemctl start docker

WORKDIR="$HOME/2dfs-registry"
if [ ! -d "$WORKDIR" ]; then
  git clone https://github.com/mitrafsnap/2dfs-registry.git "$WORKDIR"
fi

cd "$WORKDIR"

sudo docker build -t 2dfs/registry:latest .

REGISTRY_DATA="/mydata/2dfs-registry-data"
sudo mkdir -p "$REGISTRY_DATA"

sudo docker run -d \
  --name 2dfs-registry \
  -p 5000:5000 \
  -v "$REGISTRY_DATA":/var/lib/registry \
  2dfs/registry:latest