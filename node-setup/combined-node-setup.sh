#!/bin/bash
set -euox pipefail

# -------------------------------------------------------------------
# Combined builder + client node setup.
# Installs everything needed to build images (2dfs, stargz, base)
# AND pull/run them with the stargz snapshotter.
#
# Usage: sudo ./combined-node-setup.sh <REGISTRY_IP> [STARGZ_REPO_URL]
# Example: sudo ./combined-node-setup.sh 10.10.1.2
# Example: sudo ./combined-node-setup.sh 10.10.1.2 https://github.com/mitrafsnap/stargz-snapshotter.git
# -------------------------------------------------------------------

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <REGISTRY_IP> [STARGZ_REPO_URL]"
  exit 1
fi

REGISTRY_NODE="$1"
STARGZ_REPO_URL="${2:-}"

# -------------------------------------------------------------------
# Versions
# -------------------------------------------------------------------
CONTAINERD_VERSION="2.2.1"
RUNC_VERSION="1.3.4"
CNI_VERSION="1.9.0"
NERDCTL_VERSION="2.2.1"
STARGZ_VERSION="0.18.2"
GO_VERSION="1.25.0"
PROMETHEUS_VERSION="3.9.1"

ARCH="amd64"
OS="linux"

# -------------------------------------------------------------------
# Root check
# -------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root"
  exit 1
fi

# -------------------------------------------------------------------
# Registry reachability check
# -------------------------------------------------------------------
if ! curl -sf --connect-timeout 5 "http://${REGISTRY_NODE}:5000/v2/" > /dev/null; then
  echo "Error: registry ${REGISTRY_NODE}:5000 is not accessible"
  exit 1
fi

echo "▶ Installing containerd=${CONTAINERD_VERSION}, runc=${RUNC_VERSION}, cni=${CNI_VERSION}, nerdctl=${NERDCTL_VERSION}, stargz=${STARGZ_VERSION}"

# -------------------------------------------------------------------
# Temp workspace
# -------------------------------------------------------------------
TMP_DIR="$(mktemp -d)"
cd "$TMP_DIR"

# -------------------------------------------------------------------
# Step 1: containerd
# -------------------------------------------------------------------
curl -LO "https://github.com/containerd/containerd/releases/download/v${CONTAINERD_VERSION}/containerd-${CONTAINERD_VERSION}-${OS}-${ARCH}.tar.gz"
tar Cxzvf /usr/local "containerd-${CONTAINERD_VERSION}-${OS}-${ARCH}.tar.gz"

mkdir -p /usr/local/lib/systemd/system
curl -Lo /usr/local/lib/systemd/system/containerd.service \
  "https://raw.githubusercontent.com/containerd/containerd/v${CONTAINERD_VERSION}/containerd.service"

systemctl daemon-reload
systemctl enable --now containerd

# -------------------------------------------------------------------
# Step 2: runc
# -------------------------------------------------------------------
curl -LO "https://github.com/opencontainers/runc/releases/download/v${RUNC_VERSION}/runc.${ARCH}"
install -m 755 "runc.${ARCH}" /usr/local/sbin/runc

# -------------------------------------------------------------------
# Step 3: Python3
# -------------------------------------------------------------------
apt-get update
apt-get install -y python3 python3-pip python3.12-venv

# -------------------------------------------------------------------
# Step 4: FUSE (required for stargz snapshotter)
# -------------------------------------------------------------------
apt-get install -y fuse3
modprobe fuse

# -------------------------------------------------------------------
# Step 5: CNI plugins
# -------------------------------------------------------------------
curl -LO "https://github.com/containernetworking/plugins/releases/download/v${CNI_VERSION}/cni-plugins-${OS}-${ARCH}-v${CNI_VERSION}.tgz"
mkdir -p /opt/cni/bin
tar Cxzvf /opt/cni/bin "cni-plugins-${OS}-${ARCH}-v${CNI_VERSION}.tgz"

# -------------------------------------------------------------------
# Step 6: nerdctl (full) + bundled dependencies
# -------------------------------------------------------------------
curl -LO "https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-full-${NERDCTL_VERSION}-${OS}-${ARCH}.tar.gz"
tar -C /usr/local -xvf "nerdctl-full-${NERDCTL_VERSION}-${OS}-${ARCH}.tar.gz"

# -------------------------------------------------------------------
# Step 7: Go
# -------------------------------------------------------------------
curl -LO "https://go.dev/dl/go${GO_VERSION}.${OS}-${ARCH}.tar.gz"
rm -rf /usr/local/go
tar -C /usr/local -xzf "go${GO_VERSION}.${OS}-${ARCH}.tar.gz"
export PATH="/usr/local/go/bin:$PATH"

# -------------------------------------------------------------------
# Step 8: Stargz-snapshotter binaries (containerd-stargz-grpc, ctr-remote)
# -------------------------------------------------------------------
if [[ -n "$STARGZ_REPO_URL" ]]; then
  STARGZ_REPO_DIR="$(mktemp -d)"
  git clone "$STARGZ_REPO_URL" "$STARGZ_REPO_DIR"
  make -C "$STARGZ_REPO_DIR"
  install -m 755 "$STARGZ_REPO_DIR/out/containerd-stargz-grpc" /usr/local/bin/containerd-stargz-grpc
  install -m 755 "$STARGZ_REPO_DIR/out/ctr-remote" /usr/local/bin/ctr-remote
  rm -rf "$STARGZ_REPO_DIR"
else
  curl -LO "https://github.com/containerd/stargz-snapshotter/releases/download/v${STARGZ_VERSION}/stargz-snapshotter-${STARGZ_VERSION}-${OS}-${ARCH}.tar.gz"
  tar -C /usr/local/bin -xvf \
    "stargz-snapshotter-${STARGZ_VERSION}-${OS}-${ARCH}.tar.gz" \
    containerd-stargz-grpc ctr-remote
fi

# stargz-snapshotter systemd service
curl -Lo /etc/systemd/system/stargz-snapshotter.service \
  https://raw.githubusercontent.com/containerd/stargz-snapshotter/main/script/config/etc/systemd/system/stargz-snapshotter.service

# -------------------------------------------------------------------
# Step 9: 2DFS builder
# -------------------------------------------------------------------
export GOPATH=/usr/local/gopath
export GOCACHE=/usr/local/gocache
export GOTOOLCHAIN=local

BUILDER_DIR="/opt/2dfs-builder"
rm -rf "$BUILDER_DIR"
git clone https://github.com/mitrafsnap/2dfs-builder.git "$BUILDER_DIR"
cd "$BUILDER_DIR"
bash install.sh

# -------------------------------------------------------------------
# Step 10: buildah + pigz
# -------------------------------------------------------------------
apt-get install -y pigz gzip "buildah=1.33.7+ds1-1ubuntu0.24.04.3"

mkdir -p /etc/containers/registries.conf.d
cat > /etc/containers/registries.conf.d/insecure.conf <<EOF
[[registry]]
location = "${REGISTRY_NODE}:5000"
insecure = true
EOF

# -------------------------------------------------------------------
# Step 11: /mydata directories
# -------------------------------------------------------------------
mkdir -p /mydata/tmp /mydata/buildkit
chown -R "$SUDO_USER":"$(id -gn "$SUDO_USER")" /mydata

echo 'export TMPDIR=/mydata/tmp' >> /root/.bashrc
export TMPDIR=/mydata/tmp

mkdir -p /mydata/.2dfs
ln -sf /mydata/.2dfs ~/.2dfs

# -------------------------------------------------------------------
# Step 12: BuildKit daemon
# -------------------------------------------------------------------
cat > /etc/systemd/system/buildkit.service <<'EOF'
[Unit]
Description=BuildKit
Documentation=https://github.com/moby/buildkit
After=containerd.service
Requires=containerd.service

[Service]
Environment=TMPDIR=/mydata/tmp
ExecStart=/usr/local/bin/buildkitd \
  --addr unix:///run/buildkit/buildkitd.sock \
  --containerd-worker=true \
  --containerd-worker-addr=/run/containerd/containerd.sock \
  --root /mydata/buildkit
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

mkdir -p /etc/systemd/system/containerd.service.d
cat > /etc/systemd/system/containerd.service.d/override.conf <<'EOF'
[Service]
Environment=TMPDIR=/mydata/tmp
EOF

mkdir -p /etc/buildkit
tee /etc/buildkit/buildkitd.toml <<EOF
[registry."${REGISTRY_NODE}:5000"]
  http = true
  insecure = true
EOF

mkdir -p /run/buildkit

# -------------------------------------------------------------------
# Step 13: Configure stargz snapshotter
# -------------------------------------------------------------------
mkdir -p /etc/containerd-stargz-grpc

cat > /etc/containerd-stargz-grpc/config.toml <<EOF
noprefetch = true
no_background_fetch = true
disable_verification = true
prefetch_async_size = 0
log_file_access = false
metrics_address = "127.0.0.1:8234"
prefetch_timeout_sec = 120

[resolver]
request_timeout_sec = 120

[[resolver.host."${REGISTRY_NODE}:5000".mirrors]]
host = "${REGISTRY_NODE}:5000"
insecure = true

[fuse]
passthrough = true
EOF

# -------------------------------------------------------------------
# Step 14: Configure containerd with stargz proxy plugin
# -------------------------------------------------------------------
mkdir -p /etc/containerd

cat > /etc/containerd/config.toml <<EOF
version = 2

[debug]
  level = "debug"

[plugins."io.containerd.grpc.v1.cri".containerd]
  snapshotter = "stargz"
  disable_snapshot_annotations = false

[proxy_plugins]
  [proxy_plugins.stargz]
    type = "snapshot"
    address = "/run/containerd-stargz-grpc/containerd-stargz-grpc.sock"

  [proxy_plugins.stargz.exports]
    root = "/var/lib/containerd-stargz-grpc/"

[plugins."io.containerd.grpc.v1.cri".registry.mirrors]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."${REGISTRY_NODE}:5000"]
    endpoint = ["http://${REGISTRY_NODE}:5000"]
EOF

# -------------------------------------------------------------------
# Step 15: Move containerd + stargz storage to /mydata
# -------------------------------------------------------------------
systemctl stop containerd 2>/dev/null || true

if [[ ! -L /var/lib/containerd ]]; then
  mkdir -p /mydata/containerd
  if [[ -d /var/lib/containerd ]]; then
    cp -a /var/lib/containerd/* /mydata/containerd/ 2>/dev/null || true
    rm -rf /var/lib/containerd
  fi
  ln -s /mydata/containerd /var/lib/containerd
fi

mkdir -p /mydata/containerd-stargz-grpc
if [[ ! -L /var/lib/containerd-stargz-grpc ]]; then
  if [[ -d /var/lib/containerd-stargz-grpc ]]; then
    cp -a /var/lib/containerd-stargz-grpc/* /mydata/containerd-stargz-grpc/ 2>/dev/null || true
    rm -rf /var/lib/containerd-stargz-grpc
  fi
  ln -s /mydata/containerd-stargz-grpc /var/lib/containerd-stargz-grpc
fi

# -------------------------------------------------------------------
# Step 16: Start services
# -------------------------------------------------------------------
systemctl daemon-reload
systemctl enable --now stargz-snapshotter
systemctl enable --now containerd
systemctl enable --now buildkit

# -------------------------------------------------------------------
# Step 17: Prometheus (scrapes stargz)
# -------------------------------------------------------------------
curl -sL "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz" \
  | tar -xz -C /usr/local/bin --strip-components=1 \
    "prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" \
    "prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool"

mkdir -p /etc/prometheus /var/lib/prometheus

cat > /etc/prometheus/prometheus.yml <<EOF
global:
  scrape_interval: 1s
  scrape_timeout: 800ms
  external_labels:
    host: $(hostname)

scrape_configs:
  - job_name: 'stargz-snapshotter'
    static_configs:
      - targets: ['127.0.0.1:8234']

EOF

cat > /etc/systemd/system/prometheus.service <<'EOF'
[Unit]
Description=Prometheus
After=network.target

[Service]
ExecStart=/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus
Restart=always

[Install]
WantedBy=multi-user.target
EOF

promtool check config /etc/prometheus/prometheus.yml

systemctl daemon-reload
systemctl enable --now prometheus

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
cd /
rm -rf "$TMP_DIR"

# -------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------
sleep 3
containerd --version
nerdctl --version
buildkitd --version
runc --version
ctr-remote --help | head -n 5
tdfs version
buildah --version

systemctl status stargz-snapshotter --no-pager
systemctl status buildkit --no-pager
systemctl status prometheus --no-pager

echo "--- stargz metrics endpoint ---"
curl -sf http://127.0.0.1:8234/metrics > /dev/null && echo "stargz metrics OK"

echo "--- Prometheus targets ---"
curl -sf http://localhost:9090/api/v1/targets \
  | grep -q '"health":"up"' && echo "Prometheus target health: up"

echo "Combined builder + client node setup complete"
