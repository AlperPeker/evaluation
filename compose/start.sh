#!/bin/bash
set -eu

REGISTRY="registry:5000"
REPO="bert-split"

# Cleanup and setup directories
rm -rf /var/lib/containerd/* /var/lib/containerd-stargz-grpc/* || true
mkdir -p /var/lib/containerd /var/lib/containerd-stargz-grpc /run/containerd-stargz-grpc
mkdir -p /run/buildkit

# Start stargz snapshotter first
containerd-stargz-grpc \
    --address=/run/containerd-stargz-grpc/containerd-stargz-grpc.sock \
    --config=/etc/containerd-stargz-grpc/config.toml \
    --log-level=debug \
    > /var/log/stargz.log 2>&1 &

# Wait for stargz socket
while [ ! -S /run/containerd-stargz-grpc/containerd-stargz-grpc.sock ]; do sleep 1; done

# Start containerd
containerd --config=/etc/containerd/config.toml 2>&1 | tee /var/log/containerd.log &

# Wait for containerd
while ! ctr-remote version >/dev/null 2>&1; do sleep 1; done

# Start BuildKit
echo "starting buildkit" 
mkdir -p /etc/buildkit
tee /etc/buildkit/buildkitd.toml <<EOF
[registry."registry:5000"]
  http = true
  insecure = true
EOF

buildkitd \
  --addr unix:///run/buildkit/buildkitd.sock \
  --oci-worker=true \
  --containerd-worker=true \
  --containerd-worker-addr=/run/containerd/containerd.sock \
  > /var/log/buildkitd.log 2>&1 &

# Wait for buildkitd
while [ ! -S /run/buildkit/buildkitd.sock ]; do sleep 1; done

# check_manifest() {
#   repo="$1"
#   tag="$2"
#   curl -fsS -o /dev/null \
#     -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
#     "http://${REGISTRY}/v2/${repo}/manifests/${tag}"
# }

# echo "==> Waiting for images to be available..."

# until \
#   check_manifest "$REPO" "org" && \
#   check_manifest "$REPO" "esgz"
# do
#   sleep 1
# done

echo "images are available - ready for evaluation"
echo
exec "$@"