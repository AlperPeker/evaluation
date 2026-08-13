#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Fixed config
# ------------------------------------------------------------
REGISTRY="10.10.1.2:5000"
SRC_IMAGE="ubuntu:latest"
DST_IMAGE="${REGISTRY}/ubuntu:stargz"

echo "▶ Pulling source image: ${SRC_IMAGE}"
nerdctl pull "${SRC_IMAGE}"

echo "▶ Converting image using stargz (no-optimize)"
ctr-remote image optimize \
  --no-optimize \
  "docker.io/library/${SRC_IMAGE}" \
  "${DST_IMAGE}"

echo "▶ Pushing image to ${REGISTRY} (insecure)"
nerdctl push --insecure-registry "${DST_IMAGE}"

echo "✅ Done"
echo "   Source : ${SRC_IMAGE}"
echo "   Result : ${DST_IMAGE}"