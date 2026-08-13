#!/bin/bash
set -euox pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY="localhost:5000"
IMAGE="${REGISTRY}/experiment1-2dfs-stargz:latest"
BASE_IMAGE="ghcr.io/bohdangarchu/python:3.12-torch-esgz"

sudo ./tdfs build --enable-stargz --force-http "${BASE_IMAGE}" "${IMAGE}"

sudo ./tdfs image push --force-http "${IMAGE}"
