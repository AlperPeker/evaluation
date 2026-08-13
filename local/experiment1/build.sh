#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REGISTRY="localhost:5000"
IMG_BASE_NAME="experiment1-base"
IMG_STARGZ="experiment1-stargz:latest"
IMG_2DFS_STARGZ="experiment1-2dfs-stargz:latest"
BASE_IMAGE="ghcr.io/bohdangarchu/python:3.10-esgz"
TS() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

# --- normal base (parallel buildctl) ---
build_base() {
    local i=$1
    local IMAGE="${REGISTRY}/${IMG_BASE_NAME}:${i}"
    local log="build-base-${i}.log"
    local start end
    start=$(date +%s)
    echo "[build ${i}] starting (log: ${log})"
    sudo buildctl build \
        --frontend dockerfile.v0 \
        --opt filename="Dockerfile.base.${i}" \
        --local context=. \
        --local dockerfile=. \
        --output type=image,name="${IMAGE}",push=true,registry.insecure=true \
        >"${log}" 2>&1
    end=$(date +%s)
    echo "[build ${i}] done in $(( end - start ))s"
}

indices=($(ls "${SCRIPT_DIR}"/Dockerfile.base.* | grep -oP '\d+$' | sort -n))

echo "[base] start: $(TS)"
total_start=$(date +%s)

pids=()
for i in "${indices[@]}"; do
    build_base "$i" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$(( failed + 1 ))
done

total_end=$(date +%s)
echo "[base] end: $(TS) ($(( total_end - total_start ))s)"

if [ "$failed" -gt 0 ]; then
    echo "ERROR: ${failed} base build(s) failed"
    exit 1
fi

echo "clearing builder cache..."
bash "${SCRIPT_DIR}/clear-builder-cache.sh"
echo "sleeping 60s before stargz build..."
sleep 60

# --- stargz ---
echo "[stargz] start: $(TS)"
time sudo buildctl build \
    --frontend dockerfile.v0 \
    --opt filename=Dockerfile.stargz \
    --local context=. \
    --local dockerfile=. \
    --output type=image,name="${REGISTRY}/${IMG_STARGZ}",push=true,compression=estargz,oci-mediatypes=true,registry.insecure=true
echo "[stargz] end: $(TS)"

echo "clearing builder cache..."
bash "${SCRIPT_DIR}/clear-builder-cache.sh"
echo "sleeping 60s before 2dfs-stargz build..."
sleep 60

# --- 2dfs-stargz ---
echo "[2dfs-stargz] start: $(TS)"
time ./tdfs build --enable-stargz --force-http "${BASE_IMAGE}" "${REGISTRY}/${IMG_2DFS_STARGZ}"
time ./tdfs image push --force-http "${REGISTRY}/${IMG_2DFS_STARGZ}"
echo "[2dfs-stargz] end: $(TS)"
