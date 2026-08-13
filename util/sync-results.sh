#!/bin/bash
set -euox pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

sync_one() {
    local src="$1"
    local dst="$2"
    mkdir -p "${dst}"
    rsync -av "${src}" "${dst}"
}

for sub in build rebuild; do
    sync_one \
        "${REPO_ROOT}/benchmark/build_performance/results/${sub}/" \
        "${REPO_ROOT}/../bohdanresults/data/build_performance/${sub}/"
done

for sub in pull refresh; do
    sync_one \
        "${REPO_ROOT}/benchmark/pull_performance/results/${sub}/" \
        "${REPO_ROOT}/../bohdanresults/data/pull_performance/${sub}/"
done
