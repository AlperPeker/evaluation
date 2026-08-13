#!/bin/bash
set -euo pipefail

REGISTRY="localhost:5000"

# repo names as stored in the registry (2dfs images use library/ prefix)
REPOS=(
    "experiment1-base"
    "experiment1-stargz"
    "library/experiment1-2dfs-stargz"
)

delete_tag() {
    local repo=$1
    local tag=$2
    local digest
    digest=$(curl -sI \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
        -H "Accept: application/vnd.oci.image.manifest.v1+json" \
        "http://${REGISTRY}/v2/${repo}/manifests/${tag}" \
        | grep -i docker-content-digest | tr -d '\r' | awk '{print $2}')
    if [ -n "$digest" ]; then
        curl -sfX DELETE "http://${REGISTRY}/v2/${repo}/manifests/${digest}"
        echo "deleted ${repo}:${tag} (${digest})"
    else
        echo "not found: ${repo}:${tag}"
    fi
}

for repo in "${REPOS[@]}"; do
    tags=$(curl -s "http://${REGISTRY}/v2/${repo}/tags/list" \
        | python3 -c "import sys,json; [print(t) for t in json.load(sys.stdin).get('tags') or []]")
    for tag in $tags; do
        delete_tag "$repo" "$tag"
    done
done
