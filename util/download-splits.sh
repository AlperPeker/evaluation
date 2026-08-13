#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$REPO_ROOT/splits" ]; then
    curl -L https://github.com/2DFS/artifacts-evaluation/releases/download/models/splits.tar.gz -o "$REPO_ROOT/splits.tar.gz"
    tar -xvf "$REPO_ROOT/splits.tar.gz" -C "$REPO_ROOT"
    rm -f "$REPO_ROOT/splits.tar.gz"
    rm -f "$REPO_ROOT/.DS_Store"
else
    echo "splits directory already exists, skipping download."
fi