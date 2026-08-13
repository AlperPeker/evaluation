#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -f "${SCRIPT_DIR}"/*.safetensors
rm -f "${SCRIPT_DIR}/2dfs.json"
rm -f "${SCRIPT_DIR}/Dockerfile.stargz"
rm -f "${SCRIPT_DIR}"/Dockerfile.base.*
rm -rf "${SCRIPT_DIR}/.cache"
rm -f "${SCRIPT_DIR}"/*.log
