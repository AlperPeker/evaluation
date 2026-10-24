#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model_name> <split1> [split2 ...] [--text '...']"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL=$1
shift

source /home/navid/miniconda3/etc/profile.d/conda.sh || true
conda activate "${SPLIT_LLM_CONDA_ENV:-splitllm-gpu}" || true

cd "$REPO_DIR"
python serve/distributed_serve.py "$MODEL" "$@"
