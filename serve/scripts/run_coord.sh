#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL=${1:-openai-community/gpt2}
HOST=${2:-0.0.0.0}
PORT=${3:-9000}

source /home/navid/miniconda3/etc/profile.d/conda.sh || true
conda activate "${SPLIT_LLM_CONDA_ENV:-splitllm-gpu}" || true

cd "$REPO_DIR"
python serve/distributed_serve.py "$MODEL" 0 --mode coordinator --host "$HOST" --port "$PORT"
