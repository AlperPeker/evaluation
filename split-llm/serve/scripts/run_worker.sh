#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model_name> <split1[,split2,...]> [bind_host] [port] [coordinator_url] [advertise_host] [device]"
  echo "  split numbers map to field.json rows; each row contains the split JSON plus all files for that split"
  echo "  device defaults to cuda; use cpu or auto to run without a working CUDA driver"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL=$1
SPLITS=$2
HOST=${3:-0.0.0.0}
PORT=${4:-0}
COORD=${5:-}
ADVERTISE_HOST=${6:-}
DEVICE=${7:-cuda}

if [ -z "$ADVERTISE_HOST" ]; then
  ADVERTISE_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [ -z "$ADVERTISE_HOST" ]; then
  ADVERTISE_HOST="$(hostname -f 2>/dev/null || hostname)"
fi

source /home/navid/miniconda3/etc/profile.d/conda.sh || true
conda activate "${SPLIT_LLM_CONDA_ENV:-splitllm-gpu}" || true

cd "$REPO_DIR"
if [ -n "$COORD" ]; then
  python serve/distributed_serve.py "$MODEL" "$SPLITS" --mode worker --host "$HOST" --port "$PORT" --coordinator "$COORD" --advertise-host "$ADVERTISE_HOST" --device "$DEVICE"
else
  python serve/distributed_serve.py "$MODEL" "$SPLITS" --mode worker --host "$HOST" --port "$PORT" --advertise-host "$ADVERTISE_HOST" --device "$DEVICE"
fi
