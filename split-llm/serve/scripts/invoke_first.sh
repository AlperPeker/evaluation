#!/usr/bin/env bash
set -euo pipefail

# Send initial payload to a worker's /invoke endpoint.
# Usage: ./serve/scripts/invoke_first.sh <host> <port> [text] [timeout_seconds]
HOST=${1:-127.0.0.1}
PORT=${2:-8000}
TEXT=${3:-hello}
TIMEOUT=${4:-300}

source /home/navid/miniconda3/etc/profile.d/conda.sh || true
conda activate "${SPLIT_LLM_CONDA_ENV:-splitllm-gpu}" || true

HOST="$HOST" PORT="$PORT" TEXT="$TEXT" TIMEOUT="$TIMEOUT" python - <<'PY'
import os
import json
import requests

url = f"http://{os.environ['HOST']}:{os.environ['PORT']}/invoke"
payload = {"kind": "text", "text": os.environ["TEXT"]}
r = requests.post(url, json=payload, timeout=float(os.environ["TIMEOUT"]))
print(r.status_code)
data = r.json()
print(data.get("result", {}).get("response_text") or data.get("result", {}).get("assistant_message", {}).get("content") or json.dumps(data, indent=2))
PY
