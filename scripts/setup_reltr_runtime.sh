#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${RELTR_ENV_PREFIX:-$PROJECT_ROOT/.runtimes/sgg_reltr}"
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"
export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-180}"
export CONDA_REMOTE_MAX_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-10}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"

retry() {
  local attempt=1
  local maximum=6
  until "$@"; do
    if (( attempt >= maximum )); then
      echo "[ERROR] command failed after $maximum attempts: $*" >&2
      return 1
    fi
    echo "[retry] attempt=$attempt command=$*" >&2
    sleep $((attempt * 15))
    attempt=$((attempt + 1))
  done
}

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "[ERROR] conda not found: $CONDA_BIN" >&2
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  retry "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.10 pip
fi

PYTHON="$ENV_PREFIX/bin/python"
retry "$PYTHON" -m pip install --retries 10 --upgrade "pip<25" wheel setuptools
retry "$PYTHON" -m pip install --retries 10 \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2
retry "$PYTHON" -m pip install --retries 10 \
  "numpy>=1.24,<2" "scipy>=1.9,<1.12" "pillow>=9,<11" \
  "h5py>=3.8,<4" "tqdm>=4.66,<5"

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" - <<'PY'
import sys
import torch
import torchvision
from sgg_core.models.adapters.reltr import create_adapter

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"factory={create_adapter.__module__}:{create_adapter.__name__}")
PY

echo "[READY] RelTR runtime: $ENV_PREFIX"
