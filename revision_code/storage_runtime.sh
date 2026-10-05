#!/usr/bin/env bash
# Refuse to use the system disk if the experiment volume is unavailable.
set -euo pipefail
DISK=/mnt/EXPERIMENT_DISK
EXPECTED_UUID=401091e4-9c20-4043-a467-91727ed55b56
test "$(findmnt -n -o UUID -T "$DISK")" = "$EXPECTED_UUID" || {
  echo "[STOP] Experiment disk is not mounted with the expected UUID" >&2
  exit 70
}
export SGG_OLD_ROOT="$DISK/kdd_sgg_core_experiments"
export SGG_REBUTTAL_ROOT="$DISK/kdd_sgg_rebuttal_20260929"
export SGG_MODERN_PYTHON=/home/USER/miniconda3/envs/py14/bin/python
export SGG_NATIVE_PYTHON=/home/USER/miniconda3/envs/pysgg_runtime/bin/python
"$SGG_MODERN_PYTHON" -c 'import json; assert json.load(open("/mnt/EXPERIMENT_DISK/_migration/latest/status.json"))["phase"] == "complete_verified"'
mkdir -p "$SGG_REBUTTAL_ROOT"/{logs,status,results,checkpoints,cache,tmp,manifests}
export TMPDIR="$SGG_REBUTTAL_ROOT/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$SGG_REBUTTAL_ROOT/cache/xdg"
export MPLCONFIGDIR="$SGG_REBUTTAL_ROOT/cache/matplotlib"
export TORCH_HOME="$SGG_REBUTTAL_ROOT/cache/torch"
export HF_HOME="$SGG_REBUTTAL_ROOT/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$SGG_REBUTTAL_ROOT/cache/torch_extensions"
export CUDA_CACHE_PATH="$SGG_REBUTTAL_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SGG_OLD_ROOT/legacy_runtime:$SGG_OLD_ROOT/external/official_repos/PySGG:$SGG_OLD_ROOT/scripts:$SGG_OLD_ROOT:$SGG_REBUTTAL_ROOT/code"
cd "$SGG_REBUTTAL_ROOT"
