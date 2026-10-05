#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPSG_PYTHON="${OPENPSG_PYTHON:-python3}"
MAIN_PYTHON="${SGG_PYTHON:-python3}"
LOG_ROOT="$ROOT/artifacts/logs/openpsg_native_reference"
mkdir -p "$LOG_ROOT"
cd "$ROOT"

run_queue() {
  local gpu="$1"
  shift
  local model
  for model in "$@"; do
    echo "[native-start] gpu=$gpu model=$model"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      "$OPENPSG_PYTHON" scripts/evaluate_openpsg_native.py \
        --project_root "$ROOT" --model "$model" --resume \
        > "$LOG_ROOT/$model.log" 2>&1
    "$MAIN_PYTHON" scripts/build_openpsg_cache_manifest.py \
      --project_root "$ROOT" --model "$model" \
      --output "$ROOT/checkpoints/sgg/manifests/openpsg_${model}_psg.json"
    echo "[native-complete] gpu=$gpu model=$model"
  done
}

run_queue 0 motifs psgtr &
pid0=$!
run_queue 1 vctree psgformer &
pid1=$!
wait "$pid0"
wait "$pid1"
echo "[complete] all OpenPSG native reference reports passed"
