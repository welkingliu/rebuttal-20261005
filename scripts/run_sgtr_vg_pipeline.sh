#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_PYTHON="${SGG_PYTHON:-python3}"
SGTR_PYTHON="${SGTR_PYTHON:-python3}"
CACHE_ROOT="${SGTR_CACHE_ROOT:-$PROJECT_ROOT/artifacts/prediction_cache/sgtr_vg}"
OUTPUT_ROOT="${SGTR_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/experiment_4/sgtr_vg_full}"

cd "$PROJECT_ROOT"
test -x "$SGTR_PYTHON"
PYTHONPATH="$PROJECT_ROOT/external/official_repos/SGTR" \
  "$SGTR_PYTHON" -c 'import torch, torchvision, cvpods, h5py; print(f"[runtime-ok] torch={torch.__version__} torchvision={torchvision.__version__}")'

"$SGTR_PYTHON" scripts/export_sgtr_vg_predictions.py \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --eval_samples 26446 \
  --resume \
  --device cuda

"$MAIN_PYTHON" -m sgg_core.tools.validate_prediction_cache \
  --cache_root "$CACHE_ROOT" \
  --dataset vg \
  --data_root "$PROJECT_ROOT/data/vg/v1.4" \
  --eval_samples 26446 \
  --report "$CACHE_ROOT/validation_report.json"

"$MAIN_PYTHON" scripts/build_sgtr_cache_manifest.py \
  --project_root "$PROJECT_ROOT" \
  --cache_root "$CACHE_ROOT"

"$MAIN_PYTHON" -m sgg_core.experiments.experiment_4 \
  --datasets vg \
  --vg_root "$PROJECT_ROOT/data/vg/v1.4" \
  --official_manifest "$PROJECT_ROOT/checkpoints/sgg/manifests/sgtr_vg.json" \
  --steps standard grounding \
  --sgg_tasks sgdet \
  --minimum_model_families 1 \
  --minimum_models_per_dataset 1 \
  --eval_samples 26446 \
  --seen_triplets_manifest "$PROJECT_ROOT/artifacts/manifests/seen_triplets_full.json" \
  --output_dir "$OUTPUT_ROOT" \
  --device cuda

echo "[complete] cache=$CACHE_ROOT"
echo "[complete] manifest=$PROJECT_ROOT/checkpoints/sgg/manifests/sgtr_vg.json"
echo "[complete] results=$OUTPUT_ROOT"
