#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_PYTHON="${SGG_PYTHON:-python3}"
EGTR_PYTHON="${SGG_EGTR_PYTHON:-python3}"
GPU="${EGTR_GPU:-1}"
CACHE_ROOT="${EGTR_CACHE_ROOT:-$PROJECT_ROOT/artifacts/prediction_cache/egtr_vg}"
MANIFEST="${EGTR_MANIFEST:-$PROJECT_ROOT/checkpoints/sgg/manifests/egtr_vg.json}"
OUTPUT_DIR="${EGTR_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/experiment_4/egtr_vg_full}"

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT" "$(dirname "$MANIFEST")" "$OUTPUT_DIR"

"$EGTR_PYTHON" - <<'PY'
import importlib
for name in ("torch", "torchvision", "transformers", "timm", "h5py"):
    module = importlib.import_module(name)
    print(f"[runtime-ok] {name}={getattr(module, '__version__', 'unknown')}")
PY

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT" "$EGTR_PYTHON" \
  scripts/export_egtr_vg_predictions.py \
    --project_root "$PROJECT_ROOT" \
    --output_dir "$CACHE_ROOT" \
    --eval_samples 1000000000 \
    --device cuda \
    --log_every 250 \
    --resume

"$MAIN_PYTHON" -m sgg_core.tools.validate_prediction_cache \
  --cache_root "$CACHE_ROOT" \
  --dataset vg \
  --data_root "$PROJECT_ROOT/data/vg/v1.4" \
  --eval_samples 1000000000

"$MAIN_PYTHON" scripts/build_egtr_cache_manifest.py \
  --project_root "$PROJECT_ROOT" \
  --cache_root "$CACHE_ROOT" \
  --output "$MANIFEST"

"$MAIN_PYTHON" -m sgg_core.experiments.experiment_4 \
  --datasets vg \
  --vg_root "$PROJECT_ROOT/data/vg/v1.4" \
  --official_manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --seen_triplets_manifest "$PROJECT_ROOT/artifacts/manifests/seen_triplets_full.json" \
  --steps standard grounding \
  --train_samples 1 \
  --eval_samples 1000000000 \
  --minimum_model_families 1 \
  --minimum_models_per_dataset 1 \
  --sgg_tasks sgdet \
  --device cpu

echo "[complete] cache=$CACHE_ROOT"
echo "[complete] manifest=$MANIFEST"
echo "[complete] results=$OUTPUT_DIR"
