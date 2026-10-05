#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SGG_PROJECT_ROOT:-$SCRIPT_ROOT}"
if [[ ! -d "$ROOT/sgg_core" ]]; then
  ROOT="$SCRIPT_ROOT"
fi
export SGG_PROJECT_ROOT="$ROOT"
source "$ROOT/scripts/project_env.sh"
RUN_ID="${RUN_ID:-exp1a_external_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EXP1A_EXTERNAL_OUTPUT:-$SGG_ARTIFACT_DIR/experiment_1a/$RUN_ID}"
DEVICE="${EXP1A_EXTERNAL_DEVICE:-mps}"
MODEL="${EXP1A_EXTERNAL_MODEL:-$SGG_SIGLIP2_B_DIR}"

mkdir -p "$OUTPUT" "$SGG_LOG_DIR"
test -f "$SGG_GQA_TRAIN_JSON" || { echo "[miss] $SGG_GQA_TRAIN_JSON"; exit 2; }
test -f "$SGG_GQA_VAL_JSON" || { echo "[miss] $SGG_GQA_VAL_JSON"; exit 2; }
test -d "$SGG_GQA_IMAGE_ROOT" || { echo "[miss] $SGG_GQA_IMAGE_ROOT"; exit 2; }
test -f "$SGG_VRD_ROOT/json_dataset/annotations_test.json" || { echo "[miss] VRD JSON"; exit 2; }
test -d "$MODEL" || { echo "[miss] SigLIP2 model: $MODEL"; exit 2; }

"$SGG_PYTHON" -m sgg_core.experiments.experiment_1a_external \
  --dataset gqa \
  --train_ann "$SGG_GQA_TRAIN_JSON" \
  --eval_ann "$SGG_GQA_VAL_JSON" \
  --image_root "$SGG_GQA_IMAGE_ROOT" \
  --model_path "$MODEL" \
  --eval_samples "${GQA_SAMPLES:-1000}" \
  --batch_size "${CROP_BATCH_SIZE:-32}" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT/gqa"

"$SGG_PYTHON" -m sgg_core.experiments.experiment_1a_external \
  --dataset vrd \
  --data_root "$SGG_VRD_ROOT" \
  --model_path "$MODEL" \
  --eval_samples "${VRD_SAMPLES:-0}" \
  --batch_size "${CROP_BATCH_SIZE:-32}" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT/vrd"

printf '%s\n' "$OUTPUT" > "$SGG_LOG_DIR/experiment_1a_external_latest.txt"
echo "[complete] Experiment I-A external panel: $OUTPUT"
