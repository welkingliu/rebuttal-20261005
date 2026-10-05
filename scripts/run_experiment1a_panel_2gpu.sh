#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
read -r -a BACKBONES <<< "${FOUNDATION_BACKBONES:-resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b}"
PREDICTED_MASK_ROOT="${PSG_SAM_MASK_DIR:-}"
PREDICTED_MASK_TRAIN_DIR="${PSG_SAM_TRAIN_DIR:-}"
PREDICTED_MASK_EVAL_DIR="${PSG_SAM_EVAL_DIR:-}"
if [[ -n "$PREDICTED_MASK_ROOT" ]]; then
  PREDICTED_MASK_TRAIN_DIR="${PREDICTED_MASK_TRAIN_DIR:-$PREDICTED_MASK_ROOT/train}"
  PREDICTED_MASK_EVAL_DIR="${PREDICTED_MASK_EVAL_DIR:-$PREDICTED_MASK_ROOT/eval}"
fi
if [[ -n "$PREDICTED_MASK_TRAIN_DIR" || -n "$PREDICTED_MASK_EVAL_DIR" ]]; then
  if [[ -z "$PREDICTED_MASK_TRAIN_DIR" || -z "$PREDICTED_MASK_EVAL_DIR" ]]; then
    echo "Set both PSG_SAM_TRAIN_DIR and PSG_SAM_EVAL_DIR" >&2
    exit 2
  fi
fi
mkdir -p "$SGG_LOG_DIR"

"$SGG_PYTHON" "$SCRIPT_DIR/prepare_foundation_models.py" \
  --project_root "$SGG_PROJECT_ROOT" \
  --models "${BACKBONES[@]}" \
  --check_only

run_backbone() {
  local backbone="$1"
  local gpu="$2"
  local output_dir="$SGG_ARTIFACT_DIR/experiment_1a/$RUN_ID/$backbone"
  local cache_dir="$SGG_DERIVED_ROOT/features/experiment_1a/$backbone"
  local log="$SGG_LOG_DIR/experiment_1a_${backbone}_${RUN_ID}.log"
  local mask_args=()
  if [[ -n "$PREDICTED_MASK_TRAIN_DIR" ]]; then
    mask_args=(
      --predicted_mask_train_dir "$PREDICTED_MASK_TRAIN_DIR"
      --predicted_mask_eval_dir "$PREDICTED_MASK_EVAL_DIR"
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    "$SGG_PYTHON" -m sgg_core.experiments.experiment_1a \
      --psg_train_ann "$SGG_PSG_TRAIN_JSON" \
      --psg_eval_ann "$SGG_PSG_EVAL_JSON" \
      --image_root "$SGG_PSG_IMAGE_ROOT" \
      --panoptic_root "$SGG_PSG_PANOPTIC_ROOT" \
      --backbone "$backbone" \
      --output_dir "$output_dir" \
      --cache_dir "$cache_dir" \
      --train_samples "${EXP1A_TRAIN_SAMPLES:-5000}" \
      --eval_samples "${EXP1A_EVAL_SAMPLES:-1000}" \
      --seeds 17 23 31 \
      --probe_epochs "${EXP1A_PROBE_EPOCHS:-100}" \
      --early_stopping_patience "${EXP1A_EARLY_STOPPING_PATIENCE:-10}" \
      --early_stopping_min_delta "${EXP1A_EARLY_STOPPING_MIN_DELTA:-0.0}" \
      --probe_batch_size "${EXP1A_PROBE_BATCH_SIZE:-512}" \
      --roi_chunk_size "${EXP1A_ROI_CHUNK_SIZE:-128}" \
      --device cuda \
      "${mask_args[@]}" >"$log" 2>&1
}

for ((index=0; index<${#BACKBONES[@]}; index+=2)); do
  run_backbone "${BACKBONES[$index]}" 0 &
  pid0=$!
  pid1=""
  if ((index + 1 < ${#BACKBONES[@]})); then
    run_backbone "${BACKBONES[$((index + 1))]}" 1 &
    pid1=$!
  fi
  wait "$pid0"
  if [[ -n "$pid1" ]]; then
    wait "$pid1"
  fi
done

echo "Experiment I-A object-grounding panel complete: run_id=$RUN_ID"
