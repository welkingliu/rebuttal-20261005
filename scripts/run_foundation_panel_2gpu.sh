#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
read -r -a BACKBONES <<< "${FOUNDATION_BACKBONES:-resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b}"
if [[ ${#BACKBONES[@]} -eq 0 ]]; then
  echo "FOUNDATION_BACKBONES selected no models" >&2
  exit 2
fi
mkdir -p "$SGG_LOG_DIR"

"$SGG_PYTHON" "$SCRIPT_DIR/prepare_foundation_models.py" \
  --project_root "$SGG_PROJECT_ROOT" \
  --models "${BACKBONES[@]}" \
  --check_only

"$SGG_PYTHON" -m sgg_core.tools.prepare_reviewer_datasets \
  --project_root "$SGG_PROJECT_ROOT" \
  --datasets vg \
  --strict_images \
  --vg_train_samples "${TRAIN_SAMPLES:-5000}" \
  --vg_test_samples "${EVAL_SAMPLES:-2000}"

run_backbone() {
  local backbone="$1"
  local gpu="$2"
  local output_dir="$SGG_ARTIFACT_DIR/experiment_1b/$RUN_ID/$backbone"
  local log="$SGG_LOG_DIR/experiment_1b_${backbone}_${RUN_ID}.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$SGG_PYTHON" -m sgg_core.experiments.experiment_1b \
    --vg_root "$SGG_VG_ROOT" \
    --output_dir "$output_dir" \
    --cache_dir "$SGG_DERIVED_ROOT/features/experiment_1b/$backbone" \
    --feature_mode raw_backbone \
    --backbone "$backbone" \
    --reasoning "${EXP1_REASONING:-gcn}" \
    --depths 0 2 4 8 \
    --seeds 17 23 31 \
    --train_samples "${TRAIN_SAMPLES:-5000}" \
    --test_samples "${EVAL_SAMPLES:-2000}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --roi_chunk_size "${ROI_CHUNK_SIZE:-512}" \
    --eval_pair_chunk_size "${PAIR_CHUNK_SIZE:-512}" \
    --amp \
    --device cuda >"$log" 2>&1
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

echo "Experiment I-B foundation depth panel complete: run_id=$RUN_ID"
