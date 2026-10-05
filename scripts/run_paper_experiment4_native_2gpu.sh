#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

RUN_ID="${RUN_ID:-experiment4_native_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EXP4_NATIVE_OUTPUT:-$SGG_ARTIFACT_DIR/experiment_4/$RUN_ID}"
MANIFEST_SET="$SGG_MANIFEST_DIR/paper_experiment4_native_sgdet"

"$SGG_PYTHON" "$SCRIPT_DIR/build_paper_experiment4_manifest_set.py" \
  --project_root "$SGG_PROJECT_ROOT" \
  --output_dir "$MANIFEST_SET"

"$SGG_PYTHON" "$SCRIPT_DIR/run_experiment4_matrix.py" \
  --project_root "$SGG_PROJECT_ROOT" \
  --manifest_dir "$MANIFEST_SET" \
  --output_dir "$OUTPUT" \
  --seen_triplets_manifest "$SGG_MANIFEST_DIR/seen_triplets_full.json" \
  --datasets vg oi psg \
  --gpus 0 1 \
  --minimum_model_families 9 \
  --dataset_family_targets vg=5 oi=2 psg=4 \
  --task_contract sgdet_only \
  --steps standard grounding \
  --train_samples 5000 \
  --eval_samples 1000000000 \
  --resume

printf '%s\n' "$OUTPUT" > "$SGG_LOG_DIR/experiment_4_native_latest.txt"
printf '[complete] Experiment IV native SGDet panel: %s\n' "$OUTPUT"
