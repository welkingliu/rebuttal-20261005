#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

EXPERIMENT="${1:?usage: run_focused_audits.sh 2|3 [gpu]}"
GPU="${2:-0}"
if [[ "$EXPERIMENT" != "2" && "$EXPERIMENT" != "3" ]]; then
  echo "experiment must be 2 or 3" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$SGG_ARTIFACT_DIR/experiment_${EXPERIMENT}/$RUN_ID"
FAMILY_VARIABLE="EXP${EXPERIMENT}_FAMILIES"
FAMILY_CSV="${!FAMILY_VARIABLE:-Neural Motifs,VCTree,BGNN,RelTR,EGTR,OvSGTR}"
IFS=',' read -r -a FAMILY_ARRAY <<< "$FAMILY_CSV"

if [[ "$EXPERIMENT" == "2" ]]; then
  DATASETS=(vg oi psg gqa vrd)
  TARGETS=(vg=4 oi=1 psg=1 gqa=1 vrd=1)
else
  DATASETS=(vg gqa oi)
  TARGETS=(vg=4 gqa=1 oi=1)
fi

exec "$SGG_PYTHON" "$SCRIPT_DIR/run_diagnostic_matrix.py" \
  --experiment "$EXPERIMENT" \
  --project_root "$SGG_PROJECT_ROOT" \
  --manifest_dir "$SGG_OFFICIAL_MANIFEST_DIR" \
  --output_dir "$OUTPUT_ROOT" \
  --datasets "${DATASETS[@]}" \
  --families "${FAMILY_ARRAY[@]}" \
  --gpus "$GPU" \
  --train_samples "${TRAIN_SAMPLES:-5000}" \
  --eval_samples "${EVAL_SAMPLES:-2000}" \
  --dataset_model_targets "${TARGETS[@]}"
