#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SGG_PROJECT_ROOT:-$SCRIPT_ROOT}"
if [[ ! -d "$ROOT/sgg_core" ]]; then
  ROOT="$SCRIPT_ROOT"
fi
export SGG_PROJECT_ROOT="$ROOT"
source "$ROOT/scripts/project_env.sh"
: "${MANIFEST:?Set MANIFEST to one live VG PySGG manifest}"
DATASET="${DATASET:-gqa}"
FAMILY_KEY="${FAMILY_KEY:-$(basename "$MANIFEST" .json)}"
RUN_ID="${RUN_ID:-exp5_external_$(date +%Y%m%d_%H%M%S)}"
if [[ -n "${MITIGATION_STATE:-}" ]]; then
  STATE_KEY="${STATE_KEY:-$(basename "$(dirname "$MITIGATION_STATE")")_$(basename "$(dirname "$(dirname "$MITIGATION_STATE")")")}" 
else
  STATE_KEY="${STATE_KEY:-base}"
fi
OUTPUT="${EXP5_EXTERNAL_OUTPUT:-$SGG_ARTIFACT_DIR/experiment_5/$RUN_ID/external/$DATASET/$FAMILY_KEY/$STATE_KEY}"
CACHE="${EXP5_EXTERNAL_CACHE:-$SGG_DERIVED_ROOT/predictions/experiment_5_external/$DATASET/$FAMILY_KEY}"
VG_DICT="${VG_DICT:-$SGG_VG_ROOT/VG-SGG-dicts.json}"

ARGS=(
  --manifest "$MANIFEST"
  --dataset "$DATASET"
  --vg_dict "$VG_DICT"
  --cache_dir "$CACHE"
  --output_dir "$OUTPUT"
  --eval_samples "${EXTERNAL_SAMPLES:-0}"
  --device "${DEVICE:-cuda}"
)
if [[ "$DATASET" == "gqa" ]]; then
  ARGS+=(--eval_ann "$SGG_GQA_VAL_JSON" --image_root "$SGG_GQA_IMAGE_ROOT")
else
  ARGS+=(--data_root "$SGG_VRD_ROOT")
fi
if [[ -n "${MITIGATION_STATE:-}" ]]; then
  ARGS+=(--state "$MITIGATION_STATE")
fi
if [[ "${EXPORT_ONLY:-0}" == "1" ]]; then
  ARGS+=(--export_only)
fi

mkdir -p "$OUTPUT" "$CACHE"
"$SGG_PYTHON" -m sgg_core.experiments.experiment_5_external "${ARGS[@]}"
