#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SKIP_MARKER="$ROOT/artifacts/manifests/skip_causal_exports_for_converged_submission"
if [[ -f "$SKIP_MARKER" ]]; then
  echo "[skip] Causal Motifs export is outside the compute-converged submission profile"
  exit 0
fi
CORE_PYTHON="${SGG_PYTHON:-python3}"
LEGACY_PYTHON="${KAIHUA_SGG_PYTHON:-python3}"
SOURCE="$ROOT/external/official_repos/Scene-Graph-Benchmark.pytorch"
CONFIG="$SOURCE/configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
PATHS="$ROOT/scripts/legacy_vg_paths_catalog.py"
WEIGHTS="$ROOT/checkpoints/sgg/weights/causal_motifs_sum/vg"
case "${CAUSAL_EFFECT_TYPE:-none}" in
  none|NONE) EFFECT=none; EFFECT_NATIVE=none ;;
  tde|TDE) EFFECT=tde; EFFECT_NATIVE=TDE ;;
  *) echo "CAUSAL_EFFECT_TYPE must be none or TDE" >&2; exit 2 ;;
esac
CACHE="$ROOT/artifacts/prediction_cache/causal_motifs_sum_${EFFECT}_vg"
EVAL_SAMPLES="${CAUSAL_EVAL_SAMPLES:-26446}"
cd "$ROOT"
if (( $# > 0 )); then
  TASKS=("$@")
else
  TASKS=(predcls sgcls sgdet)
fi

if [[ ! -x "$LEGACY_PYTHON" ]]; then
  echo "Missing Kaihua runtime: $LEGACY_PYTHON" >&2
  exit 1
fi

first_file() {
  for path in "$@"; do
    [[ -f "$path" ]] && { printf '%s\n' "$path"; return 0; }
  done
  return 1
}

VG_H5="$(first_file "$ROOT/data/vg/v1.4/VG-SGG.h5" "$ROOT/data/vg/VG-SGG.h5" "$ROOT/data/vg/VG-SGG-with-attri.h5")"
VG_DICT="$(first_file "$ROOT/data/vg/v1.4/VG-SGG-dicts.json" "$ROOT/data/vg/VG-SGG-dicts.json" "$ROOT/data/vg/VG-SGG-dicts-with-attri.json")"
IMAGE_DATA="$(first_file "$ROOT/data/vg/v1.4/image_data.json" "$ROOT/data/vg/image_data.json")"
NATIVE_DICT="$ROOT/external/official_repos/PySGG/datasets/vg/VG-SGG-dicts-with-attri.json"
NATIVE_H5="$ROOT/external/official_repos/PySGG/datasets/vg/VG-SGG-with-attri.h5"

"$CORE_PYTHON" scripts/validate_vg_ontology_alignment.py \
  --canonical_dict "$VG_DICT" \
  --candidate "$NATIVE_DICT" \
  --report "$ROOT/artifacts/manifests/causal_motifs_${EFFECT}_vg_ontology_alignment.json"

for task in "${TASKS[@]}"; do
  case "$task" in
    predcls)
      checkpoint="$WEIGHTS/predcls/model_0030000.pth"
      gt_box=True; gt_label=True ;;
    sgcls)
      checkpoint="$WEIGHTS/sgcls/model_final.pth"
      gt_box=True; gt_label=False ;;
    sgdet)
      checkpoint="$WEIGHTS/sgdet/model_0028000.pth"
      gt_box=False; gt_label=False ;;
    *) echo "Unsupported task: $task" >&2; exit 2 ;;
  esac
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  native="$ROOT/artifacts/native_predictions/causal_motifs_sum_${EFFECT}/$task"
  mkdir -p "$native"
  (
    cd "$SOURCE"
    SGG_PROJECT_ROOT="$ROOT" SGG_LEGACY_EVAL_SAMPLES="$EVAL_SAMPLES" PYTHONPATH="$ROOT/legacy_runtime:$SOURCE" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$LEGACY_PYTHON" tools/relation_test_net.py \
        --config-file "$CONFIG" \
        MODEL.WEIGHT "$checkpoint" \
        MODEL.DEVICE cuda \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX "$gt_box" \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL "$gt_label" \
        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
        MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE "$EFFECT_NATIVE" \
        MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
        MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
        PATHS_CATALOG "$PATHS" \
        TEST.IMS_PER_BATCH 1 \
        TEST.ALLOW_LOAD_FROM_CACHE False \
        TEST.RELATION.SYNC_GATHER False \
        DTYPE float32 \
        OUTPUT_DIR "$native"
  )
  eval_file="$native/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch"
  "$LEGACY_PYTHON" "$ROOT/scripts/convert_legacy_vg_predictions.py" \
    --format kaihua \
    --task "$task" \
    --prediction_file "$eval_file" \
    --checkpoint "$checkpoint" \
    --cache_root "$CACHE" \
    --model_name "causal_motifs_sum_${EFFECT}_official" \
    --family causal_motifs_sum \
    --source_root "$SOURCE" \
    --canonical_dict "$VG_DICT" \
    --native_dict "$NATIVE_DICT" \
    --vg_h5 "$VG_H5" \
    --native_vg_h5 "$NATIVE_H5" \
    --image_data "$IMAGE_DATA" \
    --effect_type "$EFFECT" \
    --expected_images "$EVAL_SAMPLES" \
    --resume
done

"$CORE_PYTHON" scripts/finalize_legacy_vg_cache.py \
  --cache_root "$CACHE" --tasks "${TASKS[@]}" --expected_images "$EVAL_SAMPLES"

if [[ "$EVAL_SAMPLES" == 26446 ]]; then
  "$CORE_PYTHON" scripts/register_legacy_vg_manifest.py \
    --project_root "$ROOT" \
    --model causal_motifs_sum \
    --effect_type "$EFFECT" \
    --cache_root "$CACHE"
else
  echo "[diagnostic-only] subset cache is not registered as a paper manifest"
fi

echo "[complete] Causal Motifs-SUM unified cache=$CACHE"
