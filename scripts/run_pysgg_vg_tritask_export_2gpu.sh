#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYSGG_PYTHON:-python3}"
CORE_PYTHON="${SGG_PYTHON:-python3}"
REPO="$ROOT/external/official_repos/PySGG"
LOG_DIR="$ROOT/artifacts/logs/pysgg_vg_tritask_export"
TASKS=(predcls sgcls sgdet)
FAMILIES=(${PYSGG_EXPORT_FAMILIES:-motifs vctree transformer bgnn tde_motifs})

export PYTHONPATH="$ROOT/legacy_runtime:$REPO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_DIR"

run_family() {
  local gpu="$1"
  local family="$2"
  local display_family
  case "$family" in
    motifs) display_family="Neural Motifs" ;;
    vctree) display_family="VCTree" ;;
    transformer) display_family="SGG Transformer" ;;
    bgnn) display_family="BGNN" ;;
    tde_motifs) display_family="TDE-Motifs" ;;
    *) echo "Unknown family: $family" >&2; return 1 ;;
  esac
  local cache="$ROOT/artifacts/prediction_cache/pysgg_${family}_vg_tritask"
  local task output last checkpoint
  for task in "${TASKS[@]}"; do
    output="$ROOT/checkpoints/sgg/trained/pysgg/$family/$task"
    last="$output/last_checkpoint"
    if [[ ! -s "$last" ]]; then
      echo "[missing] $family/$task last_checkpoint" >&2
      return 1
    fi
    checkpoint="$(<"$last")"
    if [[ ! -f "$checkpoint" ]]; then
      echo "[missing] $checkpoint" >&2
      return 1
    fi
    echo "[export] gpu=$gpu $family/$task"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/scripts/export_pysgg_vg_task.py" \
      --project_root "$ROOT" \
      --config "$ROOT/configs/pysgg_vg_tritask/${family}_${task}.yaml" \
      --checkpoint "$checkpoint" \
      --task "$task" \
      --name "pysgg_${family}_vg_tritask" \
      --family "$display_family" \
      --output_dir "$cache" \
      --resume \
      > "$LOG_DIR/${family}_${task}.log" 2>&1
  done
}

run_queue() {
  local gpu="$1"
  shift
  local family
  for family in "$@"; do
    run_family "$gpu" "$family"
  done
}

gpu0=()
gpu1=()
for index in "${!FAMILIES[@]}"; do
  if (( index % 2 == 0 )); then
    gpu0+=("${FAMILIES[$index]}")
  else
    gpu1+=("${FAMILIES[$index]}")
  fi
done

pids=()
if (( ${#gpu0[@]} > 0 )); then
  run_queue 0 "${gpu0[@]}" &
  pids+=("$!")
fi
if (( ${#gpu1[@]} > 0 )); then
  run_queue 1 "${gpu1[@]}" &
  pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "$pid"
done

for family in "${FAMILIES[@]}"; do
  cache="$ROOT/artifacts/prediction_cache/pysgg_${family}_vg_tritask"
  "$CORE_PYTHON" "$ROOT/scripts/finalize_pysgg_vg_tritask_cache.py" \
    --cache_root "$cache" --expected_images 26446
  "$CORE_PYTHON" "$ROOT/scripts/register_pysgg_vg_tritask_manifest.py" \
    --project_root "$ROOT" --model "$family" --cache_root "$cache"
done

"$CORE_PYTHON" "$ROOT/scripts/check_vg_tritask_assets.py" \
  --project_root "$ROOT" \
  --minimum_families "${PYSGG_EXPORT_MINIMUM_FAMILIES:-${#FAMILIES[@]}}" \
  --expected_images 26446
echo "[complete] requested VG PredCls/SGCls/SGDet caches and manifests: ${FAMILIES[*]}"
