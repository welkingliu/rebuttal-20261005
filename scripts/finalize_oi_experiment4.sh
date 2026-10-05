#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_PYTHON="${SGG_PYTHON:-python3}"
EXPECTED_IMAGES=1813
OI_ROOT="$PROJECT_ROOT/data/openimages/open-images-v6"
SEEN="$PROJECT_ROOT/artifacts/manifests/seen_triplets_full.json"
EGTR_CACHE="$PROJECT_ROOT/artifacts/prediction_cache/egtr_oi"
SGTR_CACHE="$PROJECT_ROOT/artifacts/prediction_cache/sgtr_oi"
EGTR_MANIFEST="$PROJECT_ROOT/checkpoints/sgg/manifests/egtr_oi.json"
SGTR_MANIFEST="$PROJECT_ROOT/checkpoints/sgg/manifests/sgtr_oi.json"

cd "$PROJECT_ROOT"

cache_count() {
  local root="$1"
  find "$root/predictions/sgdet" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l
}

while pgrep -f 'export_egtr_vg_predictions.py.*--dataset oi.*prediction_cache/egtr_oi' >/dev/null \
   || pgrep -f 'export_sgtr_vg_predictions.py.*--dataset oi.*prediction_cache/sgtr_oi' >/dev/null; do
  printf '[wait] %s egtr=%s/%s sgtr=%s/%s\n' \
    "$(date -Iseconds)" "$(cache_count "$EGTR_CACHE")" "$EXPECTED_IMAGES" \
    "$(cache_count "$SGTR_CACHE")" "$EXPECTED_IMAGES"
  sleep 60
done

for cache in "$EGTR_CACHE" "$SGTR_CACHE"; do
  count="$(cache_count "$cache")"
  if [[ "$count" -ne "$EXPECTED_IMAGES" ]]; then
    echo "[failed] incomplete cache: $cache ($count/$EXPECTED_IMAGES)" >&2
    exit 1
  fi
done

"$MAIN_PYTHON" -m sgg_core.tools.build_seen_triplets \
  --datasets oi \
  --oi_root "$OI_ROOT" \
  --max_images 1000000000 \
  --output "$SEEN" \
  --merge_existing

for cache in "$EGTR_CACHE" "$SGTR_CACHE"; do
  "$MAIN_PYTHON" -m sgg_core.tools.validate_prediction_cache \
    --cache_root "$cache" \
    --dataset oi \
    --data_root "$OI_ROOT" \
    --eval_samples 1000000000 \
    --report "$cache/validation_report.json"
done

"$MAIN_PYTHON" scripts/build_egtr_cache_manifest.py \
  --project_root "$PROJECT_ROOT" --dataset oi \
  --cache_root "$EGTR_CACHE" --output "$EGTR_MANIFEST"
"$MAIN_PYTHON" scripts/build_sgtr_cache_manifest.py \
  --project_root "$PROJECT_ROOT" --dataset oi \
  --cache_root "$SGTR_CACHE" --output "$SGTR_MANIFEST"

run_audit() {
  local manifest="$1"
  local output="$2"
  local log="$3"
  "$MAIN_PYTHON" -m sgg_core.experiments.experiment_4 \
    --datasets oi \
    --oi_root "$OI_ROOT" \
    --official_manifest "$manifest" \
    --output_dir "$output" \
    --seen_triplets_manifest "$SEEN" \
    --steps standard grounding \
    --sgg_tasks sgdet \
    --train_samples 1 \
    --eval_samples 1000000000 \
    --minimum_model_families 1 \
    --minimum_models_per_dataset 1 \
    --allow_incomplete_audits \
    --device cpu >"$log" 2>&1
}

run_audit "$EGTR_MANIFEST" \
  "$PROJECT_ROOT/artifacts/experiment_4/egtr_oi_full" \
  "$PROJECT_ROOT/artifacts/logs/egtr_oi_audit.log" &
egtr_audit_pid=$!
run_audit "$SGTR_MANIFEST" \
  "$PROJECT_ROOT/artifacts/experiment_4/sgtr_oi_full" \
  "$PROJECT_ROOT/artifacts/logs/sgtr_oi_audit.log" &
sgtr_audit_pid=$!
wait "$egtr_audit_pid"
wait "$sgtr_audit_pid"

echo "[complete] Experiment IV OI panel"
echo "[complete] $PROJECT_ROOT/artifacts/experiment_4/egtr_oi_full/summary.json"
echo "[complete] $PROJECT_ROOT/artifacts/experiment_4/sgtr_oi_full/summary.json"
