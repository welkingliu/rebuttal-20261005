#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
RUN_ID="${RUN_ID:-exp4_converged_depth_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$ROOT/artifacts/experiment_4/$RUN_ID"
MANIFEST_DIR="$ROOT/artifacts/manifests/experiment4_converged_depth"

cd "$ROOT"
"$PYTHON" scripts/check_vg_tritask_assets.py \
  --project_root "$ROOT" --minimum_families 2 --expected_images 26446

mkdir -p "$MANIFEST_DIR"
rm -f "$MANIFEST_DIR"/*.json
for model in motifs transformer; do
  cp "$ROOT/checkpoints/sgg/manifests/pysgg_${model}_vg_tritask.json" \
    "$MANIFEST_DIR/"
done

"$PYTHON" scripts/run_experiment4_matrix.py \
  --project_root "$ROOT" \
  --manifest_dir "$MANIFEST_DIR" \
  --output_dir "$OUTPUT" \
  --seen_triplets_manifest "$ROOT/artifacts/manifests/seen_triplets_full.json" \
  --datasets vg \
  --gpus 0 1 \
  --minimum_model_families 2 \
  --dataset_family_targets vg=2 \
  --task_contract tritask_depth \
  --steps standard grounding \
  --resume

echo "[complete] converged two-family VG tri-task depth panel: $OUTPUT/summary.json"
