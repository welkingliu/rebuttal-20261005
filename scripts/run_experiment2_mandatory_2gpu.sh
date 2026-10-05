#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
RUN_ID="${RUN_ID:-exp2_mandatory_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EXP2_OUTPUT:-$ROOT/artifacts/experiment_2/$RUN_ID}"
read -r -a EXP2_GPU_LIST <<< "${EXP2_GPUS:-0 1}"

cd "$ROOT"
"$PYTHON" scripts/check_mandatory_experiment_assets.py \
  --project_root "$ROOT" --require base tritask_caches live_manifests

"$PYTHON" scripts/run_diagnostic_matrix.py \
  --experiment 2 \
  --analysis_scope both \
  --project_root "$ROOT" \
  --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
  --output_dir "$OUTPUT" \
  --datasets vg \
  --families "Neural Motifs" "SGG Transformer" \
  --gpus "${EXP2_GPU_LIST[@]}" \
  --minimum_families 2 \
  --maximum_families 2 \
  --dataset_model_targets vg=2 \
  --full_levels 0 0.1 0.25 0.5 1.0 \
  --perturbation_strategies \
    key_node_mask random_node_mask unrelated_node_mask on_manifold_replacement \
    color_jitter \
  --skip_pair_audit \
  --skip_physical_consistency \
  --train_samples "${EXP2_TRAIN_SAMPLES:-5000}" \
  --eval_samples "${EXP2_EVAL_SAMPLES:-2000}" \
  --resume

printf '%s\n' "$OUTPUT" > "$ROOT/artifacts/logs/experiment_2_mandatory_latest.txt"
echo "[complete] Experiment II mandatory matrix: $OUTPUT/summary.json"
