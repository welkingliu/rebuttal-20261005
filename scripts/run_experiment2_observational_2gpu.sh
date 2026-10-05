#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"
ROOT="$SGG_PROJECT_ROOT"
PYTHON="$SGG_PYTHON"
RUN_ID="${RUN_ID:-exp2_observational_$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/artifacts/experiment_2/$RUN_ID"
MANIFESTS="$ROOT/checkpoints/sgg/manifests"

mkdir -p "$OUT"
cd "$ROOT"

run_dataset() {
  local dataset="$1"
  shift
  local -a families=("$@")
  "$PYTHON" scripts/run_diagnostic_matrix.py \
    --experiment 2 \
    --analysis_scope observational \
    --project_root "$ROOT" \
    --manifest_dir "$MANIFESTS" \
    --output_dir "$OUT/$dataset" \
    --datasets "$dataset" \
    --families "${families[@]}" \
    --gpus 0 1 \
    --minimum_families 2 \
    --maximum_families 4 \
    --dataset_model_targets "$dataset=2" \
    --train_samples 5000 \
    --eval_samples 1000000000 \
    --resume
}

# Two architectures per dataset are enough for the association analysis. The
# full ten-family standard benchmark remains Experiment IV.
run_dataset vg EGTR KERN SGTR
run_dataset oi EGTR SGTR
run_dataset psg "Neural Motifs" VCTree

printf '%s\n' "$OUT" > "$ROOT/artifacts/logs/experiment_2_observational_latest.txt"
echo "Experiment II observational matrix complete: $OUT"
