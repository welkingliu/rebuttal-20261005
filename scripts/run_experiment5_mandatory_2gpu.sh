#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
RUN_ID="${RUN_ID:-exp5_mandatory_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EXP5_OUTPUT:-$ROOT/artifacts/experiment_5/$RUN_ID}"
GATE_OUTPUT="${EXP5_GATE_OUTPUT:-$OUTPUT/gate}"
GATE_REPORT="$GATE_OUTPUT/gate_report.json"
read -r -a MITIGATION_GPU_LIST <<< "${MITIGATION_GPUS:-0 1}"
WAIT_GPU_ARGS=()
if [[ -n "${MITIGATION_WAIT_GPU:-}" ]]; then
  WAIT_GPU_ARGS+=(--wait_gpu "$MITIGATION_WAIT_GPU")
fi

cd "$ROOT"
"$PYTHON" scripts/check_mandatory_experiment_assets.py \
  --project_root "$ROOT" --require base tritask_caches live_manifests

"$PYTHON" scripts/run_experiment5_gate.py \
  --project_root "$ROOT" \
  --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
  --output_dir "$GATE_OUTPUT" \
  --family "SGG Transformer" \
  --gpu "${MITIGATION_GATE_GPU:-1}" \
  --seed 17 \
  --epochs "${MITIGATION_GATE_EPOCHS:-3}" \
  --minimum_epochs "${MITIGATION_GATE_MINIMUM_EPOCHS:-2}" \
  --early_stopping_patience 1 \
  --train_samples "${MITIGATION_GATE_TRAIN_SAMPLES:-1000}" \
  --eval_samples "${MITIGATION_GATE_EVAL_SAMPLES:-500}" \
  --minimum_validation_objects "${MITIGATION_GATE_MINIMUM_OBJECTS:-500}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"

"$PYTHON" scripts/run_experiment5_matrix.py \
  --project_root "$ROOT" \
  --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
  --output_dir "$OUTPUT" \
  --gate_report "$GATE_REPORT" \
  --classic_family "Neural Motifs" \
  --transformer_family "SGG Transformer" \
  --dataset vg \
  --seeds 17 23 31 \
  --training_modes supervised_control grounding \
  --gpus "${MITIGATION_GPU_LIST[@]}" \
  "${WAIT_GPU_ARGS[@]}" \
  --epochs "${MITIGATION_EPOCHS:-5}" \
  --minimum_epochs "${MITIGATION_MINIMUM_EPOCHS:-3}" \
  --early_stopping_patience "${MITIGATION_EARLY_STOPPING_PATIENCE:-1}" \
  --train_samples "${MITIGATION_TRAIN_SAMPLES:-5000}" \
  --eval_samples "${MITIGATION_EVAL_SAMPLES:-1000}" \
  --test_samples "${MITIGATION_TEST_SAMPLES:-26446}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --resume

printf '%s\n' "$OUTPUT" > "$ROOT/artifacts/logs/experiment_5_mandatory_latest.txt"
echo "[complete] Experiment V mandatory matrix: $OUTPUT/summary.json"
