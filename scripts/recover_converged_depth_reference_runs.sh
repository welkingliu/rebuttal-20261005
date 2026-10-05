#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?Set RUN_ID to the Experiment IV submission run id}"
TRANSFORMER_PID="${TRANSFORMER_PID:?Set TRANSFORMER_PID to the strict Transformer evaluator}"
MOTIFS_PID="${MOTIFS_PID:?Set MOTIFS_PID to the diagnostic Motifs evaluator}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"

RUN_ROOT="$ROOT/artifacts/experiment_4/${RUN_ID}_exp4"
SHARD_ROOT="$RUN_ROOT/shards"
TRANSFORMER_OUTPUT="$SHARD_ROOT/pysgg_transformer_vg_tritask/vg"
MOTIFS_OUTPUT="$SHARD_ROOT/pysgg_motifs_vg_tritask/vg"
TRANSFORMER_SUMMARY="$TRANSFORMER_OUTPUT/summary.json"
TRANSFORMER_RESULTS="$TRANSFORMER_OUTPUT/vg/results.json"
MOTIFS_SUMMARY="$MOTIFS_OUTPUT/summary.json"
MOTIFS_RESULTS="$MOTIFS_OUTPUT/vg/results.json"
TRANSFORMER_LOG="$RUN_ROOT/logs/pysgg_transformer_vg_tritask_vg_allow_incomplete.log"

wait_for_pid() {
  local pid="$1"
  local label="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  echo "[recovery] $label exited at $(date -Is)"
}

run_transformer_diagnostic() {
  echo "[recovery] strict Transformer output missing; starting diagnostic rerun"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" \
    "$SGG_PYTHON" -m sgg_core.experiments.experiment_4 \
      --datasets vg \
      --official_manifest \
        "$ROOT/artifacts/manifests/experiment4_converged_depth/pysgg_transformer_vg_tritask.json" \
      --model_panel "$ROOT/sgg_core/models/model_panel.json" \
      --output_dir "$TRANSFORMER_OUTPUT" \
      --minimum_model_families 1 \
      --minimum_models_per_dataset 1 \
      --steps standard grounding \
      --seen_triplets_manifest "$ROOT/artifacts/manifests/seen_triplets_full.json" \
      --train_samples 5000 \
      --eval_samples 1000000000 \
      --device cuda \
      --vg_root "$ROOT/data/vg/v1.4" \
      --allow_incomplete_audits \
    > "$TRANSFORMER_LOG" 2>&1
}

source "$ROOT/scripts/project_env.sh"
wait_for_pid "$TRANSFORMER_PID" "strict Transformer evaluator"
if [[ ! -f "$TRANSFORMER_SUMMARY" || ! -f "$TRANSFORMER_RESULTS" ]]; then
  run_transformer_diagnostic
fi

wait_for_pid "$MOTIFS_PID" "diagnostic Motifs evaluator"
test -f "$TRANSFORMER_SUMMARY"
test -f "$TRANSFORMER_RESULTS"
test -f "$MOTIFS_SUMMARY"
test -f "$MOTIFS_RESULTS"

echo "[recovery] both depth-family result shards are durable"
echo "[recovery] IV reference reconciliation remains pending"

"$SGG_PYTHON" "$ROOT/scripts/register_pysgg_live_manifests.py" \
  --project_root "$ROOT" --models motifs transformer
"$SGG_PYTHON" "$ROOT/scripts/check_mandatory_experiment_assets.py" \
  --project_root "$ROOT" --require all

echo "[recovery] smoke-loading live families in parallel"
CUDA_VISIBLE_DEVICES=0 "$SGG_PYTHON" \
  "$ROOT/scripts/smoke_pysgg_live_adapters.py" \
  --project_root "$ROOT" --models motifs --device cuda &
motifs_smoke_pid=$!
CUDA_VISIBLE_DEVICES=1 "$SGG_PYTHON" \
  "$ROOT/scripts/smoke_pysgg_live_adapters.py" \
  --project_root "$ROOT" --models transformer --device cuda &
transformer_smoke_pid=$!
smoke_status=0
wait "$motifs_smoke_pid" || smoke_status=1
wait "$transformer_smoke_pid" || smoke_status=1
if (( smoke_status != 0 )); then
  echo "[recovery-failed] parallel live-adapter smoke failed" >&2
  exit "$smoke_status"
fi

echo "[recovery] starting II-B on GPU 0 and V/Transformer on GPU 1"
RUN_ID="${RUN_ID}_exp2" EXP2_GPUS="0" \
  bash "$ROOT/scripts/run_experiment2_mandatory_2gpu.sh" &
exp2_pid=$!
RUN_ID="${RUN_ID}_exp5" MITIGATION_GPUS="0 1" \
MITIGATION_WAIT_GPU="0=$exp2_pid" \
  bash "$ROOT/scripts/run_experiment5_mandatory_2gpu.sh" &
exp5_pid=$!

status=0
wait "$exp2_pid" || status=1
wait "$exp5_pid" || status=1
if (( status != 0 )); then
  echo "[recovery-failed] II-B or V did not complete" >&2
  exit "$status"
fi
echo "[recovery-complete] downstream II-B and V completed at $(date -Is)"
