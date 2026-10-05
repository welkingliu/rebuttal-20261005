#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
QUEUE_ID="${MAC_QUEUE_ID:-mac_cpu_queue_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="$ROOT/artifacts/mac_queue/$QUEUE_ID"
LOG_DIR="$ROOT/artifacts/logs"
QUEUE_LOG="$LOG_DIR/${QUEUE_ID}.log"

mkdir -p "$QUEUE_ROOT" "$LOG_DIR"
printf '%s\n' "$QUEUE_ID" > "$LOG_DIR/mac_cpu_queue_latest_run_id.txt"
exec > >(tee -a "$QUEUE_LOG") 2>&1

on_error() {
  local rc=$?
  printf '[FAILED] queue=%s rc=%s time=%s\n' \
    "$QUEUE_ID" "$rc" "$(date -Iseconds)"
  touch "$QUEUE_ROOT/FAILED"
  exit "$rc"
}
trap on_error ERR

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python runtime: $PYTHON" >&2
  exit 1
fi

# Keep the Mac awake while the queue owns unfinished CPU work.
caffeinate -dimsu -w $$ &
CAFFEINATE_PID=$!
cleanup() {
  kill "$CAFFEINATE_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_step() {
  local name="$1"
  local expected="$2"
  shift 2
  local marker="$QUEUE_ROOT/${name}.complete"
  if [[ -s "$marker" && -s "$expected" ]]; then
    echo "[resume-skip] $name expected=$expected"
    return
  fi
  echo "[start] $name time=$(date -Iseconds)"
  "$@"
  if [[ ! -s "$expected" ]]; then
    echo "[failed] $name did not create $expected" >&2
    return 1
  fi
  printf '%s\n' "$(date -Iseconds)" > "$marker"
  echo "[complete] $name expected=$expected"
}

cd "$ROOT"

CURRENT_RUN_ID="$(<"$LOG_DIR/mac_exp4_vg_full_latest_run_id.txt")"
CURRENT_SUMMARY="$ROOT/artifacts/experiment_4/$CURRENT_RUN_ID/summary.json"
CURRENT_PID="$(
  pgrep -f "sgg_core[.]experiments[.]experiment_4.*$CURRENT_RUN_ID" \
    | sort -n | tail -n 1 || true
)"
echo "[wait] current Experiment IV run=$CURRENT_RUN_ID pid=${CURRENT_PID:-none}"
if [[ -n "$CURRENT_PID" ]]; then
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 30
  done
fi
if [[ ! -s "$CURRENT_SUMMARY" ]]; then
  echo "Current Experiment IV ended without summary: $CURRENT_SUMMARY" >&2
  exit 1
fi
echo "[complete] current Experiment IV summary=$CURRENT_SUMMARY"

KERN_SGCLS_OUT="$ROOT/artifacts/experiment_4/${QUEUE_ID}_kern_sgcls_full"
run_step \
  exp4_kern_sgcls_full "$KERN_SGCLS_OUT/summary.json" \
  env PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" "$PYTHON" \
    -m sgg_core.experiments.experiment_4 \
    --datasets vg \
    --vg_root "$ROOT/data/vg/v1.4" \
    --official_manifest "$ROOT/checkpoints/sgg/manifests/kern_official_vg.json" \
    --seen_triplets_manifest "$ROOT/artifacts/manifests/seen_triplets_full.json" \
    --output_dir "$KERN_SGCLS_OUT" \
    --steps standard \
    --sgg_tasks sgcls \
    --recall_ks 1 5 10 20 50 100 \
    --train_samples 5000 \
    --eval_samples 1000000000 \
    --minimum_model_families 1 \
    --minimum_models_per_dataset 1 \
    --allow_unlisted_family \
    --allow_incomplete_audits \
    --device cpu

PSG_SGDET_OUT="$ROOT/artifacts/experiment_4/${QUEUE_ID}_psg_sgdet_full"
run_step \
  exp4_psg_sgdet_full "$PSG_SGDET_OUT/summary.json" \
  env PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" "$PYTHON" \
    -m sgg_core.experiments.experiment_4 \
    --datasets psg \
    --psg_train_ann "$ROOT/data/psg/psg_train_val.json" \
    --psg_eval_ann "$ROOT/data/psg/psg_val_test.json" \
    --official_manifest "$ROOT/checkpoints/sgg/manifests/openpsg_motifs_psg.json" \
    --official_manifest "$ROOT/checkpoints/sgg/manifests/openpsg_vctree_psg.json" \
    --official_manifest "$ROOT/checkpoints/sgg/manifests/openpsg_psgtr_psg.json" \
    --official_manifest "$ROOT/checkpoints/sgg/manifests/openpsg_psgformer_psg.json" \
    --seen_triplets_manifest "$ROOT/artifacts/manifests/seen_triplets_full.json" \
    --output_dir "$PSG_SGDET_OUT" \
    --steps standard \
    --sgg_tasks sgdet \
    --recall_ks 1 5 10 20 50 100 \
    --train_samples 5000 \
    --eval_samples 1000000000 \
    --minimum_model_families 4 \
    --minimum_models_per_dataset 4 \
    --allow_unlisted_family \
    --allow_incomplete_audits \
    --device cpu

EXP2_VG_OUT="$ROOT/artifacts/experiment_2/${QUEUE_ID}_vg_observational_full"
run_step \
  exp2_vg_observational_full "$EXP2_VG_OUT/summary.json" \
  "$PYTHON" scripts/run_diagnostic_matrix.py \
    --experiment 2 \
    --analysis_scope observational \
    --project_root "$ROOT" \
    --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
    --output_dir "$EXP2_VG_OUT" \
    --datasets vg \
    --families EGTR SGTR kern \
    --gpus cpu0 cpu1 \
    --device cpu \
    --minimum_families 3 \
    --maximum_families 3 \
    --dataset_model_targets vg=3 \
    --train_samples 5000 \
    --eval_samples 1000000000 \
    --resume

EXP2_PSG_OUT="$ROOT/artifacts/experiment_2/${QUEUE_ID}_psg_observational_full"
run_step \
  exp2_psg_observational_full "$EXP2_PSG_OUT/summary.json" \
  "$PYTHON" scripts/run_diagnostic_matrix.py \
    --experiment 2 \
    --analysis_scope observational \
    --project_root "$ROOT" \
    --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
    --output_dir "$EXP2_PSG_OUT" \
    --datasets psg \
    --families "Neural Motifs" VCTree \
    --gpus cpu0 cpu1 \
    --device cpu \
    --minimum_families 2 \
    --maximum_families 2 \
    --dataset_model_targets psg=2 \
    --train_samples 5000 \
    --eval_samples 1000000000 \
    --resume

INTEGRATION_REPORT="$ROOT/artifacts/manifests/${QUEUE_ID}_official_integration.json"
run_step \
  official_integration "$INTEGRATION_REPORT" \
  "$PYTHON" scripts/check_official_integration.py \
    --project_root "$ROOT" \
    --check_factory_imports \
    --report "$INTEGRATION_REPORT"

CONTRACT_REPORT="$ROOT/artifacts/manifests/${QUEUE_ID}_server_remaining_contract.json"
echo "[report] submission contract; remaining failures are expected to be server-only"
set +e
"$PYTHON" scripts/check_submission_contract.py \
  --project_root "$ROOT" \
  --report "$CONTRACT_REPORT"
CONTRACT_RC=$?
set -e
if [[ ! -s "$CONTRACT_REPORT" ]]; then
  echo "Submission contract did not create $CONTRACT_REPORT" >&2
  exit 1
fi
echo "[report-complete] contract_rc=$CONTRACT_RC report=$CONTRACT_REPORT"

printf '%s\n' "$(date -Iseconds)" > "$QUEUE_ROOT/ALL_MAC_TASKS_COMPLETE"
sync
echo "[all-mac-tasks-complete] queue=$QUEUE_ID"
echo "[shutdown-request] time=$(date -Iseconds)"

if ! /usr/bin/osascript -e 'tell application "System Events" to shut down'; then
  echo "System Events shutdown failed; trying Finder" >&2
  /usr/bin/osascript -e 'tell application "Finder" to shut down'
fi
