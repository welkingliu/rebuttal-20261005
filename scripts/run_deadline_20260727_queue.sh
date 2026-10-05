#!/usr/bin/env bash
set -euo pipefail

# Deadline-converged queue:
#   1. discard Experiment V processes started before the live-validation fix;
#   2. run Causal Motifs-TDE tri-task export and the corrected mitigation gate;
#   3. expand the formal two-family mitigation matrix only after the gate passes.

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
LEGACY_PYTHON="${KAIHUA_SGG_PYTHON:-python3}"
RUN_ID="${RUN_ID:-deadline_20260727_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/artifacts/logs/$RUN_ID"
EXP5_OUTPUT="$ROOT/artifacts/experiment_5/${RUN_ID}_exp5"
GATE_OUTPUT="$EXP5_OUTPUT/gate"
GATE_REPORT="$GATE_OUTPUT/gate_report.json"
CAUSAL_LOG="$LOG_DIR/causal_motifs_tde.log"
GATE_LOG="$LOG_DIR/experiment5_gate.log"
MATRIX_LOG="$LOG_DIR/experiment5_matrix.log"
STALE_EXP5_ROOT="postcache_submission_20260723_104627_exp5"

mkdir -p "$LOG_DIR" "$EXP5_OUTPUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[failed] missing core Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$LEGACY_PYTHON" ]]; then
  echo "[failed] missing legacy Python: $LEGACY_PYTHON" >&2
  exit 1
fi

terminate_tree() {
  local parent="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    terminate_tree "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  kill -TERM "$parent" 2>/dev/null || true
}

stop_pre_fix_exp5() {
  local pid
  local -a parents=()
  while read -r pid; do
    [[ -n "$pid" ]] && parents+=("$pid")
  done < <(
    pgrep -f \
      "sgg_core[.]experiments[.]experiment_5.*${STALE_EXP5_ROOT}" \
      2>/dev/null || true
  )
  if (( ${#parents[@]} == 0 )); then
    echo "[stale-exp5] no pre-fix process is running"
    return
  fi
  if (( EUID != 0 )); then
    echo "[failed] root is required to stop pre-fix Experiment V processes" >&2
    printf "[stale-exp5] pid=%s\n" "${parents[@]}" >&2
    exit 1
  fi
  printf "[stale-exp5] stopping pid=%s and descendants\n" "${parents[@]}"
  for pid in "${parents[@]}"; do
    terminate_tree "$pid"
  done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "${parents[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${parents[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stale-exp5] force stopping pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

stop_pre_fix_exp5

"$PYTHON" scripts/check_mandatory_experiment_assets.py \
  --project_root "$ROOT" --require base tritask_caches live_manifests
"$PYTHON" scripts/check_legacy_vg_assets.py --project_root "$ROOT"

rm -f "$ROOT/artifacts/manifests/skip_causal_exports_for_converged_submission"

echo "[launch] GPU 0: Causal Motifs-SUM with TDE, PredCls/SGCls/SGDet"
(
  env \
    CUDA_VISIBLE_DEVICES=0 \
    CAUSAL_EFFECT_TYPE=TDE \
    CAUSAL_EVAL_SAMPLES=26446 \
    SGG_PROJECT_ROOT="$ROOT" \
    SGG_PYTHON="$PYTHON" \
    KAIHUA_SGG_PYTHON="$LEGACY_PYTHON" \
    PYTHONPATH="$PYTHONPATH" \
    bash "$ROOT/scripts/run_causal_motifs_vg_export.sh"
) >"$CAUSAL_LOG" 2>&1 &
causal_pid=$!

echo "[launch] GPU 1: corrected Experiment V gate"
(
  "$PYTHON" scripts/run_experiment5_gate.py \
    --project_root "$ROOT" \
    --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
    --output_dir "$GATE_OUTPUT" \
    --family "SGG Transformer" \
    --gpu 1 \
    --seed 17 \
    --epochs 3 \
    --minimum_epochs 2 \
    --early_stopping_patience 1 \
    --train_samples 1000 \
    --eval_samples 500 \
    --minimum_validation_objects 500 \
    --gradient_accumulation_steps 4
) >"$GATE_LOG" 2>&1 &
gate_pid=$!

printf '%s\n' "$causal_pid" >"$LOG_DIR/causal.pid"
printf '%s\n' "$gate_pid" >"$LOG_DIR/gate.pid"
printf '%s\n' "$RUN_ID" >"$ROOT/artifacts/logs/deadline_20260727_latest.txt"

causal_status=0
gate_status=0
wait "$causal_pid" || causal_status=$?
wait "$gate_pid" || gate_status=$?

if (( causal_status != 0 )); then
  echo "[warning] Causal Motifs export failed; Experiment V can still proceed."
  tail -n 100 "$CAUSAL_LOG" || true
fi
if (( gate_status != 0 )) || [[ ! -f "$GATE_REPORT" ]]; then
  echo "[failed] corrected Experiment V gate did not pass; matrix not expanded." >&2
  tail -n 150 "$GATE_LOG" >&2 || true
  exit 2
fi

"$PYTHON" - "$GATE_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "pass" or report.get("passed") is not True:
    raise SystemExit("Experiment V gate report is not passing")
PY

echo "[launch] GPUs 0/1: formal Experiment V, two families x two modes x three seeds"
"$PYTHON" scripts/run_experiment5_matrix.py \
  --project_root "$ROOT" \
  --manifest_dir "$ROOT/checkpoints/sgg/manifests" \
  --output_dir "$EXP5_OUTPUT" \
  --gate_report "$GATE_REPORT" \
  --classic_family "Neural Motifs" \
  --transformer_family "SGG Transformer" \
  --dataset vg \
  --seeds 17 23 31 \
  --training_modes supervised_control grounding \
  --gpus 0 1 \
  --epochs 5 \
  --minimum_epochs 3 \
  --early_stopping_patience 1 \
  --train_samples 5000 \
  --eval_samples 1000 \
  --test_samples 26446 \
  --gradient_accumulation_steps 4 \
  --resume \
  >"$MATRIX_LOG" 2>&1

echo "[complete] deadline queue"
echo "causal_status=$causal_status"
echo "causal_log=$CAUSAL_LOG"
echo "gate_report=$GATE_REPORT"
echo "experiment5_summary=$EXP5_OUTPUT/summary.json"
