#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYSGG_PYTHON:-python3}"
REPO="$ROOT/external/official_repos/PySGG"
LOG_DIR="$ROOT/artifacts/logs/pysgg_vg_tritask_training"
FAMILIES=(${PYSGG_FAMILIES:-motifs vctree transformer bgnn tde_motifs})
TASKS=(${PYSGG_TASKS:-predcls sgcls sgdet})

export PYTHONPATH="$ROOT/legacy_runtime:$REPO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p "$LOG_DIR"

"$PYTHON" "$ROOT/scripts/patch_pysgg_predcls_eval.py" \
  --project_root "$ROOT"

latest_checkpoint() {
  local output="$1"
  [[ -d "$output" ]] || return 0
  find "$output" -type f \
    \( -name model_final.pth -o -name 'model_[0-9]*.pth' \) \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

log_has_completed_test() {
  local log="$1"
  [[ -f "$log" ]] \
    && grep -q 'dataset(26446 images)' "$log" \
    && grep -q 'SGG eval:' "$log"
}

log_has_completed_training() {
  local log="$1"
  [[ -f "$log" ]] && grep -q 'Total training time:' "$log"
}

run_single_gpu_test() {
  local checkpoint="$1"
  local config="$2"
  local log="$3"
  local family="$4"
  local task="$5"
  local eval_output="$ROOT/artifacts/native_predictions/pysgg_${family}_${task}_formal"
  local attempt

  mkdir -p "$eval_output" "$(dirname "$log")"
  for attempt in 1 2; do
    echo "[evaluate] $family/$task single-GPU attempt=$attempt log=$log"
    cd "$REPO"
    if CUDA_VISIBLE_DEVICES="${PYSGG_EVAL_GPU:-0}" "$PYTHON" \
        tools/relation_test_net.py \
        --config-file "$config" \
        MODEL.WEIGHT "$checkpoint" \
        OUTPUT_DIR "$eval_output" \
        TEST.IMS_PER_BATCH 1 \
        TEST.ALLOW_LOAD_FROM_CACHE False \
        TEST.RELATION.SYNC_GATHER False \
        > "$log" 2>&1 \
        && log_has_completed_test "$log"; then
      return 0
    fi
    echo "[evaluate-retry] $family/$task attempt=$attempt failed" >&2
    sleep 10
  done
  return 1
}

register_completed_checkpoint() {
  local output="$1"
  local checkpoint="$2"
  local config="$3"
  local log="$4"
  local evaluation_log="$5"
  local family="$6"
  local task="$7"
  mkdir -p "$output"
  printf '%s\n' "$checkpoint" > "$output/last_checkpoint"
  "$PYTHON" - "$output/.sgg_training_complete.json" "$checkpoint" \
    "$config" "$log" "$evaluation_log" "$family" "$task" <<'PY'
import datetime as dt
import json
import os
import sys
import torch
import yaml

marker, checkpoint, config, log, evaluation_log, family, task = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu")
cfg = yaml.safe_load(open(config, encoding="utf-8"))
record = {
    "schema_version": 2,
    "family": family,
    "task": task,
    "checkpoint": os.path.abspath(checkpoint),
    "checkpoint_bytes": os.path.getsize(checkpoint),
    "checkpoint_iteration": int(payload.get("iteration", -1)),
    "configured_max_iteration": int(cfg["SOLVER"]["MAX_ITER"]),
    "training_log": os.path.abspath(log),
    "formal_evaluation_log": os.path.abspath(evaluation_log),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "completion_contract": [
        "training_loop_completed",
        "single_gpu_full_vg_test_evaluation_exit_0",
        "full_vg_test_metrics_logged",
        "checkpoint_present",
    ],
}
tmp = marker + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, marker)
print(record["checkpoint_iteration"])
PY
}

port=12100
failures=()
for family in "${FAMILIES[@]}"; do
  for task in "${TASKS[@]}"; do
    config="$ROOT/configs/pysgg_vg_tritask/${family}_${task}.yaml"
    output="$ROOT/checkpoints/sgg/trained/pysgg/$family/$task"
    log="$LOG_DIR/${family}_${task}.log"
    evaluation_log="$LOG_DIR/${family}_${task}_single_gpu_test.log"
    marker="$output/.sgg_training_complete.json"
    final="$(latest_checkpoint "$output")"
    if [[ -s "$marker" && -s "$output/last_checkpoint" ]]; then
      final="$(<"$output/last_checkpoint")"
      if [[ ! -f "$final" ]]; then
        echo "[invalid] $family/$task marker points to missing checkpoint: $final" >&2
        exit 1
      fi
      echo "[skip-complete] $family/$task checkpoint=$final"
      continue
    fi
    if [[ -n "$final" && -f "$final" ]] && log_has_completed_training "$log"; then
      echo "[recover-training-complete] $family/$task checkpoint=$final"
      if log_has_completed_test "$evaluation_log" \
          || run_single_gpu_test \
            "$final" "$config" "$evaluation_log" "$family" "$task"; then
        actual_iter="$(register_completed_checkpoint \
          "$output" "$final" "$config" "$log" "$evaluation_log" \
          "$family" "$task")"
        echo "[import-complete] $family/$task checkpoint=$final iteration=$actual_iter"
      else
        echo "[failed-evaluation] $family/$task; continuing with the next task" >&2
        failures+=("$family/$task:evaluation")
      fi
      continue
    fi
    if find "$output" -type f -name 'model_*.pth' -print -quit 2>/dev/null \
        | grep -q .; then
      echo "[restart] $family/$task has a periodic checkpoint but PySGG does not resume optimizer state"
    fi
    echo "[train] $family/$task log=$log"
    cd "$REPO"
    set +e
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -m torch.distributed.launch \
      --master_port "$port" --nproc_per_node=2 \
      tools/relation_train_net.py --config-file "$config" --skip-test \
      SOLVER.PRE_VAL False SOLVER.TO_VAL False \
      > "$log" 2>&1
    training_status=$?
    set -e
    final="$(latest_checkpoint "$output")"
    if (( training_status != 0 )) || ! log_has_completed_training "$log"; then
      echo "[failed-training] $family/$task status=$training_status; continuing with the next task" >&2
      failures+=("$family/$task:training")
      port=$((port + 1))
      continue
    fi
    if [[ -z "$final" || ! -f "$final" ]]; then
      echo "[failed-checkpoint] $family/$task exited without a checkpoint; continuing" >&2
      failures+=("$family/$task:checkpoint")
      port=$((port + 1))
      continue
    fi
    if run_single_gpu_test \
        "$final" "$config" "$evaluation_log" "$family" "$task"; then
      actual_iter="$(register_completed_checkpoint \
        "$output" "$final" "$config" "$log" "$evaluation_log" \
        "$family" "$task")"
      echo "[train-complete] $family/$task checkpoint=$final iteration=$actual_iter"
    else
      echo "[failed-evaluation] $family/$task; continuing with the next task" >&2
      failures+=("$family/$task:evaluation")
    fi
    port=$((port + 1))
  done
done

if (( ${#failures[@]} > 0 )); then
  printf '[complete-with-failures] %s\n' "${failures[*]}" >&2
  exit 1
fi

echo "[complete] all requested task-specific PySGG runs"
