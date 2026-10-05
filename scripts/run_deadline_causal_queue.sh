#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$ROOT/artifacts/logs"
POLL_SECONDS="${CAUSAL_QUEUE_POLL_SECONDS:-300}"
PYTHON="${KAIHUA_SGG_PYTHON:-python3}"

wait_for_run() {
  local pointer="$1"
  local completion="$2"
  local expected_command="$3"
  [[ -s "$pointer" ]] || { echo "Missing queue pointer: $pointer" >&2; exit 1; }
  local run_id="$(<"$pointer")"
  local pid_file="$LOG_DIR/${run_id}.pid"
  local log="$LOG_DIR/${run_id}.log"
  [[ -s "$pid_file" ]] || { echo "Missing PID file: $pid_file" >&2; exit 1; }
  local pid="$(<"$pid_file")"
  echo "[wait] run=$run_id pid=$pid"
  while [[ -d "/proc/$pid" ]]; do
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$command" == *"$expected_command"* ]] || break
    sleep "$POLL_SECONDS"
  done
  if ! grep -qF "$completion" "$log"; then
    echo "[failed] prerequisite did not complete: $log" >&2
    tail -n 100 "$log" >&2 || true
    exit 1
  fi
}

wait_for_run \
  "$LOG_DIR/latest_post_export_queue.txt" \
  "[complete] deadline post-training export queue" \
  "run_deadline_post_training_queue.sh"

setup_run="$(<"$LOG_DIR/latest_kaihua_setup.txt")"
setup_log="$LOG_DIR/${setup_run}.log"
grep -qF '[complete] KAIHUA_SGG_PYTHON=' "$setup_log" || {
  echo "[failed] Kaihua runtime is not ready: $setup_log" >&2
  exit 1
}

run_effect() {
  local gpu="$1"
  local effect="$2"
  local samples="$3"
  echo "[causal] gpu=$gpu effect=$effect samples=$samples"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CAUSAL_EFFECT_TYPE="$effect" \
  CAUSAL_EVAL_SAMPLES="$samples" \
  SGG_PROJECT_ROOT="$ROOT" \
  SGG_PYTHON="$PYTHON" \
  KAIHUA_SGG_PYTHON="$PYTHON" \
    bash "$ROOT/scripts/run_causal_motifs_vg_export.sh"
}

run_effect 0 none 20 & smoke_none=$!
run_effect 1 TDE 20 & smoke_tde=$!
wait "$smoke_none"
wait "$smoke_tde"
echo "[smoke-complete] Causal Motifs none/TDE"

run_effect 0 none 26446 & full_none=$!
run_effect 1 TDE 26446 & full_tde=$!
wait "$full_none"
wait "$full_tde"

echo "[complete] Causal Motifs none/TDE full VG caches and manifests"
