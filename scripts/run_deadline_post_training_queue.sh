#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$ROOT/artifacts/logs"
LATEST="$LOG_DIR/latest_deadline_training.txt"
POLL_SECONDS="${POST_QUEUE_POLL_SECONDS:-300}"

[[ -s "$LATEST" ]] || { echo "Missing queue pointer: $LATEST" >&2; exit 1; }
RUN_ID="$(<"$LATEST")"
TRAIN_PID_FILE="$LOG_DIR/${RUN_ID}.pid"
TRAIN_LOG="$LOG_DIR/${RUN_ID}.log"
[[ -s "$TRAIN_PID_FILE" ]] || { echo "Missing PID file: $TRAIN_PID_FILE" >&2; exit 1; }
TRAIN_PID="$(<"$TRAIN_PID_FILE")"

echo "[wait] training_run=$RUN_ID pid=$TRAIN_PID"
while [[ -d "/proc/$TRAIN_PID" ]]; do
  command="$(tr '\0' ' ' < "/proc/$TRAIN_PID/cmdline" 2>/dev/null || true)"
  [[ "$command" == *run_pysgg_vg_tritask_training_2gpu.sh* ]] || break
  sleep "$POLL_SECONDS"
done

if ! grep -q '^\[complete\] all requested task-specific PySGG runs$' "$TRAIN_LOG"; then
  echo "[failed] training queue did not complete cleanly: $TRAIN_LOG" >&2
  tail -n 100 "$TRAIN_LOG" >&2 || true
  exit 1
fi

echo "[export] motifs and transformer full VG caches"
PYSGG_EXPORT_FAMILIES="motifs transformer" \
PYSGG_EXPORT_MINIMUM_FAMILIES=2 \
  bash "$ROOT/scripts/run_pysgg_vg_tritask_export_2gpu.sh"

echo "[complete] deadline post-training export queue"
