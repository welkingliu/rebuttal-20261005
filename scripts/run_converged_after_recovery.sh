#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UPSTREAM_PID="${UPSTREAM_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
CHAIN_ID="${CHAIN_ID:-converged_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/artifacts/logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"
printf '%s\n' "$CHAIN_ID" > "$LOG_DIR/latest_converged_submission_chain.txt"

if [[ -n "$UPSTREAM_PID" ]]; then
  echo "[wait] upstream_pid=$UPSTREAM_PID"
  while [[ -d "/proc/$UPSTREAM_PID" ]]; do
    sleep "$POLL_SECONDS"
  done
fi

# The submission chain performs the resumable export before validating the
# tri-task caches. Requiring metadata here would make a freshly completed
# training run fail before its predictions can be exported.
echo "[start] export/validate IV-depth -> parallel II-B/V chain=$CHAIN_ID"
CHAIN_ID="$CHAIN_ID" bash scripts/run_experiment4_submission_chain.sh
echo "[complete] compute-converged submission chain=$CHAIN_ID"
