#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

CHAIN_ID="${CHAIN_ID:-mandatory_$(date +%Y%m%d_%H%M%S)}"
export CHAIN_ID
mkdir -p "$SGG_LOG_DIR"

echo "chain_id=$CHAIN_ID"
echo "scope=IV broad SGDet plus two-family tri-task; parallel II/V controlled tests"

bash "$SCRIPT_DIR/prepare_mandatory_experiments.sh"
bash "$SCRIPT_DIR/run_experiment4_submission_chain.sh"

echo "submission_summary=$SGG_ARTIFACT_DIR/submission/$CHAIN_ID/summary.json"
