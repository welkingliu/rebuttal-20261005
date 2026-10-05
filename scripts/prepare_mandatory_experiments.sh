#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
cd "$ROOT"

mkdir -p \
  "$ROOT/artifacts/logs" \
  "$ROOT/artifacts/manifests" \
  "$ROOT/artifacts/experiment_2" \
  "$ROOT/artifacts/experiment_4" \
  "$ROOT/artifacts/experiment_5" \
  "$ROOT/artifacts/prediction_cache" \
  "$ROOT/checkpoints/sgg/trained/pysgg" \
  "$ROOT/checkpoints/sgg/manifests"

"$PYTHON" scripts/generate_pysgg_vg_tritask_configs.py \
  --project_root "$ROOT" \
  --train_batch "${PYSGG_TRAIN_BATCH:-8}" \
  --test_batch "${PYSGG_TEST_BATCH:-2}" \
  --workers "${PYSGG_WORKERS:-4}"

"$PYTHON" scripts/check_mandatory_experiment_assets.py \
  --project_root "$ROOT" --require base external

echo "[ready] No additional model download is required for IV-B through V."
echo "         The chain reuses PySGG, shared_detector.pth, GloVe, and the"
echo "         task checkpoints/caches produced by IV-B and IV-C."
