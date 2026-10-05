#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYSGG_PYTHON="${PYSGG_PYTHON:-python3}"
CORE_PYTHON="${SGG_PYTHON:-python3}"

cd "$ROOT"

echo "[stage] safe Transformer PredCls/SGCls/SGDet training"
PYSGG_FAMILIES="transformer" \
PYSGG_TASKS="predcls sgcls sgdet" \
PYSGG_PYTHON="$PYSGG_PYTHON" \
  bash scripts/run_pysgg_vg_tritask_training_2gpu.sh

for task in predcls sgcls sgdet; do
  marker="$ROOT/checkpoints/sgg/trained/pysgg/transformer/$task/.sgg_training_complete.json"
  [[ -s "$marker" ]] || {
    echo "[failed] missing Transformer completion marker: $marker" >&2
    exit 1
  }
done

echo "[stage] export Motifs and Transformer full-VG tri-task caches"
PYSGG_EXPORT_FAMILIES="motifs transformer" \
PYSGG_EXPORT_MINIMUM_FAMILIES=2 \
PYSGG_PYTHON="$PYSGG_PYTHON" \
SGG_PYTHON="$CORE_PYTHON" \
  bash scripts/run_pysgg_vg_tritask_export_2gpu.sh

echo "[stage] start converged submission experiments"
bash scripts/run_converged_after_recovery.sh

echo "[complete] safe Transformer and converged submission queue"
