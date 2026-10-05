#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${OPENPSG_PYTHON:-python3}"
if (( $# )); then
  MODELS=("$@")
else
  MODELS=(psgtr psgformer motifs vctree)
fi

cd "$PROJECT_ROOT"
test -x "$PYTHON"
PYTHONPATH="$PROJECT_ROOT/external/official_repos/OpenPSG" \
  "$PYTHON" -c 'import torch, mmcv, mmdet, detectron2, openpsg; print("[runtime-ok]", torch.__version__, mmcv.__version__, mmdet.__version__)'

for model in "${MODELS[@]}"; do
  echo "[smoke] model=$model"
  "$PYTHON" scripts/export_openpsg_predictions.py \
    --project_root "$PROJECT_ROOT" \
    --model "$model" \
    --output_dir "$PROJECT_ROOT/artifacts/prediction_cache/openpsg_${model}_psg_smoke" \
    --eval_samples 1 \
    --log_every 1
done

echo "[complete] OpenPSG smoke panel"
