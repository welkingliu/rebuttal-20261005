#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
CHAIN_ID="${CHAIN_ID:-mandatory_$(date +%Y%m%d_%H%M%S)}"
cd "$ROOT"

if [[ ! -f "$ROOT/checkpoints/sgg/manifests/pysgg_bgnn_vg_sgdet.json" ]]; then
  bash scripts/run_pysgg_bgnn_sgdet_released.sh
fi

if [[ ! -f "$ROOT/artifacts/manifests/pysgg_worker_transport_smoke.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/smoke_pysgg_worker_transport.py \
    --project_root "$ROOT" --device cuda
fi

if "$PYTHON" scripts/check_vg_tritask_assets.py \
    --project_root "$ROOT" --minimum_families 2 --expected_images 26446 \
    >/dev/null 2>&1; then
  echo "[skip-ready] complete Motifs/Transformer VG tri-task caches"
else
  PYSGG_FAMILIES="motifs transformer" \
    bash scripts/run_pysgg_vg_tritask_training_2gpu.sh
  PYSGG_EXPORT_FAMILIES="motifs transformer" \
  PYSGG_EXPORT_MINIMUM_FAMILIES=2 \
    bash scripts/run_pysgg_vg_tritask_export_2gpu.sh
fi
RUN_ID="${CHAIN_ID}_exp4" bash scripts/run_experiment4_converged_depth_2gpu.sh

"$PYTHON" scripts/register_pysgg_live_manifests.py \
  --project_root "$ROOT" --models motifs transformer
"$PYTHON" scripts/check_mandatory_experiment_assets.py \
  --project_root "$ROOT" --require all
"$PYTHON" scripts/smoke_pysgg_live_adapters.py \
  --project_root "$ROOT" --models motifs transformer --device cuda

echo "[stage] run II-B on GPU 0; run V/Transformer on GPU 1 immediately"
echo "[stage] V/Motifs will acquire GPU 0 as soon as II-B exits"
RUN_ID="${CHAIN_ID}_exp2" EXP2_GPUS="0" \
  bash scripts/run_experiment2_mandatory_2gpu.sh &
exp2_pid=$!
RUN_ID="${CHAIN_ID}_exp5" MITIGATION_GPUS="0 1" \
MITIGATION_WAIT_GPU="0=$exp2_pid" \
  bash scripts/run_experiment5_mandatory_2gpu.sh &
exp5_pid=$!

status=0
wait "$exp2_pid" || status=1
wait "$exp5_pid" || status=1
if (( status != 0 )); then
  echo "[failed] II-B or V did not complete; inspect their run summaries" >&2
  exit "$status"
fi

"$PYTHON" scripts/aggregate_mandatory_submission.py \
  --experiment4 "$ROOT/artifacts/experiment_4/${CHAIN_ID}_exp4" \
  --experiment2 "$ROOT/artifacts/experiment_2/${CHAIN_ID}_exp2" \
  --experiment5 "$ROOT/artifacts/experiment_5/${CHAIN_ID}_exp5" \
  --output_dir "$ROOT/artifacts/submission/${CHAIN_ID}"

echo "[complete] mandatory IV -> II -> V submission chain: $CHAIN_ID"
