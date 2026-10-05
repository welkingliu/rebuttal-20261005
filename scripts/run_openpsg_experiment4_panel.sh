#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPSG_PYTHON="${OPENPSG_PYTHON:-python3}"
MAIN_PYTHON="${SGG_PYTHON:-python3}"
if (( $# )); then
  MODELS=("$@")
else
  MODELS=(psgtr psgformer motifs vctree)
fi

PSG_ROOT="$PROJECT_ROOT/data/psg"
COCO_ROOT="$PROJECT_ROOT/data/coco"
PSG_OFFICIAL_TEST="$PROJECT_ROOT/data/derived/psg/psg_official_test.json"
SEEN="$PROJECT_ROOT/artifacts/manifests/seen_triplets_full.json"

cd "$PROJECT_ROOT"
test -x "$OPENPSG_PYTHON"
test -x "$MAIN_PYTHON"
test -f "$SEEN"

"$MAIN_PYTHON" scripts/build_psg_official_test_split.py \
  --project_root "$PROJECT_ROOT" \
  --output "$PSG_OFFICIAL_TEST"

"$MAIN_PYTHON" -m sgg_core.tools.prepare_reviewer_datasets \
  --project_root "$PROJECT_ROOT" \
  --datasets psg \
  --psg_eval_ann "$PSG_OFFICIAL_TEST" \
  --external_samples 0 \
  --strict_images

for model in "${MODELS[@]}"; do
  cache="$PROJECT_ROOT/artifacts/prediction_cache/openpsg_${model}_psg"
  manifest="$PROJECT_ROOT/checkpoints/sgg/manifests/openpsg_${model}_psg.json"
  output="$PROJECT_ROOT/artifacts/experiment_4/openpsg_${model}_psg_full"

  echo "[export] model=$model"
  "$OPENPSG_PYTHON" scripts/export_openpsg_predictions.py \
    --project_root "$PROJECT_ROOT" \
    --model "$model" \
    --output_dir "$cache" \
    --eval_samples 2177 \
    --resume

  "$MAIN_PYTHON" -m sgg_core.tools.validate_prediction_cache \
    --cache_root "$cache" \
    --dataset psg \
    --data_root "$PSG_ROOT" \
    --train_ann "$PSG_ROOT/psg_train_val.json" \
    --eval_ann "$PSG_OFFICIAL_TEST" \
    --image_root "$COCO_ROOT" \
    --panoptic_root "$COCO_ROOT/panoptic_val2017" \
    --eval_samples 2177 \
    --report "$cache/validation_report.json"

  "$MAIN_PYTHON" scripts/build_openpsg_cache_manifest.py \
    --project_root "$PROJECT_ROOT" \
    --model "$model" \
    --cache_root "$cache" \
    --output "$manifest"

  echo "[experiment-4] model=$model"
  "$MAIN_PYTHON" -m sgg_core.experiments.experiment_4 \
    --datasets psg \
    --psg_train_ann "$PSG_ROOT/psg_train_val.json" \
    --psg_eval_ann "$PSG_OFFICIAL_TEST" \
    --psg_image_root "$COCO_ROOT" \
    --official_manifest "$manifest" \
    --steps standard \
    --sgg_tasks sgdet \
    --train_samples 1 \
    --eval_samples 2177 \
    --minimum_model_families 1 \
    --minimum_models_per_dataset 1 \
    --seen_triplets_manifest "$SEEN" \
    --output_dir "$output" \
    --device cuda
done

echo "[complete] OpenPSG Experiment-IV panel"
