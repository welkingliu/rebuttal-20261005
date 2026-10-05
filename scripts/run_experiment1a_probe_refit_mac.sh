#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-$HOME/miniconda3/envs/sgg_proj/bin/python}"
RUN_ID="${RUN_ID:-exp1a_refit_zscore_$(date +%Y%m%d_%H%M%S)}"
SOURCE_RUN="${EXP1A_SOURCE_RUN:-exp1a_converged_20260718_174828}"
DEVICE="${EXP1A_REFIT_DEVICE:-mps}"
OUTPUT="$ROOT/artifacts/experiment_1a/$RUN_ID"
CACHE_ROOT="$ROOT/data/derived/features/experiment_1a"
BACKBONES=(
  cradio_v4_so400m dinov2_b radio_v25_b siglip2_b sam_vit_b resnet50
)

cd "$ROOT"
mkdir -p "$OUTPUT" "$ROOT/artifacts/logs"

for backbone in "${BACKBONES[@]}"; do
  train_count="$(find "$CACHE_ROOT/$backbone" -maxdepth 1 -type f \
    -name '*_train_5000_*.pt' | wc -l | tr -d ' ')"
  eval_count="$(find "$CACHE_ROOT/$backbone" -maxdepth 1 -type f \
    -name '*_eval_1000_*.pt' | wc -l | tr -d ' ')"
  [[ "$train_count" -eq 1 ]] || {
    echo "[failed] expected one train cache for $backbone" >&2
    exit 1
  }
  [[ "$eval_count" -eq 1 ]] || {
    echo "[failed] expected one eval cache for $backbone" >&2
    exit 1
  }
  train_cache="$(find "$CACHE_ROOT/$backbone" -maxdepth 1 -type f \
    -name '*_train_5000_*.pt' | sort)"
  eval_cache="$(find "$CACHE_ROOT/$backbone" -maxdepth 1 -type f \
    -name '*_eval_1000_*.pt' | sort)"
  source_summary="$ROOT/artifacts/experiment_1a/$SOURCE_RUN/$backbone/summary.json"
  [[ -s "$source_summary" ]] || {
    echo "[failed] missing source summary: $source_summary" >&2
    exit 1
  }

  echo "[start] $backbone device=$DEVICE"
  "$PYTHON" -m sgg_core.experiments.experiment_1a \
    --backbone "$backbone" \
    --train_cache "$train_cache" \
    --eval_cache "$eval_cache" \
    --source_summary "$source_summary" \
    --output_dir "$OUTPUT/$backbone" \
    --feature_normalization zscore \
    --probe_epochs 500 \
    --early_stopping_patience 20 \
    --early_stopping_min_delta 1e-5 \
    --probe_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --seeds 17 23 31 \
    --device "$DEVICE"
  echo "[complete] $backbone"
done

printf '%s\n' "$RUN_ID" > "$ROOT/artifacts/logs/experiment_1a_refit_latest.txt"
echo "[complete] Experiment I-A normalized probe refit: $OUTPUT"
