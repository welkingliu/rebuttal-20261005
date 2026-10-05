#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

RUN_ID="${RUN_ID:-experiment3_formal_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EXP3_OUTPUT:-$SGG_ARTIFACT_DIR/experiment_3/$RUN_ID}"
TRAIN_SAMPLES="${EXP3_TRAIN_SAMPLES:-5000}"
EVAL_SAMPLES="${EXP3_EVAL_SAMPLES:-2000}"
MOTIFS_MANIFEST="${EXP3_MOTIFS_MANIFEST:-$SGG_OFFICIAL_MANIFEST_DIR/pysgg_motifs_vg_live.json}"
TRANSFORMER_MANIFEST="${EXP3_TRANSFORMER_MANIFEST:-$SGG_OFFICIAL_MANIFEST_DIR/pysgg_transformer_vg_live.json}"

mkdir -p "$OUTPUT/logs"

run_model() {
  local gpu="$1"
  local key="$2"
  local manifest="$3"
  local model_output="$OUTPUT/$key"
  local log="$OUTPUT/logs/$key.log"

  if [[ "${RESUME:-1}" == "1" && -s "$model_output/experiment_3.json" ]]; then
    printf '[resume] Experiment III %s already complete: %s\n' "$key" "$model_output"
    return
  fi

  mkdir -p "$model_output"
  printf '[start] Experiment III model=%s gpu=%s\n' "$key" "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    "$SGG_PYTHON" -m sgg_core.experiments.experiment_3 \
      --dataset vg \
      --data_root "$SGG_VG_ROOT" \
      --official_manifest "$manifest" \
      --output_dir "$model_output" \
      --train_samples "$TRAIN_SAMPLES" \
      --eval_samples "$EVAL_SAMPLES" \
      --top_k_motifs 20 \
      --minimum_train_support 20 \
      --minimum_eval_support 100 \
      --seed 17 \
      --device cuda 2>&1 |
    sed -u "s/^/[gpu=$gpu model=$key] /" |
    tee "$log"
  printf '[complete] Experiment III %s 实验结束\n' "$key"
}

for manifest in "$MOTIFS_MANIFEST" "$TRANSFORMER_MANIFEST"; do
  if [[ ! -s "$manifest" ]]; then
    printf 'Missing live manifest: %s\n' "$manifest" >&2
    exit 2
  fi
done

run_model 0 neural_motifs "$MOTIFS_MANIFEST" &
pid0=$!
run_model 1 sgg_transformer "$TRANSFORMER_MANIFEST" &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if ((status != 0)); then
  exit "$status"
fi

"$SGG_PYTHON" "$SCRIPT_DIR/validate_paper_experiment3.py" \
  --run_root "$OUTPUT"
printf '%s\n' "$OUTPUT" > "$SGG_LOG_DIR/experiment_3_latest.txt"
printf '[complete] Experiment III formal audit: %s\n' "$OUTPUT"
