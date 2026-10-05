#!/usr/bin/env bash
set -euo pipefail

echo "[BLOCKED] This recovery queue predates the live-validation gate." >&2
echo "Run scripts/run_experiment5_gate.py, then the gated formal matrix." >&2
exit 2

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
FAMILY="${MITIGATION_FAMILY:?Set MITIGATION_FAMILY to motifs or transformer}"
GPU="${MITIGATION_GPU:-0}"
OUTPUT="${EXP5_OUTPUT:?Set EXP5_OUTPUT to the existing Experiment V directory}"

case "$FAMILY" in
  motifs)
    MANIFEST="$ROOT/checkpoints/sgg/manifests/pysgg_motifs_vg_live.json"
    FAMILY_DIR="Neural_Motifs"
    ;;
  transformer)
    MANIFEST="$ROOT/checkpoints/sgg/manifests/pysgg_transformer_vg_live.json"
    FAMILY_DIR="SGG_Transformer"
    ;;
  *)
    echo "Unsupported MITIGATION_FAMILY=$FAMILY" >&2
    exit 2
    ;;
esac

is_complete() {
  local result="$1"
  [[ -f "$result" ]] || return 1
  "$PYTHON" - "$result" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"before_validation", "after_validation", "acceptance", "selected_epoch"}
if not required.issubset(payload):
    raise SystemExit(1)
PY
}

cd "$ROOT"
for mode in supervised_control grounding; do
  for seed in 17 23 31; do
    run_dir="$OUTPUT/$mode/$FAMILY_DIR/seed_$seed"
    result="$run_dir/mitigation_results.json"
    log="$OUTPUT/logs/${mode}_${FAMILY_DIR}_seed_${seed}.log"
    mkdir -p "$run_dir" "$(dirname "$log")"
    if is_complete "$result"; then
      echo "[resume-skip] family=$FAMILY mode=$mode seed=$seed"
      continue
    fi
    echo "[start] family=$FAMILY mode=$mode seed=$seed gpu=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" \
      "$PYTHON" -m sgg_core.experiments.experiment_5 \
        --manifest "$MANIFEST" \
        --dataset vg \
        --output_dir "$run_dir" \
        --epochs "${MITIGATION_EPOCHS:-5}" \
        --minimum_epochs "${MITIGATION_MINIMUM_EPOCHS:-3}" \
        --early_stopping_patience "${MITIGATION_EARLY_STOPPING_PATIENCE:-1}" \
        --train_samples "${MITIGATION_TRAIN_SAMPLES:-5000}" \
        --eval_samples "${MITIGATION_EVAL_SAMPLES:-1000}" \
        --test_samples "${MITIGATION_TEST_SAMPLES:-26446}" \
        --seed "$seed" \
        --training_mode "$mode" \
        --gradient_accumulation_steps \
          "${GRADIENT_ACCUMULATION_STEPS:-4}" \
        --device cuda \
        --data_root "$ROOT/data/vg/v1.4" \
      >"$log" 2>&1
    is_complete "$result"
    echo "[complete] family=$FAMILY mode=$mode seed=$seed"
  done
done
