#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

: "${EXP5_RUN_ROOT:?Set EXP5_RUN_ROOT to a completed TDE-Motifs matrix directory}"
RUN_ROOT="$(cd "$EXP5_RUN_ROOT" && pwd)"
MANIFEST="${EXP5_MANIFEST:-$SGG_OFFICIAL_MANIFEST_DIR/pysgg_tde_motifs_vg_live.json}"
FAMILY_KEY="TDE_Motifs"
SELECTED_MODE="${EXP5_SELECTED_MODE:-supervised_control}"
SELECTED_SEED="${EXP5_SELECTED_SEED:-17}"
SELECTED_STATE="$RUN_ROOT/$SELECTED_MODE/$FAMILY_KEY/seed_$SELECTED_SEED/mitigated_state_dict.pth"
POSTHOC="$RUN_ROOT/posthoc"

if [[ ! -s "$SELECTED_STATE" ]]; then
  printf 'Selected Experiment V state is missing: %s\n' "$SELECTED_STATE" >&2
  exit 2
fi
mkdir -p "$POSTHOC/test" "$POSTHOC/external" "$POSTHOC/logs"

run_test() {
  local key="$1"
  shift
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$SGG_PYTHON" "$SCRIPT_DIR/run_experiment5_checkpoint_test.py" \
      --manifest "$MANIFEST" \
      --data_root "$SGG_VG_ROOT" \
      --output "$POSTHOC/test/$key.json" \
      --train_samples 5000 \
      --test_samples 26446 \
      --tasks predcls sgcls sgdet \
      --device cuda \
      "$@" 2>&1 |
    sed -u "s/^/[gpu=0 test=$key] /" |
    tee "$POSTHOC/logs/test_$key.log"
}

if [[ ! -s "$POSTHOC/test/baseline.json" ]]; then
  run_test baseline
fi
if [[ ! -s "$POSTHOC/test/selected.json" ]]; then
  run_test selected --checkpoint "$SELECTED_STATE"
fi

run_external_dataset() {
  local gpu="$1"
  local dataset="$2"
  local cache="$SGG_DERIVED_ROOT/predictions/experiment_5_external/$dataset/pysgg_tde_motifs_vg_live"
  local dataset_output="$POSTHOC/external/$dataset"
  local base_output="$dataset_output/base"
  local common=(
    --manifest "$MANIFEST"
    --dataset "$dataset"
    --vg_dict "$SGG_VG_ROOT/VG-SGG-dicts.json"
    --cache_dir "$cache"
    --eval_samples 0
    --device cuda
  )
  local dataset_args=()
  if [[ "$dataset" == "gqa" ]]; then
    dataset_args=(
      --eval_ann "$SGG_GQA_VAL_JSON"
      --image_root "$SGG_GQA_IMAGE_ROOT"
    )
  else
    dataset_args=(--data_root "$SGG_VRD_ROOT")
  fi

  mkdir -p "$dataset_output"
  if [[ ! -s "$base_output/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      "$SGG_PYTHON" -m sgg_core.experiments.experiment_5_external \
        "${common[@]}" "${dataset_args[@]}" \
        --output_dir "$base_output" 2>&1 |
      sed -u "s/^/[gpu=$gpu external=$dataset base] /" |
      tee "$POSTHOC/logs/external_${dataset}_base.log"
  fi

  local mode seed state output
  for mode in supervised_control grounding; do
    for seed in 17 23 31; do
      state="$RUN_ROOT/$mode/$FAMILY_KEY/seed_$seed/mitigated_state_dict.pth"
      output="$dataset_output/${mode}_seed_$seed"
      if [[ -s "$output/summary.json" ]]; then
        printf '[resume] external=%s mode=%s seed=%s\n' "$dataset" "$mode" "$seed"
        continue
      fi
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$SGG_PYTHON" -m sgg_core.experiments.experiment_5_external \
          "${common[@]}" "${dataset_args[@]}" \
          --state "$state" \
          --seed "$seed" \
          --output_dir "$output" 2>&1 |
        sed -u "s/^/[gpu=$gpu external=$dataset mode=$mode seed=$seed] /" |
        tee "$POSTHOC/logs/external_${dataset}_${mode}_${seed}.log"
      printf '[complete] Experiment V %s %s seed %s 外部实验结束\n' \
        "$dataset" "$mode" "$seed"
    done
  done
}

run_external_dataset 0 gqa &
pid0=$!
run_external_dataset 1 vrd &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if ((status != 0)); then
  exit "$status"
fi

"$SGG_PYTHON" "$SCRIPT_DIR/validate_paper_experiment5_posthoc.py" \
  --posthoc_root "$POSTHOC"
printf '[complete] Experiment V full test and external transfer: %s\n' "$POSTHOC"
