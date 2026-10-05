#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

ACTION="${1:-preflight}"
MASTER_ID="${PAPER_RUN_ID:-paper_$(date +%Y%m%d_%H%M%S)}"
STATE_ROOT="$SGG_ARTIFACT_DIR/paper_reproduction/$MASTER_ID"
mkdir -p "$STATE_ROOT" "$SGG_LOG_DIR"

stage_enabled() {
  local stage="$1"
  local selected="${PAPER_STAGES:-1a 1a_external 1b 2a 2b 3 4_native 4_depth 5 5_posthoc}"
  [[ " $selected " == *" $stage "* ]]
}

run_stage() {
  local stage="$1"
  shift
  local marker="$STATE_ROOT/$stage.complete"
  if [[ "${RESUME:-1}" == "1" && -s "$marker" ]]; then
    printf '[resume] stage=%s marker=%s\n' "$stage" "$marker"
    return
  fi
  printf '[start] stage=%s time=%s\n' "$stage" "$(date -Iseconds)"
  "$@"
  printf 'stage=%s\ncompleted_at=%s\n' "$stage" "$(date -Iseconds)" > "$marker"
  printf '[complete] stage=%s time=%s\n' "$stage" "$(date -Iseconds)"
}

prepare_shared_assets() {
  "$SGG_PYTHON" "$SCRIPT_DIR/generate_pysgg_vg_tritask_configs.py" \
    --project_root "$SGG_PROJECT_ROOT"
  if [[ ! -s "$SGG_MANIFEST_DIR/seen_triplets_full.json" ]]; then
    "$SGG_PYTHON" -m sgg_core.tools.build_seen_triplets \
      --datasets vg oi psg \
      --vg_root "$SGG_VG_ROOT" \
      --oi_root "$SGG_OI_ROOT" \
      --psg_train_ann "$SGG_PSG_TRAIN_JSON" \
      --psg_eval_ann "$SGG_PSG_EVAL_JSON" \
      --max_images 1000000000 \
      --output "$SGG_MANIFEST_DIR/seen_triplets_full.json"
  fi
  if [[ ! -s "$SGG_OFFICIAL_MANIFEST_DIR/pysgg_tde_motifs_vg_live.json" ]]; then
    "$SGG_PYTHON" "$SCRIPT_DIR/register_tde_motifs_live_manifest.py" \
      --project_root "$SGG_PROJECT_ROOT"
  fi
}

case "$ACTION" in
  preflight)
    exec "$SGG_PYTHON" "$SCRIPT_DIR/preflight_paper_reproduction.py" \
      --project_root "$SGG_PROJECT_ROOT" \
      --report "$STATE_ROOT/preflight.json" \
      --strict
    ;;
  smoke)
    "$SGG_PYTHON" "$SCRIPT_DIR/preflight_paper_reproduction.py" \
      --project_root "$SGG_PROJECT_ROOT" \
      --report "$STATE_ROOT/preflight.json"
    "$SGG_PYTHON" -m compileall -q sgg_core scripts
    "$SGG_PYTHON" -m unittest discover -s tests -p 'test_*.py' -q
    "$SGG_PYTHON" "$SCRIPT_DIR/generate_paper_appendix_tables.py" --check
    bash -n \
      "$SCRIPT_DIR/run_paper_experiment3_2gpu.sh" \
      "$SCRIPT_DIR/run_paper_experiment4_native_2gpu.sh" \
      "$SCRIPT_DIR/run_paper_experiment5_posthoc.sh"
    printf '[complete] smoke test: full unit suite, appendix, imports, shell syntax\n'
    ;;
  formal)
    "$SGG_PYTHON" "$SCRIPT_DIR/preflight_paper_reproduction.py" \
      --project_root "$SGG_PROJECT_ROOT" \
      --report "$STATE_ROOT/preflight.json" \
      --strict
    prepare_shared_assets

    if stage_enabled 1a; then
      run_stage 1a env \
        RUN_ID="${MASTER_ID}_exp1a" \
        RESUME="${RESUME:-1}" \
        bash "$SCRIPT_DIR/run_experiment1a_panel_2gpu.sh"
    fi
    if stage_enabled 1a_external; then
      run_stage 1a_external env \
        RUN_ID="${MASTER_ID}_exp1a_external" \
        bash "$SCRIPT_DIR/run_experiment1a_external_mac.sh"
    fi
    if stage_enabled 1b; then
      run_stage 1b env \
        RUN_ID="${MASTER_ID}_exp1b" \
        bash "$SCRIPT_DIR/run_foundation_panel_2gpu.sh"
    fi
    if stage_enabled 2a; then
      run_stage 2a env \
        RUN_ID="${MASTER_ID}_exp2a" \
        bash "$SCRIPT_DIR/run_experiment2_observational_2gpu.sh"
    fi
    if stage_enabled 2b; then
      run_stage 2b env \
        RUN_ID="${MASTER_ID}_exp2b" \
        bash "$SCRIPT_DIR/run_experiment2_mandatory_2gpu.sh"
    fi
    if stage_enabled 3; then
      run_stage 3 env \
        RUN_ID="${MASTER_ID}_exp3" \
        RESUME="${RESUME:-1}" \
        bash "$SCRIPT_DIR/run_paper_experiment3_2gpu.sh"
    fi
    if stage_enabled 4_native; then
      run_stage 4_native env \
        RUN_ID="${MASTER_ID}_exp4_native" \
        bash "$SCRIPT_DIR/run_paper_experiment4_native_2gpu.sh"
    fi
    if stage_enabled 4_depth; then
      run_stage 4_depth env \
        RUN_ID="${MASTER_ID}_exp4_depth" \
        bash "$SCRIPT_DIR/run_experiment4_converged_depth_2gpu.sh"
    fi
    exp5_root="$SGG_ARTIFACT_DIR/experiment_5/${MASTER_ID}_exp5"
    if stage_enabled 5; then
      run_stage 5 "$SGG_PYTHON" \
        "$SCRIPT_DIR/run_paper_experiment5_tde_motifs.py" \
        --project_root "$SGG_PROJECT_ROOT" \
        --manifest "$SGG_OFFICIAL_MANIFEST_DIR/pysgg_tde_motifs_vg_live.json" \
        --data_root "$SGG_VG_ROOT" \
        --output_dir "$exp5_root" \
        --gpus 0 1 \
        --resume
    fi
    if stage_enabled 5_posthoc; then
      run_stage 5_posthoc env \
        EXP5_RUN_ROOT="$exp5_root" \
        bash "$SCRIPT_DIR/run_paper_experiment5_posthoc.sh"
    fi
    printf '[complete] full paper reproduction: %s\n' "$STATE_ROOT"
    ;;
  paper)
    "$SGG_PYTHON" "$SCRIPT_DIR/generate_paper_figures.py"
    "$SGG_PYTHON" \
      "$SGG_PROJECT_ROOT/tex/kdd2027_submission/figures/generate_experiment3_motif_intervention.py"
    "$SGG_PYTHON" "$SCRIPT_DIR/generate_paper_appendix_tables.py" --check
    printf '[complete] paper result figures regenerated; curated appendix validated\n'
    ;;
  *)
    printf 'Usage: %s {preflight|smoke|formal|paper}\n' "$0" >&2
    exit 2
    ;;
esac
