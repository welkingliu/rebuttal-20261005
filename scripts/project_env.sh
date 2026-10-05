#!/usr/bin/env bash

# Source this file from experiment command scripts.
# On a server, set SGG_PROJECT_ROOT before sourcing:
#   export SGG_PROJECT_ROOT=/path/to/project

_SGG_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SGG_DEFAULT_PROJECT_ROOT="$(cd "$_SGG_ENV_DIR/.." && pwd)"
export SGG_PROJECT_ROOT="${SGG_PROJECT_ROOT:-$_SGG_DEFAULT_PROJECT_ROOT}"
export SGG_CODE_DIR="$SGG_PROJECT_ROOT/sgg_core"
export SGG_DATA_DIR="$SGG_PROJECT_ROOT/data"
export SGG_DERIVED_ROOT="$SGG_DATA_DIR/derived"
export SGG_ARTIFACT_DIR="$SGG_PROJECT_ROOT/artifacts"
export SGG_LOG_DIR="$SGG_ARTIFACT_DIR/logs"
export SGG_MANIFEST_DIR="$SGG_ARTIFACT_DIR/manifests"
export SGG_CHECKPOINT_DIR="$SGG_PROJECT_ROOT/checkpoints"
export SGG_SGG_CHECKPOINT_DIR="$SGG_CHECKPOINT_DIR/sgg"
export SGG_FOUNDATION_CHECKPOINT_DIR="$SGG_CHECKPOINT_DIR/foundation"
export SGG_DINOV2_REPO="${SGG_DINOV2_REPO:-$SGG_PROJECT_ROOT/external/foundation_repos/dinov2}"
export SGG_DINOV2_B_WEIGHTS="${SGG_DINOV2_B_WEIGHTS:-$SGG_FOUNDATION_CHECKPOINT_DIR/dinov2/dinov2_vitb14_pretrain.pth}"
export SGG_DINOV2_L_WEIGHTS="${SGG_DINOV2_L_WEIGHTS:-$SGG_FOUNDATION_CHECKPOINT_DIR/dinov2/dinov2_vitl14_pretrain.pth}"
export SGG_DINOV3_B_DIR="${SGG_DINOV3_B_DIR:-$SGG_FOUNDATION_CHECKPOINT_DIR/hf_models/dinov3_b}"
export SGG_DINOV3_L_DIR="${SGG_DINOV3_L_DIR:-$SGG_FOUNDATION_CHECKPOINT_DIR/hf_models/dinov3_l}"
export SGG_SIGLIP2_B_DIR="${SGG_SIGLIP2_B_DIR:-$SGG_FOUNDATION_CHECKPOINT_DIR/hf_models/siglip2_b}"
export SGG_CRADIO_V4_SO400M_DIR="${SGG_CRADIO_V4_SO400M_DIR:-$SGG_FOUNDATION_CHECKPOINT_DIR/hf_models/cradio_v4_so400m}"
export SGG_SAM_VIT_B_DIR="${SGG_SAM_VIT_B_DIR:-$SGG_FOUNDATION_CHECKPOINT_DIR/hf_models/sam_vit_b}"
export SGG_RADIO_REPO="${SGG_RADIO_REPO:-$SGG_PROJECT_ROOT/external/foundation_repos/radio}"
export SGG_RADIO_V25_B_WEIGHTS="${SGG_RADIO_V25_B_WEIGHTS:-$SGG_FOUNDATION_CHECKPOINT_DIR/radio/radio-v2.5-b_half.pth.tar}"
export TORCH_HOME="${TORCH_HOME:-$SGG_FOUNDATION_CHECKPOINT_DIR/torch_hub}"
export HF_HOME="${HF_HOME:-$SGG_FOUNDATION_CHECKPOINT_DIR/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export SGG_PYTHON="${SGG_PYTHON:-$SGG_PROJECT_ROOT/.venv/bin/python}"

if [ ! -x "$SGG_PYTHON" ]; then
  _SGG_PARENT_VENV="$(dirname "$SGG_PROJECT_ROOT")/.venv/bin/python"
  if [ -x "$_SGG_PARENT_VENV" ]; then
    SGG_PYTHON="$_SGG_PARENT_VENV"
  else
    SGG_PYTHON="${PYTHON:-python3}"
  fi
fi

sgg_run_dir() {
  local experiment="$1"
  local name="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  printf "%s/%s/%s_%s" "$SGG_ARTIFACT_DIR" "$experiment" "$stamp" "$name"
}

sgg_first_existing() {
  local path
  for path in "$@"; do
    if [ -e "$path" ]; then
      printf "%s" "$path"
      return 0
    fi
  done
  printf "%s" "$1"
  return 0
}

export SGG_GQA_TRAIN_JSON
SGG_GQA_TRAIN_JSON="$(sgg_first_existing \
  "$SGG_DATA_DIR/gqa/train_sceneGraphs.json" \
  "$SGG_DATA_DIR/gqa/sceneGraphs/train_sceneGraphs.json")"

export SGG_GQA_VAL_JSON
SGG_GQA_VAL_JSON="$(sgg_first_existing \
  "$SGG_DATA_DIR/gqa/val_sceneGraphs.json" \
  "$SGG_DATA_DIR/gqa/sceneGraphs/val_sceneGraphs.json")"

export SGG_OI_ROOT
# Keep one canonical Open Images root.  Choosing the first existing parent
# directory made a fresh setup download into data/openimages/ while the formal
# preflight expected data/openimages/open-images-v6/.
SGG_OI_ROOT="${SGG_OI_ROOT:-$SGG_DATA_DIR/openimages/open-images-v6}"

export SGG_VG_ROOT
SGG_VG_ROOT="${SGG_VG_ROOT:-$(sgg_first_existing \
  "$SGG_DATA_DIR/vg/v1.4" \
  "$SGG_DATA_DIR/vg")}" 

export SGG_GQA_IMAGE_ROOT
SGG_GQA_IMAGE_ROOT="${SGG_GQA_IMAGE_ROOT:-$SGG_DATA_DIR/gqa/images}"

export SGG_PSG_TRAIN_JSON
SGG_PSG_TRAIN_JSON="${SGG_PSG_TRAIN_JSON:-$SGG_DATA_DIR/psg/psg_train_val.json}"

export SGG_PSG_EVAL_JSON
SGG_PSG_EVAL_JSON="${SGG_PSG_EVAL_JSON:-$(sgg_first_existing \
  "$SGG_DATA_DIR/psg/psg_val_test.json" \
  "$SGG_DATA_DIR/psg/psg_train_val.json")}" 

export SGG_PSG_IMAGE_ROOT
SGG_PSG_IMAGE_ROOT="${SGG_PSG_IMAGE_ROOT:-$SGG_DATA_DIR/coco}"

export SGG_PSG_PANOPTIC_ROOT
SGG_PSG_PANOPTIC_ROOT="${SGG_PSG_PANOPTIC_ROOT:-$SGG_DATA_DIR/coco}"

export SGG_VRD_ROOT
SGG_VRD_ROOT="${SGG_VRD_ROOT:-$SGG_DATA_DIR/vrd}"

export SGG_OFFICIAL_MANIFEST_DIR
SGG_OFFICIAL_MANIFEST_DIR="${SGG_OFFICIAL_MANIFEST_DIR:-$SGG_SGG_CHECKPOINT_DIR/manifests}"

export PYTHONPATH="$SGG_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
