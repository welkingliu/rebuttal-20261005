#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "" ]; then
  export SGG_PROJECT_ROOT="$1"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/project_env.sh"

PROJECT_ROOT="$SGG_PROJECT_ROOT"

mkdir -p \
  "$PROJECT_ROOT/data" \
  "$PROJECT_ROOT/data/vg" \
  "$PROJECT_ROOT/data/openimages" \
  "$PROJECT_ROOT/data/openimages/open-images-v6/annotations" \
  "$PROJECT_ROOT/data/openimages/open-images-v6/images" \
  "$PROJECT_ROOT/data/openimages/open-images-v6/images/train" \
  "$PROJECT_ROOT/data/openimages/open-images-v6/images/validation" \
  "$PROJECT_ROOT/data/openimages/open-images-v6/manifests" \
  "$PROJECT_ROOT/data/gqa" \
  "$PROJECT_ROOT/data/psg" \
  "$PROJECT_ROOT/data/vrd" \
  "$PROJECT_ROOT/data/coco" \
  "$PROJECT_ROOT/data/derived/features" \
  "$PROJECT_ROOT/data/derived/cache" \
  "$PROJECT_ROOT/data/derived/sam_psg/train/masks" \
  "$PROJECT_ROOT/data/derived/sam_psg/eval/masks" \
  "$PROJECT_ROOT/artifacts/experiment_1" \
  "$PROJECT_ROOT/artifacts/experiment_1a" \
  "$PROJECT_ROOT/artifacts/experiment_1b" \
  "$PROJECT_ROOT/artifacts/experiment_2" \
  "$PROJECT_ROOT/artifacts/experiment_3" \
  "$PROJECT_ROOT/artifacts/experiment_4" \
  "$PROJECT_ROOT/artifacts/experiment_5" \
  "$PROJECT_ROOT/artifacts/logs" \
  "$PROJECT_ROOT/artifacts/manifests" \
  "$PROJECT_ROOT/checkpoints/sgg" \
  "$PROJECT_ROOT/checkpoints/sgg/manifests" \
  "$PROJECT_ROOT/checkpoints/sgg/weights" \
  "$PROJECT_ROOT/checkpoints/sgg/prediction_caches" \
  "$PROJECT_ROOT/checkpoints/foundation" \
  "$PROJECT_ROOT/checkpoints/foundation/torch_hub" \
  "$PROJECT_ROOT/checkpoints/foundation/huggingface/hub" \
  "$PROJECT_ROOT/external/official_repos" \
  "$PROJECT_ROOT/external/official_archives" \
  "$PROJECT_ROOT/external/environment_locks"

echo "Project layout ready under: $PROJECT_ROOT"
echo "SGG checkpoints:          $PROJECT_ROOT/checkpoints/sgg"
echo "Official model manifests: $PROJECT_ROOT/checkpoints/sgg/manifests"
echo "Official source checkouts: $PROJECT_ROOT/external/official_repos"
echo "Experiment artifacts:     $PROJECT_ROOT/artifacts"
