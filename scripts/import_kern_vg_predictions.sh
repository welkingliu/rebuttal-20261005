#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
SOURCE="$ROOT/external/official_repos/KERN"
NATIVE="$ROOT/checkpoints/sgg/native_predictions/kern/vg"
CACHE="$ROOT/artifacts/prediction_cache/kern_vg_official"
WEIGHTS="$ROOT/checkpoints/sgg/weights/kern/vg"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

first_file() {
  for path in "$@"; do
    [[ -f "$path" ]] && { printf '%s\n' "$path"; return 0; }
  done
  return 1
}

VG_H5="$(first_file "$ROOT/data/vg/v1.4/VG-SGG.h5" "$ROOT/data/vg/VG-SGG.h5" "$ROOT/data/vg/VG-SGG-with-attri.h5")"
VG_DICT="$(first_file "$ROOT/data/vg/v1.4/VG-SGG-dicts.json" "$ROOT/data/vg/VG-SGG-dicts.json" "$ROOT/data/vg/VG-SGG-dicts-with-attri.json")"
IMAGE_DATA="$(first_file "$ROOT/data/vg/v1.4/image_data.json" "$ROOT/data/vg/image_data.json")"

"$PYTHON" scripts/validate_vg_ontology_alignment.py \
  --canonical_dict "$VG_DICT" \
  --candidate "$VG_DICT" \
  --report "$ROOT/artifacts/manifests/kern_vg_ontology_alignment.json"

for task in sgcls sgdet; do
  checkpoint="$WEIGHTS/kern_sgcls_predcls.tar"
  [[ "$task" == sgdet ]] && checkpoint="$WEIGHTS/kern_sgdet.tar"
  "$PYTHON" scripts/convert_legacy_vg_predictions.py \
    --format kern \
    --task "$task" \
    --prediction_file "$NATIVE/kern_${task}.pkl" \
    --checkpoint "$checkpoint" \
    --cache_root "$CACHE" \
    --model_name kern_official \
    --family kern \
    --source_root "$SOURCE" \
    --canonical_dict "$VG_DICT" \
    --native_dict "$VG_DICT" \
    --vg_h5 "$VG_H5" \
    --native_vg_h5 "$VG_H5" \
    --image_data "$IMAGE_DATA" \
    --resume
done

"$PYTHON" scripts/finalize_legacy_vg_cache.py \
  --cache_root "$CACHE" --tasks sgcls sgdet

"$PYTHON" scripts/register_legacy_vg_manifest.py \
  --project_root "$ROOT" --model kern --cache_root "$CACHE"

echo "[complete] KERN unified cache=$CACHE"
