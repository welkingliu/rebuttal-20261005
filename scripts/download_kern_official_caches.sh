#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${SGG_PYTHON:-python3}"
OUT="$ROOT/checkpoints/sgg/native_predictions/kern/vg"
cd "$ROOT"
mkdir -p "$OUT"

"$PYTHON" -c 'import gdown' >/dev/null 2>&1 || {
  echo "gdown is required: $PYTHON -m pip install 'gdown>=5.2,<6'" >&2
  exit 1
}

download() {
  local id="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    echo "[ok] $output"
    return
  fi
  "$PYTHON" -m gdown "$id" -O "$output.part"
  mv "$output.part" "$output"
}

# Official prediction caches published by the KERN repository setup script.
download 1yY0bb2zPJZC3lumK1mQWSQ0NM0WMSFUK "$OUT/kern_sgcls.pkl"
download 1Tvxf0OCjRKut8m_iNDgtcz_PNfIQ3Let "$OUT/kern_sgdet.pkl"

echo "[complete] Official KERN SGCls/SGDet native caches are ready in $OUT"
echo "[note] The official repository does not publish a direct PredCls cache."
