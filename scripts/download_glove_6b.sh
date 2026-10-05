#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOWNLOAD="$ROOT/data/downloads/glove.6B.zip"
TARGET="$ROOT/data/derived/glove/glove.6B.200d.txt"

mkdir -p "$(dirname "$DOWNLOAD")" "$(dirname "$TARGET")"
if [[ ! -s "$TARGET" ]]; then
  curl --fail --location --retry 8 --retry-all-errors \
    --connect-timeout 20 --max-time 7200 --continue-at - \
    https://nlp.stanford.edu/data/glove.6B.zip \
    --output "$DOWNLOAD"
  unzip -j -o "$DOWNLOAD" glove.6B.200d.txt -d "$(dirname "$TARGET")"
fi

lines="$(wc -l < "$TARGET")"
columns="$(awk 'NR==1 {print NF; exit}' "$TARGET")"
if [[ "$lines" != "400000" || "$columns" != "201" ]]; then
  echo "[failed] invalid GloVe 6B 200d file: lines=$lines columns=$columns" >&2
  exit 1
fi
sha256sum "$TARGET"
echo "[ready] $TARGET lines=$lines dimensions=200"
