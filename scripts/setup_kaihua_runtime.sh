#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${KAIHUA_SGG_PYTHON:-python3}"
SOURCE="$ROOT/external/official_repos/Scene-Graph-Benchmark.pytorch"
CUDA_HOME="${KAIHUA_CUDA_HOME:-$(cd "$(dirname "$PYTHON")/.." && pwd)}"
CC="${KAIHUA_CC:-/usr/bin/gcc-10}"
CXX="${KAIHUA_CXX:-/usr/bin/g++-10}"
cd "$ROOT"

[[ -x "$PYTHON" ]] || { echo "Missing runtime Python: $PYTHON" >&2; exit 1; }
[[ -f "$SOURCE/setup.py" ]] || { echo "Missing pinned Kaihua source: $SOURCE" >&2; exit 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "Missing CUDA nvcc: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
[[ -x "$CC" && -x "$CXX" ]] || { echo "Missing GCC/G++ 10" >&2; exit 1; }

"$PYTHON" -m pip install \
  'yacs>=0.1.8' 'ninja>=1.10' 'cython<3' 'tqdm>=4.60' \
  'opencv-python-headless<5' 'pandas<3' 'dill>=0.3.8,<1'

export CUDA_HOME CC CXX
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONPATH="$ROOT/legacy_runtime:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
import os
import subprocess
import torch

nvcc = subprocess.check_output(
    [os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"],
    text=True,
)
if torch.version.cuda != "11.3" or "release 11.3" not in nvcc:
    raise RuntimeError(
        "Kaihua toolchain mismatch: torch={} nvcc={}".format(
            torch.version.cuda, nvcc.strip()
        )
    )
print("[toolchain] torch={} cuda={}".format(torch.__version__, torch.version.cuda))
PY

for source in \
  "$SOURCE/maskrcnn_benchmark/csrc/cuda/deform_conv_cuda.cu" \
  "$SOURCE/maskrcnn_benchmark/csrc/cuda/deform_pool_cuda.cu"; do
  if grep -q 'AT_CHECK' "$source"; then
    sed -i 's/AT_CHECK/TORCH_CHECK/g' "$source"
    echo "[compat] AT_CHECK -> TORCH_CHECK: $source"
  fi
done
for source in \
  "$SOURCE/maskrcnn_benchmark/utils/imports.py" \
  "$SOURCE/maskrcnn_benchmark/utils/c2_model_loading.py"; do
  if grep -q 'torch\._six\.PY3' "$source"; then
    sed -i 's/torch\._six\.PY3/True/g' "$source"
    echo "[compat] fixed Python-3 guard: $source"
  fi
done

(
  cd "$SOURCE"
  rm -f maskrcnn_benchmark/_C*.so
  rm -rf build
  "$PYTHON" setup.py build_ext --inplace
)

"$PYTHON" - <<'PY'
import torch
from apex import amp
import maskrcnn_benchmark._C as C
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
assert cfg is not None and callable(build_detection_model)
assert callable(C.nms) and callable(C.roi_align_forward)
amp.init(enabled=False)
print("[ok] Kaihua source, compiled operators and float32 Apex shim import")
PY

echo "[complete] KAIHUA_SGG_PYTHON=$PYTHON"
