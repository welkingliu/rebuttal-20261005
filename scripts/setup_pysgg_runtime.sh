#!/usr/bin/env bash
set -euo pipefail

ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${CONDA:-conda}"
SOURCE_ENV="${PYSGG_SOURCE_ENV:-$ROOT/.runtimes/openpsg_runtime}"
ENV_PREFIX="${PYSGG_ENV:-$ROOT/.runtimes/pysgg_runtime}"
REPO="$ROOT/external/official_repos/PySGG"
SYSTEM_CC="${PYSGG_CC:-/usr/bin/gcc-10}"
SYSTEM_CXX="${PYSGG_CXX:-/usr/bin/g++-10}"
CUDA_VERSION="${PYSGG_CUDA_VERSION:-11.3}"
CUDA_CHANNEL="${PYSGG_CUDA_CHANNEL:-nvidia/label/cuda-11.3.1}"

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  source_cuda="$($SOURCE_ENV/bin/python -c 'import torch; print(torch.version.cuda)')"
  target_cuda="$($ENV_PREFIX/bin/python -c 'import torch; print(torch.version.cuda)')"
  if [[ "$source_cuda" != "$target_cuda" ]]; then
    if [[ "${PYSGG_RECREATE:-0}" != "1" ]]; then
      echo "[mismatch] target torch CUDA=$target_cuda source torch CUDA=$source_cuda" >&2
      echo "Rerun once with PYSGG_RECREATE=1 to replace the incomplete target env." >&2
      exit 2
    fi
    "$CONDA" remove -y -p "$ENV_PREFIX" --all
  fi
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" --clone "$SOURCE_ENV"
fi

if [[ ! -x "$SYSTEM_CC" || ! -x "$SYSTEM_CXX" ]]; then
  cat >&2 <<EOF
[missing] PySGG requires GCC/G++ 10 to compile its legacy CUDA extension.
Install them once with:
  sudo apt-get update
  sudo apt-get install -y gcc-10 g++-10
EOF
  exit 2
fi

# Keep the compiler outside Conda. Installing gcc_linux-64 together with a
# cloned environment can force mutually incompatible binutils/ld builds.
"$CONDA" install -y -p "$ENV_PREFIX" \
  -c "$CUDA_CHANNEL" -c conda-forge \
  "cuda-nvcc=$CUDA_VERSION" ninja

"$ENV_PREFIX/bin/python" -m pip install \
  'numpy==1.23.5' 'scipy==1.10.1' 'opencv-python==4.8.1.78' \
  yacs cython matplotlib tqdm overrides gpustat gitpython \
  ipdb graphviz tensorboardx termcolor scikit-learn

export CUDA_HOME="$ENV_PREFIX"
export CC="$SYSTEM_CC"
export CXX="$SYSTEM_CXX"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONPATH="$ROOT/legacy_runtime:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYSGG_EXPECTED_CUDA="$CUDA_VERSION"

"$ENV_PREFIX/bin/python" - <<'PY'
import subprocess
import torch

nvcc = subprocess.check_output(
    [str(__import__("os").environ["CUDA_HOME"] + "/bin/nvcc"), "--version"],
    text=True,
)
expected = __import__("os").environ["PYSGG_EXPECTED_CUDA"]
if torch.version.cuda != expected or ("release " + expected) not in nvcc:
    raise RuntimeError(
        f"CUDA toolchain mismatch: torch={torch.version.cuda!r}, nvcc={nvcc!r}"
    )
print(f"[toolchain] torch={torch.__version__} cuda={torch.version.cuda}")
PY

# PySGG's pinned CUDA sources predate the AT_CHECK -> TORCH_CHECK rename.
# This mechanical compatibility patch changes assertion spelling only.
for source in \
  "$REPO/pysgg/csrc/cuda/deform_conv_cuda.cu" \
  "$REPO/pysgg/csrc/cuda/deform_pool_cuda.cu"; do
  if grep -q 'AT_CHECK' "$source"; then
    sed -i 's/AT_CHECK/TORCH_CHECK/g' "$source"
    echo "[compat] AT_CHECK -> TORCH_CHECK: $source"
  fi
done
if grep -R -n 'AT_CHECK' "$REPO/pysgg/csrc"; then
  echo "[failed] unresolved AT_CHECK compatibility sites" >&2
  exit 2
fi
for source in \
  "$REPO/pysgg/utils/imports.py" \
  "$REPO/pysgg/utils/c2_model_loading.py"; do
  if grep -q 'torch\._six\.PY3' "$source"; then
    sed -i 's/torch\._six\.PY3/True/g' "$source"
    echo "[compat] fixed Python-3 guard: $source"
  fi
done
if grep -R -n 'torch\._six\.PY3' "$REPO/pysgg"; then
  echo "[failed] unresolved torch._six.PY3 compatibility sites" >&2
  exit 2
fi
VG_DATASET="$REPO/pysgg/data/datasets/visual_genome.py"
if grep -q 'if os.path.exists(filename) or not check_img_file:' "$VG_DATASET"; then
  sed -i \
    's/if os.path.exists(filename) or not check_img_file:/if not check_img_file or os.path.exists(filename):/' \
    "$VG_DATASET"
  echo "[compat] short-circuit disabled VG image checks: $VG_DATASET"
fi

cd "$REPO"
rm -rf build
"$ENV_PREFIX/bin/python" setup.py build_ext --inplace
"$ENV_PREFIX/bin/python" -c \
  'import torch, pysgg, pysgg._C; from apex import amp; print(torch.__version__, torch.version.cuda, pysgg._C.__file__)'

echo "[ready] PYSGG_PYTHON=$ENV_PREFIX/bin/python"
