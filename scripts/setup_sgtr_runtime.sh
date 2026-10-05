#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${SGTR_CONDA:-conda}"
ENV_ROOT="${SGTR_ENV_ROOT:-$PROJECT_ROOT/.runtimes/sgtr_runtime}"
PYTHON="$ENV_ROOT/bin/python"
SOURCE_ROOT="$PROJECT_ROOT/external/official_repos/SGTR"
CPU_PATCH="$PROJECT_ROOT/scripts/patches/sgtr_cpu_extension_psroi.patch"
REL_POSTPROCESS="$SOURCE_ROOT/cvpods/modeling/meta_arch/one_stage_sgg/rel_detr_inference.py"
POSTPROCESS_PATCH="$PROJECT_ROOT/scripts/patches/sgtr_postprocess_device.patch"

if [[ ! -x "$PYTHON" ]]; then
  "$CONDA" create -y -p "$ENV_ROOT" python=3.10 pip
fi

if ! "$PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --retries 8 --timeout 120 \
    --index-url https://download.pytorch.org/whl/cu117 \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117
fi

"$PYTHON" -m pip install --retries 8 --timeout 120 \
  'setuptools==80.9.0' wheel

"$PYTHON" -m pip install --retries 8 --timeout 120 \
  'numpy==1.23.5' 'scipy==1.10.1' 'h5py==3.8.0' cython 'pillow==9.5.0' tabulate cloudpickle \
  tqdm shapely portalocker easydict termcolor colorama appdirs ninja yacs \
  'opencv-python-headless==4.8.1.78' overrides tensorboardx \
  scikit-learn pycocotools matplotlib six

# The released file uses CRLF. Normalize it so the compatibility patch remains
# deterministic across archive extraction and file transfer tools.
sed -i 's/\r$//' "$REL_POSTPROCESS"
if ! patch --dry-run --reverse --batch --silent \
    --directory="$SOURCE_ROOT" --strip=1 < "$POSTPROCESS_PATCH"; then
  patch --batch --forward --directory="$SOURCE_ROOT" --strip=1 < "$POSTPROCESS_PATCH"
fi

if ! PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -c 'import cvpods._C' >/dev/null 2>&1; then
  if ! patch --dry-run --reverse --batch --silent \
      --directory="$SOURCE_ROOT" --strip=1 < "$CPU_PATCH"; then
    patch --batch --forward --directory="$SOURCE_ROOT" --strip=1 < "$CPU_PATCH"
  fi
  # SGTR's DETR path does not use CUDA custom ops. A CPU cvpods extension keeps
  # the legacy package importable without coupling it to the host CUDA toolkit.
  (
    cd "$SOURCE_ROOT"
    CUDA_VISIBLE_DEVICES="" MAX_JOBS="${MAX_JOBS:-4}" \
      "$PYTHON" setup.py build_ext --inplace
  )
fi

PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -c \
  'import torch, torchvision, cvpods, cvpods._C, h5py; print(f"[ready] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'
