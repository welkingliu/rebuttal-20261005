#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA="${OPENPSG_CONDA:-conda}"
ENV_ROOT="${OPENPSG_ENV_ROOT:-$PROJECT_ROOT/.runtimes/openpsg_runtime}"
PYTHON="$ENV_ROOT/bin/python"
SOURCE_ROOT="$PROJECT_ROOT/external/official_repos/OpenPSG"

if [[ ! -x "$PYTHON" ]]; then
  "$CONDA" create -y -p "$ENV_ROOT" python=3.8 pip
fi

"$PYTHON" -m pip install --retries 8 --timeout 120 \
  'pip==23.3.2' 'setuptools==59.5.0' wheel

if ! "$PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --retries 8 --timeout 120 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    'torch==1.10.0+cu113' 'torchvision==0.11.1+cu113'
fi

if ! "$PYTHON" -c 'import mmcv; assert mmcv.__version__ == "1.4.3"' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --retries 8 --timeout 120 \
    'mmcv-full==1.4.3' \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
fi

"$PYTHON" -m pip install --retries 8 --timeout 120 \
  'numpy==1.21.6' 'scipy==1.7.3' 'h5py==3.7.0' \
  'pillow==9.5.0' 'opencv-python-headless==4.5.5.64' \
  'mmdet==2.21.0' 'yapf==0.32.0' cython matplotlib pycocotools six \
  terminaltables graphviz xmltodict timm==0.6.13 scikit-learn tqdm

# mmcv declares opencv-python without an upper bound. Its current wheel can
# leave OpenCV 5 Python files beside the pinned 4.5 headless binary, producing
# a mixed cv2 installation. Keep exactly one OpenCV distribution.
"$PYTHON" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
"$PYTHON" -m pip install --force-reinstall --no-deps \
  'opencv-python-headless==4.5.5.64'

if ! "$PYTHON" -c 'import detectron2' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --retries 8 --timeout 120 \
    'detectron2==0.6' \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
fi

if ! "$PYTHON" -c 'from panopticapi.utils import rgb2id' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --retries 8 --timeout 120 \
    'https://codeload.github.com/cocodataset/panopticapi/zip/refs/heads/master'
fi

if ! PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -c 'import openpsg' >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-deps -e "$SOURCE_ROOT"
fi

PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -c '
import torch, torchvision, mmcv, mmdet, detectron2, openpsg
from mmcv.ops import batched_nms
from panopticapi.utils import rgb2id
print(
    f"[ready] torch={torch.__version__} torchvision={torchvision.__version__} "
    f"mmcv={mmcv.__version__} mmdet={mmdet.__version__} "
    f"cuda={torch.version.cuda} available={torch.cuda.is_available()}"
)
'
