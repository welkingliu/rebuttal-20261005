# Installation Inventory

Python standard-library modules are intentionally omitted. Official SGG
repositories are mutually incompatible, so there is no scientifically sound
single environment containing every model.

The same family-to-environment mapping is available as
`sgg_core/models/environment_catalog.json` for automated checks.

## Core audit environment

Required Python packages are:

`torch`, `torchvision`, `numpy`, `tqdm`, `Pillow`, `h5py`, `timm`,
`transformers`, `huggingface-hub`, `safetensors`, `einops`, and `gdown`.

For all coded visual backbones plus the optional VLM controls, also install the
official OpenAI CLIP package, `accelerate`, and `qwen-vl-utils`.

```bash
conda create -n sgg-core python=3.10 -y
conda activate sgg-core

# Install the CUDA build of torch/torchvision appropriate for the server driver.
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Do not install the unrelated PyPI package named `clip`; the requirements file
`requirements-foundation.txt` uses OpenAI's official Git repository. Install
that optional file only when running CLIP or Qwen3-VL controls. On a server
without GitHub access, upload an official CLIP source archive to
`external/official_repos/CLIP` and run `python -m pip install -e` on that local
directory instead.

Recommended non-Python tools are `git`, `git-lfs`, `gcc`, `g++`, `make`,
`ninja`, and a CUDA toolkit matching PyTorch when a repository builds custom
CUDA/C++ operators.

## SGG model environments

First fetch the pinned source trees with
`scripts/prepare_official_models.py --repos_only --survey_repositories`. Then
create the following isolated environments. Each manifest's
`environment_python` must point to its own environment.

| Environment | Model families | Installation source |
| --- | --- | --- |
| `sgg-microsoft` | IMP, Neural Motifs, GRCNN, RelDN | `external/official_repos/microsoft_scene_graph_benchmark` requirements and compiled ops |
| `sgg-kaihua` | VCTree, TDE-Motifs | `external/official_repos/Scene-Graph-Benchmark.pytorch` requirements and `setup.py build develop` |
| `sgg-gpsnet` | GPS-Net | official GPS-Net repository requirements |
| `sgg-pysgg` | BGNN | `external/official_repos/PySGG` environment/requirements |
| `sgg-reltr` | RelTR | `external/official_repos/RelTR/requirements.txt` |
| `sgg-sgtr` | SGTR | SGTR requirements plus its pinned cvpods build |
| `sgg-penet` | PE-Net | PENET requirements and maskrcnn benchmark extension |
| `sgg-openpsg` | PSGTR, PSGFormer and OpenPSG ports | OpenPSG's pinned MMCV, MMDetection, Detectron2, and panopticapi stack |
| `sgg-ovsgtr` | OvSGTR | OvSGTR requirements plus its GroundingDINO dependency |
| `sgg-egtr` | EGTR | EGTR requirements plus `lib/fpn/make.sh` |
| `sgg-pgsg` | PGSG | Pix2Grp/LAVIS requirements |
| `sgg-hiker` | HiKER-SGG | HiKER requirements and its custom ops |
| `sgg-llm4sgg` | LLM4SGG | torch-LLM4SGG requirements |
| `sgg-apt` | APT | APT requirements and maskrcnn benchmark extension |

Important official version anchors:

- EGTR documents NVIDIA PyTorch container `21.11-py3` and compiles
  `lib/fpn`.
- PGSG documents Python 3.8, PyTorch 2.0.0, torchvision 0.15.0, CUDA 11.8,
  and Transformers 4.29.2.
- HiKER-SGG documents PyTorch 1.12.0 with CUDA 11.6.
- OpenPSG and the older maskrcnn/cvpods repositories must retain their own old
  MMCV, Detectron2, or CUDA-extension versions.

Do not upgrade these old environments to match `sgg-core`. Export an exact lock
after each successful install:

```bash
conda env export --no-builds > external/environment_locks/<env>.yml
python -m pip freeze > external/environment_locks/<env>.txt
```

## Foundation/control environments

The Experiment I-A/I-B backbones ResNet, Swin, DINOv2, DINOv3, CLIP, SigLIP 2, and
RADIO use `sgg-core`. DINOv3 needs Transformers 4.56 or newer; Qwen3-VL needs
Transformers 4.57 or newer, which is why the core lower bound is 4.57.

Keep compiled controls separate:

```bash
# Clone these official repositories first, or upload pinned source archives to
# the same paths when the server cannot reach GitHub.
git clone https://github.com/IDEA-Research/GroundingDINO.git \
  external/official_repos/GroundingDINO
git clone https://github.com/facebookresearch/sam2.git \
  external/official_repos/sam2

conda create -n sgg-grounding python=3.10 -y
conda activate sgg-grounding
python -m pip install -e external/official_repos/GroundingDINO

conda create -n sgg-sam2 python=3.10 -y
conda activate sgg-sam2
python -m pip install "torch>=2.5.1" "torchvision>=0.20.1"
python -m pip install -e external/official_repos/sam2
```

SAM 2 may compile a CUDA extension. Grounding DINO also compiles custom code.
The main SGG matrix does not require either control environment unless that
control is explicitly reported.

## Verification

```bash
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_official_model_assets.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --survey_repositories
```
