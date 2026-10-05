# Vision Foundation Panel

The machine-readable list is
`sgg_core/backbones/foundation_backbone_catalog.json`.

## Main six-backbone panel

These six are intentionally small enough for two RTX 3090 GPUs and cover
different pretraining regimes:

| Backbone | Role | Default input | 3090 policy |
| --- | --- | ---: | --- |
| ResNet-50 | supervised CNN reference | 224 | main |
| DINOv2-B/14 | self-supervised dense baseline | 518 | main |
| SigLIP 2-B/16 | recent vision-language dense model | 224 | main |
| RADIOv2.5-B/16 | multi-teacher foundation model | 512 | main |
| C-RADIOv4-SO400M | 2026 multi-teacher foundation model | 512 | main |
| SAM ViT-B encoder | segmentation-specialized diagnostic backbone | 1024 | main |

Extended appendix backbones are Swin-B, CLIP ViT-L/14, DINOv2-L, DINOv3-B/L,
SigLIP So400m, and OpenCLIP ViT-H/14. Run them one image per GPU; do not run two
large backbones in one process.

The following are controls, not interchangeable feature backbones. The frozen
SAM ViT-B image encoder above is a deliberately isolated diagnostic backbone;
full prompted SAM/SAM2 mask-generation systems remain controls:

| Model | Correct role |
| --- | --- |
| Grounding DINO | open-vocabulary object-box grounding |
| SAM 2.1 | promptable segmentation quality/coverage |
| Florence-2-large | promptable detection and caption control |
| Qwen3-VL-4B-Instruct | generative object/relation audit on one 3090 |

Their outputs require a frozen parser and separate grounding/generative
metrics. Never merge free-text VLM scores into standard VG-150 R/mR values.

Run the six comparable feature encoders for the main object-grounding audit:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
nohup env RUN_ID="$RUN_ID" PSG_SAM_MASK_DIR="$SGG_DERIVED_ROOT/sam_psg" \
  bash scripts/run_experiment1a_panel_2gpu.sh \
  > "artifacts/logs/experiment_1a_${RUN_ID}.log" 2>&1 &
```

Each child writes `artifacts/logs/experiment_1a_<backbone>_<RUN_ID>.log`.
`scripts/run_foundation_panel_2gpu.sh` now runs Experiment I-B, the VG-150
PredCls relation-depth component study.

`scripts/project_env.sh` keeps downloads inside the project by exporting
`TORCH_HOME=checkpoints/foundation/torch_hub` and
`HF_HOME=checkpoints/foundation/huggingface`. On a connected workstation,
populate those two directories, upload them to the same server paths, then run
with `HF_HUB_OFFLINE=1` when the server cannot access Hugging Face. Torch Hub
sources and checkpoints must both be present before using offline mode.

The canonical offline DINOv2-B layout is:

```text
external/foundation_repos/dinov2/hubconf.py
checkpoints/foundation/dinov2/dinov2_vitb14_pretrain.pth
```

The loader uses these local assets before Torch Hub networking. Set
`SGG_OFFLINE=1` to turn a missing local weight into an immediate error. To rerun
only selected failed models without repeating completed models, pass a
space-separated list such as
`FOUNDATION_BACKBONES="cradio_v4_so400m sam_vit_b"`.

## Download and readiness tool

On a connected workstation, download the complete six-model panel into the
canonical project paths and produce one transferable archive:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_foundation_models.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --models resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b \
  --hf_endpoint "$HF_ENDPOINT" \
  --download \
  --smoke_load \
  --device cpu \
  --package ../foundation_models_main.tar
```

The package is intentionally uncompressed by default because safetensors and
released checkpoints gain little from gzip while packaging becomes much
slower. After uploading and extracting it at the server project root, verify
without network access:

```bash
HF_HUB_OFFLINE=1 SGG_OFFLINE=1 \
python scripts/prepare_foundation_models.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --models resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b \
  --check_only \
  --smoke_load \
  --device cuda
```

The script pins both GitHub commits and Hugging Face revisions. DINOv3-B remains
an optional supported model, but is excluded from the default panel because its
gated access is not reproducible for every reviewer.
The default Hugging Face transport endpoint is `https://hf-mirror.com`; this
does not alter repository IDs or pinned revisions.

## Memory contract

All graph loaders use image batch size 1. Experiments I-A/I-B default to ROI chunks
of 512, evaluation pair chunks of 512, FP16 feature caches, CUDA AMP, and four
gradient-accumulation steps. The effective training batch is four images per
optimizer step without padding variable-size scene graphs.
