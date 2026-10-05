# Asset Setup and Deployment

Legacy Causal Motifs-SUM and KERN checkpoints, native-cache conversion, and
strict VG-150 ontology alignment are documented in `LEGACY_MODEL_SETUP.md`.

## Core environment

```bash
conda create -n sgg_proj python=3.10 -y
conda activate sgg_proj
python -m pip install -r requirements.txt
python -m pip install -e .
source scripts/project_env.sh
```

Legacy official repositories need isolated environments. Do not install their
mutually incompatible CUDA extensions into `sgg_proj`.

## Datasets

Place the five datasets under `data/` as documented in `DATASETS.md`, then run:

```bash
"$SGG_PYTHON" -m sgg_core.tools.prepare_reviewer_datasets \
  --project_root "$SGG_PROJECT_ROOT" --oi_root "$SGG_OI_ROOT" \
  --datasets vg oi gqa psg vrd --strict_images \
  --main_samples 2000 --vg_train_samples 5000 --vg_val_samples 1000 \
  --vg_test_samples 2000 \
  --external_samples 1000 --verify_image_content
```

## Six foundation models

Use the Hugging Face mirror without putting a token in a script:

```bash
export HF_ENDPOINT=https://hf-mirror.com
# Export HF_TOKEN only in the interactive shell when a repository needs it.

"$SGG_PYTHON" scripts/prepare_foundation_models.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --models resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b \
  --hf_endpoint "$HF_ENDPOINT" --download --smoke_load --device cpu
```

Server-side offline verification:

```bash
HF_HUB_OFFLINE=1 SGG_OFFLINE=1 \
"$SGG_PYTHON" scripts/prepare_foundation_models.py \
  --project_root "$SGG_PROJECT_ROOT" \
  --models resnet50 dinov2_b siglip2_b radio_v25_b cradio_v4_so400m sam_vit_b \
  --check_only --smoke_load --device cuda
```

## Official SGG weights and sources

The catalog contains 16 submission assets from RelTR, EGTR, PSGTR, PGSG,
OvSGTR, BGNN, SGTR, Neural Motifs, VCTree, and PSGFormer. Download automatic
assets and verify all files:

```bash
"$SGG_PYTHON" scripts/prepare_official_models.py \
  --project_root "$SGG_PROJECT_ROOT" --weights_only

"$SGG_PYTHON" scripts/extract_official_weight_archives.py \
  --project_root "$SGG_PROJECT_ROOT"

"$SGG_PYTHON" scripts/check_official_integration.py \
  --project_root "$SGG_PROJECT_ROOT"
```

BGNN, SGTR, and the OpenPSG Motifs/VCTree/PSGFormer files are browser/manual
downloads in the catalog. Their exact destination paths are printed by:

```bash
"$SGG_PYTHON" scripts/prepare_official_models.py \
  --project_root "$SGG_PROJECT_ROOT" --weights_only --verify_only \
  --models bgnn_vg bgnn_oi sgtr_vg sgtr_oi \
    openpsg_motifs_psg openpsg_vctree_psg openpsg_psgformer_psg
```

Fetch pinned source archives on a connected machine:

```bash
"$SGG_PYTHON" scripts/prepare_official_models.py \
  --project_root "$SGG_PROJECT_ROOT" --repos_only \
  --repositories microsoft_sgg reltr egtr pgsg ovsgtr kaihua_sgg pysgg \
  --repository_transport archive
```

Upload the resulting `external/official_archives/` and
`checkpoints/sgg/weights/` directories to the same relative server paths. For
example, from the Mac:

```bash
rsync -avP external/official_archives/ \
  user@SERVER:/path/to/kdd_sgg_core_experiments/external/official_archives/
rsync -avP checkpoints/sgg/weights/ \
  user@SERVER:/path/to/kdd_sgg_core_experiments/checkpoints/sgg/weights/
```

Then materialize archives without network access:

```bash
"$SGG_PYTHON" scripts/prepare_official_models.py \
  --project_root "$SGG_PROJECT_ROOT" --repos_only \
  --repositories microsoft_sgg reltr egtr pgsg ovsgtr kaihua_sgg pysgg \
  --repository_transport local
```

## Integration modes

Having a `.pth`/`.ckpt` file is only the download layer. It does not make a
model runnable in this benchmark. Readiness has four distinct levels:

| Level | Required evidence | Enables |
| --- | --- | --- |
| Downloaded | Runtime checkpoint and companion config | Nothing by itself |
| Standard | Pinned source or complete official prediction cache, exact ontology, manifest, reference reproduction | Experiment IV |
| Diagnostic | Live adapter plus validated visual/union/node interventions | Experiments II and III |
| Trainable | Diagnostic adapter plus differentiable GT-aligned object/relation logits | Experiment V |

Never promote a model to a higher level by setting manifest flags before the
corresponding runtime behavior has been tested.

### Live adapter

Required for Experiments II-B and V. The optional Experiment III may reuse the
same adapters. The factory contract is in
`sgg_core/models/OFFICIAL_FACTORY_CONTRACT.md`. Each adapter must run inside
the official repository's environment and declare exact task, ontology,
checkpoint SHA, source commit, and intervention support.

### Official prediction cache

Useful for Experiment IV when a legacy environment cannot import into the core
process. In the official environment, use
`OfficialPredictionCacheWriter` from
`sgg_core/models/prediction_cache_writer.py` while iterating the exact official
test loader. Export one NPZ per image/task, then finalize `metadata.json`.

Validate complete coverage in the core environment:

```bash
"$SGG_PYTHON" -m sgg_core.tools.validate_prediction_cache \
  --cache_root /path/to/cache --dataset vg \
  --data_root "$SGG_VG_ROOT"
```

Create the strict manifest after recording a reproduced official metric:

```bash
"$SGG_PYTHON" scripts/create_prediction_cache_manifest.py \
  --cache_root /path/to/cache \
  --name motifs_vg_official --family "Neural Motifs" \
  --paradigm sequential_context \
  --source_url https://github.com/OWNER/REPO \
  --source_root /absolute/pinned/repo --source_commit FULL_COMMIT \
  --training_dataset VG-150 --reference_dataset vg \
  --checkpoint sgdet=/absolute/model_final.pth \
  --reference_metric SGDet/R@50=0.217 \
  --output "$SGG_OFFICIAL_MANIFEST_DIR/motifs_vg.json"
```

Prediction caches cannot be counted for perturbation or mitigation. Do not set
those manifest flags to true.

### OpenPSG Experiment-IV panel

OpenPSG uses an isolated Python 3.8 runtime because its released checkpoints
require Torch 1.10, MMCV 1.4.3, and MMDetection 2.21:

```bash
bash scripts/setup_openpsg_runtime.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_openpsg_smoke_panel.sh
```

After all four smoke caches pass validation, launch the complete PSG test split:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash scripts/run_openpsg_experiment4_panel.sh \
  > artifacts/logs/openpsg_experiment4_panel.log 2>&1 &

tail -n 200 -f artifacts/logs/openpsg_experiment4_panel.log
```

PSGTR and PSGFormer are exported through the official non-panoptic branch so
their relation indices and boxes share one entity axis for unified box-IoU
SGDet. Published OpenPSG references use panoptic-mask IoU and remain a separate
native protocol. Motifs and VCTree checkpoints already contain all GloVe
embedding parameters; the strict exporter uses shape-only placeholders during
construction and requires the checkpoint to replace them before inference.

## Current hard gate

The downloaded 16-checkpoint inventory declares SGDet only. Formal VG task
coverage additionally needs at least four verified PredCls and four SGCls runs.
Use official task-specific checkpoints if available. Otherwise train those
official configurations and report three training seeds for newly trained
models. Relabeling an SGDet output as PredCls/SGCls is invalid.

The submission does not need ten executable families. The literature taxonomy
still covers at least ten, while the converged executable contract separates a
five-family SGDet breadth panel from a two-family matched VG depth panel.
Integrate in this order:

1. Build at least five Experiment-IV SGDet manifests, using official prediction caches
   when importing a legacy repository is impractical.
2. Complete PredCls, SGCls, and SGDet for Neural Motifs and SGG Transformer on
   the full VG test split.
3. Use those same two families for live diagnostics and trainable grounding.

A live manifest counts for Experiment II-B only when `diagnostic_contract`
declares `gt_pair_predict=true` and every required perturbation changes the
input actually consumed by `predict`. Loading a checkpoint or supporting SGDet
inference alone is not sufficient. Experiment III remains optional.

Finally run:

```bash
"$SGG_PYTHON" scripts/check_kdd_readiness.py \
  --project_root "$SGG_PROJECT_ROOT" --strict
```
