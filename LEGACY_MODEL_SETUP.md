# Causal Motifs-SUM and KERN

## Official Causal Motifs-SUM weights

- SGDet: `https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602`
- SGCls: `https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s`
- PredCls: `https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws`

Place the checkpoints at:

```text
checkpoints/sgg/weights/causal_motifs_sum/vg/predcls/model_0030000.pth
checkpoints/sgg/weights/causal_motifs_sum/vg/sgcls/model_final.pth
checkpoints/sgg/weights/causal_motifs_sum/vg/sgdet/model_0028000.pth
```

Check all legacy assets before GPU work:

```bash
source scripts/project_env.sh
"$SGG_PYTHON" scripts/check_legacy_vg_assets.py \
  --project_root "$SGG_PROJECT_ROOT"
```

The Causal repository runs through its own compiled maskrcnn-benchmark package.
The setup reuses the existing PySGG legacy Python but prepends a float32-only
Apex compatibility layer. Do not load these weights into PySGG's
`MotifPredictor`.

After preparing that environment:

```bash
source scripts/project_env.sh
export KAIHUA_SGG_PYTHON=python3
bash scripts/setup_kaihua_runtime.sh
CUDA_VISIBLE_DEVICES=0 CAUSAL_EFFECT_TYPE=none CAUSAL_EVAL_SAMPLES=20 \
  bash scripts/run_causal_motifs_vg_export.sh predcls
CUDA_VISIBLE_DEVICES=0 CAUSAL_EFFECT_TYPE=none \
  bash scripts/run_causal_motifs_vg_export.sh predcls sgcls sgdet
CUDA_VISIBLE_DEVICES=0 CAUSAL_EFFECT_TYPE=tde \
  bash scripts/run_causal_motifs_vg_export.sh predcls sgcls sgdet
```

## KERN without CUDA 9 inference

KERN's two checkpoint files are old `torch.save` payloads despite their `.tar`
suffix. Do not extract them. The official repository also publishes SGCls and
SGDet prediction caches, which avoids running CUDA 9 code on RTX 3090.

- SGCls cache (1,074,558,011 bytes): `https://drive.google.com/file/d/1yY0bb2zPJZC3lumK1mQWSQ0NM0WMSFUK/view`
- SGDet cache (6,624,342,381 bytes): `https://drive.google.com/file/d/1Tvxf0OCjRKut8m_iNDgtcz_PNfIQ3Let/view`

```bash
source scripts/project_env.sh
bash scripts/download_kern_official_caches.sh
bash scripts/import_kern_vg_predictions.sh
```

The official repository does not expose a direct PredCls cache. Therefore the
formal KERN row supports SGCls and SGDet unless the native PredCls runtime is
successfully ported and reproduced.

Every import performs exact 151-object/51-predicate order validation, writes
one `.npz` per image and task, finalizes `metadata.json`, and registers a strict
prediction-cache manifest.
