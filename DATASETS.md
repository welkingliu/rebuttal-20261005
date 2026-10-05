# Five-Dataset Contract

The loaders use five evaluation datasets. COCO images are storage backing for
PSG and are not counted as a sixth SGG dataset.

```text
data/
  vg/v1.4/
    VG-SGG.h5 (or VG-SGG-with-attri.h5)
    VG-SGG-dicts.json (or VG-SGG-dicts-with-attri.json)
    image_data.json
    VG_100K/
  openimages/open-images-v6/
    annotations/class-descriptions-boxable.csv
    annotations/oidv6-train-annotations-vrd.csv
    annotations/oidv6-validation-annotations-vrd.csv
    images/train/
    images/validation/
  gqa/
    train_sceneGraphs.json
    val_sceneGraphs.json
    images/
  psg/
    psg_train_val.json
    psg_val_test.json
  coco/
    train2017/  val2017/
    panoptic_train2017/  panoptic_val2017/
  vrd/json_dataset/
    annotations_train.json
    annotations_test.json
    objects.json
    predicates.json
```

Open Images can be prepared or verified with the bundled downloader:

```bash
source scripts/project_env.sh
python -m sgg_core.tools.download_openimages \
  --oi_root "$SGG_OI_ROOT" \
  --profile reduced_2gpu
```

Verify all five entry points and the exact selected raw images:

```bash
python -m sgg_core.tools.prepare_reviewer_datasets \
  --project_root "$SGG_PROJECT_ROOT" \
  --oi_root "$SGG_OI_ROOT" \
  --datasets vg oi gqa psg vrd \
  --strict_images \
  --main_samples 2000 \
  --vg_train_samples 5000 \
  --vg_val_samples 1000 \
  --vg_test_samples 2000 \
  --external_samples 1000 \
  --verify_image_content
```

Strict PSG readiness includes both RGB images and panoptic PNGs. Strict VRD
readiness also requires the official images under `data/vrd/sg_dataset/` or
`data/vrd/images/`; annotation JSON alone is insufficient for visual grounding.

The canonical Open Images layout is split-aware. Older project releases stored
all JPEGs directly under `images/`; the downloader, loader, and preflight retain
read-only compatibility with that flat layout, so existing data does not need
to be downloaded again. Every newly downloaded image is written to
`images/<split>/`.

Formal runs must use separate training and evaluation annotations. GQA uses the
training vocabulary for validation. PSG excludes evaluation image IDs from its
training loader. VG uses split 0 for mining/training and split 2 for evaluation.
VRD uses its official train/test JSON files.

Experiment I-A also requires PSG panoptic masks and independent SAM caches:

```text
data/derived/sam_psg/
  train/manifest.json
  train/masks/*.npz
  eval/manifest.json
  eval/masks/*.npz
```

The cache writer stores segment IDs, and Experiment I-A rejects a cache whose
annotation path, segment order, or image coverage differs from the requested
split. See `RUNNING.md` for the two generation commands.
