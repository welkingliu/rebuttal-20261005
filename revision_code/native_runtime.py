"""Shared, disk-scoped native PySGG runtime. Compatible with Python 3.8."""
import argparse
import json
import os
from pathlib import Path

import torch

from common import OLD, ROOT, atomic_json, canonical_path, sole_match, sha256


def assets(family, task):
    name = "pysgg_%s_vg_live.json" % family
    manifest = json.loads((OLD / "checkpoints/sgg/manifests" / name).read_text())
    checkpoint = canonical_path(manifest["checkpoints"][task]["path"])
    expected = manifest["checkpoints"][task]["sha256"]
    if family == "transformer" and task == "sgcls":
        checkpoint = sole_match("checkpoints/sgg/trained/pysgg/transformer_object_refine_fixed_20260724/sgcls/*/*/model_final.pth")
        expected = "ed150f8ad4615cd059bc79a1819ffd8b13a0f54d97f1940156be79a83444a59a"
    config = OLD / "configs/pysgg_vg_tritask" / (family + "_" + task + ".yaml")
    cache = OLD / "artifacts/prediction_cache" / (
        "pysgg_transformer_object_refine_fixed_20260724" if family == "transformer"
        else "causal_motifs_sum_tde_vg_entity_aligned")
    return manifest, checkpoint, expected, config, cache


def configure(family, task, output):
    from pysgg.config import cfg
    manifest, checkpoint, expected, config, cache = assets(family, task)
    cfg.merge_from_file(str(config))
    cfg.defrost()
    cfg.PATHS_CATALOG = str(OLD / "external/official_repos/PySGG/pysgg/config/paths_catalog.py")
    cfg.MODEL.WEIGHT = str(checkpoint)
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = task in ("sgcls", "predcls")
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = task == "predcls"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.OUTPUT_DIR = str(output / "runtime")
    cfg.freeze()
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    os.chdir(str(OLD / "external/official_repos/PySGG"))
    return cfg


def dataset(cfg, split, all_training=False):
    from pysgg.config.paths_catalog import DatasetCatalog
    from pysgg.data.datasets.visual_genome import VGDataset
    key = getattr(cfg.DATASETS, {"test": "TEST", "val": "VAL", "train": "TRAIN"}[split])[0]
    kwargs = DatasetCatalog.get(key, cfg)["args"]
    kwargs.update(filter_duplicate_rels=False, filter_non_overlap=False)
    if all_training:
        kwargs["num_val_im"] = 0
    ds = VGDataset(**kwargs)
    assert len(ds.ind_to_classes) == 151 and len(ds.ind_to_predicates) == 51
    return ds


def load_model(family, task, output):
    from pysgg.modeling.detector import build_detection_model
    from pysgg.utils.checkpoint import DetectronCheckpointer
    from pysgg.data.transforms import build_transforms
    from export_pysgg_vg_task import validate_checkpoint_coverage
    cfg = configure(family, task, output)
    manifest, checkpoint, expected, config, cache = assets(family, task)
    digest = sha256(checkpoint)
    if digest != expected:
        raise RuntimeError("Checkpoint hash mismatch: %s" % checkpoint)
    model = build_detection_model(cfg).cuda()
    payload = torch.load(str(checkpoint), map_location="cpu")
    coverage = validate_checkpoint_coverage(model, payload)
    DetectronCheckpointer(cfg, model, save_dir=str(output / "runtime"))._load_model(payload, {})
    model.eval()
    provenance = dict(family=family, task=task, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                      config=str(config), config_sha256=sha256(config), coverage=coverage,
                      source_files={str(p.relative_to(OLD)): sha256(p) for p in
                         (OLD / "external/official_repos/PySGG/pysgg/modeling/roi_heads/relation_head").glob("*.py")})
    atomic_json(output / "model_provenance.json", provenance)
    return model, cfg, build_transforms(cfg, is_train=False), provenance


def image_id(ds, index):
    return Path(ds.filenames[index]).stem


def infer(model, cfg, transform, ds, index):
    from pysgg.structures.image_list import to_image_list
    image, target, _ = ds[index]
    image, target = transform(image, target)
    images = to_image_list([image.cuda()], size_divisible=int(cfg.DATALOADER.SIZE_DIVISIBILITY))
    with torch.no_grad():
        prediction = model(images, [target.to("cuda")], logger=None)[0]
    return prediction, target
