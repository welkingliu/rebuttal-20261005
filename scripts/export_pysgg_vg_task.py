#!/usr/bin/env python3
"""Export one real PySGG VG-150 task into the strict prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_prediction(path):
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            return all(name in payload for name in (
                "pred_boxes", "pred_entity_scores", "pred_box_scores",
                "pred_rel_pairs", "pred_rel_scores",
            ))
    except (OSError, ValueError):
        return False


def validate_checkpoint_coverage(model, payload, minimum=0.98):
    """Reject a wrong-family or partial checkpoint before permissive loading."""
    loaded = payload.get("model", payload)
    loaded = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in loaded.items()
    }
    current = model.state_dict()
    matched = {
        key for key, value in current.items()
        if key in loaded and tuple(value.shape) == tuple(loaded[key].shape)
    }
    total_parameters = sum(value.numel() for value in current.values())
    matched_parameters = sum(current[key].numel() for key in matched)
    relation_keys = {
        key for key in current if key.startswith("roi_heads.relation.")
    }
    matched_relation = relation_keys & matched
    relation_total = sum(current[key].numel() for key in relation_keys)
    relation_loaded = sum(current[key].numel() for key in matched_relation)
    coverage = matched_parameters / max(total_parameters, 1)
    relation_coverage = relation_loaded / max(relation_total, 1)
    if coverage < float(minimum) or relation_coverage < float(minimum):
        raise RuntimeError(
            "Checkpoint does not fully instantiate the requested PySGG model: "
            "parameter_coverage=%.4f relation_coverage=%.4f" % (
                coverage, relation_coverage,
            )
        )
    return {
        "parameter_coverage": coverage,
        "relation_parameter_coverage": relation_coverage,
        "matched_tensors": len(matched),
        "model_tensors": len(current),
    }


def convert_prediction(prediction, num_objects=151):
    prediction = prediction.convert("xyxy")
    width, height = map(float, prediction.size)
    boxes = prediction.bbox.float().cpu()
    boxes /= boxes.new_tensor([width, height, width, height])
    labels = prediction.get_field("pred_labels").long().cpu()
    scores = prediction.get_field("pred_scores").float().cpu()
    entity_scores = scores.new_zeros((labels.numel(), int(num_objects)))
    if labels.numel():
        if int(labels.min()) < 1 or int(labels.max()) >= int(num_objects):
            raise ValueError("PySGG object labels are outside VG-150 IDs")
        entity_scores[:, 0] = (1.0 - scores).clamp(min=0.0)
        entity_scores[torch.arange(labels.numel()), labels] = scores
    # PySGG's pred_scores is the refined object-class confidence used by its
    # official triplet ranking, not a separate detector objectness value. Keep
    # it in entity_scores and use neutral box scores so the unified evaluator
    # multiplies each object confidence exactly once.
    box_scores = torch.ones_like(scores)
    pairs = prediction.get_field("rel_pair_idxs").long().cpu()
    relation_scores = prediction.get_field("pred_rel_scores").float().cpu()
    if relation_scores.ndim != 2 or relation_scores.size(1) != 51:
        raise ValueError("PySGG relation scores must use the 50-predicate ontology")
    if pairs.size(0) != relation_scores.size(0):
        raise ValueError("PySGG pair and relation rows differ")
    return {
        "pred_boxes": boxes.numpy(),
        "pred_entity_scores": entity_scores.numpy(),
        "pred_box_scores": box_scores.numpy(),
        "pred_rel_pairs": pairs.numpy(),
        "pred_rel_scores": relation_scores.numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("predcls", "sgcls", "sgdet"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_samples", type=int, default=1_000_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    source = root / "external/official_repos/PySGG"
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(source))

    from pysgg.config import cfg
    from pysgg.data import make_data_loader
    from pysgg.modeling.detector import build_detection_model
    from pysgg.utils.checkpoint import DetectronCheckpointer

    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.PATHS_CATALOG = str(source / "pysgg/config/paths_catalog.py")
    cfg.MODEL.WEIGHT = str(checkpoint)
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = args.task in ("predcls", "sgcls")
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = args.task == "predcls"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.TEST.ALLOW_LOAD_FROM_CACHE = False
    cfg.TEST.RELATION.SYNC_GATHER = False
    cfg.OUTPUT_DIR = str(output / "runtime" / args.task)
    cfg.freeze()
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # PySGG's dataset catalog resolves DATA_DIR relative to its repository.
    # All user-provided paths above are absolute, so this is deterministic.
    os.chdir(str(source))
    model = build_detection_model(cfg).to(torch.device("cuda"))
    checkpoint_payload = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_coverage = validate_checkpoint_coverage(model, checkpoint_payload)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer._load_model(checkpoint_payload, {})
    del checkpoint_payload
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loader = make_data_loader(cfg, mode="test", is_distributed=False)[0]
    dataset = loader.dataset
    total = min(len(dataset), int(args.eval_samples))
    source_marker = json.loads((source / ".official_source.json").read_text())
    seen = json.loads((root / "artifacts/manifests/seen_triplets_full.json").read_text())
    ontology_id = seen["_metadata"]["vg"]["ontology_id"]
    state = {
        "schema": "pysgg_vg_task_export_v1",
        "model_name": args.name,
        "architecture_family": args.family,
        "task": args.task,
        "source_commit": source_marker["commit"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": int(parameter_count),
        "ontology_id": ontology_id,
        "images": total,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint_load_coverage": checkpoint_coverage,
        "task_flags": {
            "use_gt_box": bool(cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX),
            "use_gt_object_label": bool(
                cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
            ),
        },
    }
    state_path = output / ("state_" + args.task + ".json")
    if state_path.is_file() and json.loads(state_path.read_text()) != state:
        raise RuntimeError("Refusing to mix PySGG task provenance")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    prediction_dir = output / "predictions" / args.task
    prediction_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for batch_index, batch in enumerate(loader):
        if batch_index >= total:
            break
        images, targets, indices = batch
        image_ids = [
            int(dataset.img_info[int(dataset_index)]["image_id"])
            for dataset_index in indices
        ]
        paths = [prediction_dir / (str(image_id) + ".npz") for image_id in image_ids]
        if args.resume and all(valid_prediction(path) for path in paths):
            completed = batch_index + 1
            if completed % args.log_every == 0 or completed == total:
                print(json.dumps({
                    "model": args.name, "task": args.task,
                    "completed": completed, "total": total, "resumed": True,
                }), flush=True)
            continue
        targets = [target.to("cuda") for target in targets]
        with torch.no_grad():
            predictions = model(images.to("cuda"), targets, logger=None)
        for prediction, path in zip(predictions, paths):
            np.savez_compressed(str(path), **convert_prediction(prediction.to("cpu")))
        completed = batch_index + 1
        if completed % args.log_every == 0 or completed == total:
            print(json.dumps({
                "model": args.name, "task": args.task, "completed": completed,
                "total": total,
                "images_per_second": completed / max(time.monotonic() - started, 1e-6),
            }), flush=True)


if __name__ == "__main__":
    main()
