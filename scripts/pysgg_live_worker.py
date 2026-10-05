#!/usr/bin/env python3
"""Persistent Python-3.8 PySGG worker for the modern benchmark process."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torchvision.transforms import functional as tvf


NUM_OBJECTS = 151


def _full_object_logits(refine_logits, attribute_on):
    """Return the per-entity refined class logits passed to post-processing."""
    values = refine_logits
    if attribute_on and isinstance(values[0], (list, tuple)):
        values, _ = values
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise RuntimeError(
            "PySGG live inference expects one image of refined object logits"
        )
    logits = values[0]
    if logits.ndim != 2 or logits.shape[1] != NUM_OBJECTS:
        raise RuntimeError(
            "Unexpected refined object-logit shape: %s" % (tuple(logits.shape),)
        )
    return logits


def _load_model(args):
    from pysgg.config import cfg
    from pysgg.data.transforms import build_transforms
    from pysgg.modeling.detector import build_detection_model
    from pysgg.utils.checkpoint import DetectronCheckpointer

    source = Path(args.source_root).resolve()
    cfg.merge_from_file(str(Path(args.config).resolve()))
    cfg.defrost()
    cfg.PATHS_CATALOG = str(source / "pysgg/config/paths_catalog.py")
    cfg.MODEL.WEIGHT = str(Path(args.checkpoint).resolve())
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.OUTPUT_DIR = str(Path(args.queue_dir).resolve() / "runtime")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    cfg.freeze()
    model = build_detection_model(cfg).to(torch.device(args.device))
    payload = torch.load(str(Path(args.checkpoint).resolve()), map_location="cpu")
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
    total = sum(value.numel() for value in current.values())
    coverage = sum(current[key].numel() for key in matched) / max(total, 1)
    if coverage < 0.98:
        raise RuntimeError("checkpoint parameter coverage %.4f < 0.98" % coverage)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=str(Path(args.queue_dir)))
    checkpointer._load_model(payload, {})
    del payload
    model.eval()
    return (
        model, build_transforms(cfg, is_train=False), total, coverage,
        int(cfg.DATALOADER.SIZE_DIVISIBILITY),
    )


def _target(payload, image):
    from pysgg.structures.bounding_box import BoxList

    height, width = image.shape[-2:]
    boxes = torch.from_numpy(np.asarray(payload["boxes"])).float()
    if boxes.numel() and float(boxes.max()) <= 1.5:
        boxes *= boxes.new_tensor([width, height, width, height])
    target = BoxList(boxes, (width, height), "xyxy")
    labels = torch.from_numpy(np.asarray(payload["entity_labels"])).long()
    pairs = torch.from_numpy(np.asarray(payload["rel_pairs"])).long()
    predicates = torch.from_numpy(np.asarray(payload["rel_labels"])).long()
    target.add_field("labels", labels)
    target.add_field("attributes", torch.zeros((len(target), 10), dtype=torch.long))
    relation = torch.zeros((len(target), len(target)), dtype=torch.long)
    for pair, predicate in zip(pairs.tolist(), predicates.tolist()):
        subject, obj = map(int, pair)
        if 0 <= subject < len(target) and 0 <= obj < len(target):
            relation[subject, obj] = int(predicate)
    target.add_field("relation", relation, is_triplet=True)
    target.add_field("relation_tuple", torch.cat((pairs, predicates[:, None]), dim=1))
    return target, pairs


def _run_request(model, transform, input_path, device, size_divisibility):
    from pysgg.structures.image_list import to_image_list

    with np.load(str(input_path), allow_pickle=False) as payload:
        image = torch.from_numpy(np.asarray(payload["image"])).float()
        target, gt_pairs = _target(payload, image)
        require_gt_pairs = bool(
            np.asarray(payload["require_gt_pairs"]).item()
        ) if "require_gt_pairs" in payload.files else False
    pil = tvf.to_pil_image(image.clamp(0.0, 1.0))
    image, target = transform(pil, target)
    images = to_image_list(
        [image.to(device)], size_divisible=int(size_divisibility)
    )
    captured = {}

    def capture_refined_logits(module, inputs):
        _, refine_logits = inputs[0]
        captured["entity_logits"] = _full_object_logits(
            refine_logits, bool(module.attribute_on)
        ).detach()

    post_processor = model.roi_heads["relation"].post_processor
    sampler = model.roi_heads["relation"].samp_processor
    original_pair_cap = int(sampler.max_proposal_pairs)
    if require_gt_pairs:
        num_entities = len(target)
        sampler.max_proposal_pairs = max(
            original_pair_cap, num_entities * max(num_entities - 1, 0)
        )
    hook = post_processor.register_forward_pre_hook(capture_refined_logits)
    try:
        with torch.no_grad():
            prediction = model(
                images, [target.to(device)], logger=None
            )[0]
    finally:
        hook.remove()
        sampler.max_proposal_pairs = original_pair_cap
    pairs = prediction.get_field("rel_pair_idxs").long()
    relation_scores = prediction.get_field("pred_rel_scores").float()
    rows = {}
    for index, pair in enumerate(pairs.tolist()):
        rows.setdefault((int(pair[0]), int(pair[1])), index)
    missing = [tuple(map(int, pair)) for pair in gt_pairs.tolist()
               if tuple(map(int, pair)) not in rows]
    if missing:
        raise RuntimeError(
            "PySGG omitted %d annotated GT pairs "
            "(entities=%d, pair_cap=%d, require_gt_pairs=%s, examples=%s)"
            % (
                len(missing), len(target), original_pair_cap,
                require_gt_pairs, missing[:5],
            )
        )
    selected = torch.as_tensor(
        [rows[tuple(map(int, pair))] for pair in gt_pairs.tolist()],
        device=relation_scores.device, dtype=torch.long,
    )
    entity_scores = captured.get("entity_logits")
    if entity_scores is None:
        raise RuntimeError("PySGG did not expose refined object logits")
    if entity_scores.shape[0] != len(target):
        raise RuntimeError(
            "Refined object logits are not aligned with GT entities: %d != %d"
            % (entity_scores.shape[0], len(target))
        )
    return {
        "pred_rel_scores": relation_scores[selected].cpu().numpy(),
        "pred_entity_scores": entity_scores.cpu().numpy(),
        "pred_rel_pairs": gt_pairs.cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queue_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    queue = Path(args.queue_dir).resolve()
    queue.mkdir(parents=True, exist_ok=True)
    model, transform, parameter_count, coverage, size_divisibility = _load_model(args)
    (queue / "READY.json").write_text(json.dumps({
        "parameter_count": int(parameter_count),
        "checkpoint_coverage": float(coverage),
        "object_score_semantics": "full_refined_logits",
        "pid": os.getpid(),
    }) + "\n")
    while True:
        requests = sorted(queue.glob("*.request.json"))
        if not requests:
            time.sleep(0.02)
            continue
        request = requests[0]
        request_id = request.name.split(".", 1)[0]
        input_path = queue / (request_id + ".input.npz")
        output_path = queue / (request_id + ".output.npz")
        try:
            output = _run_request(
                model, transform, input_path, args.device, size_divisibility
            )
            temporary = queue / (request_id + ".output.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **output)
            os.replace(str(temporary), str(output_path))
            (queue / (request_id + ".done.json")).write_text(
                json.dumps({"status": "ok"}) + "\n"
            )
        except Exception as exc:
            (queue / (request_id + ".error.json")).write_text(json.dumps({
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            }) + "\n")
        finally:
            request.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
