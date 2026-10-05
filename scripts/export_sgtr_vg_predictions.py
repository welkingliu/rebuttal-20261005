#!/usr/bin/env python3
"""Export released SGTR VG/OI checkpoints to the strict cache schema."""

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
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.data.data_utils import build_vg_test_loader
from sgg_core.data.oi_data_utils import build_oi_loader
from sgg_core.models.prediction_cache_writer import OfficialPredictionCacheWriter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(source_root: Path, config_path: Path, checkpoint: Path,
               device: torch.device, dataset: str):
    experiment_root = (
        source_root / "playground/sgg/detr.res101.c5.one_stage_rel_tfmer"
    )
    for path in (source_root, experiment_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault(
        "CVPODS_OUTPUT", str(PROJECT_ROOT / "artifacts/runtime/sgtr")
    )

    from cvpods.configs.base_config import ConfigDict
    if dataset == "vg":
        from config_vg_sgtr import config
    else:
        from config_oiv6 import config
    from net import build_model

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    # Architecture and post-processing values must come from the checkpoint's
    # own config. Host-specific output and weight paths are replaced below.
    config.merge(ConfigDict(saved_config))
    config.MODEL.DEVICE = str(device)
    config.MODEL.WEIGHTS = str(checkpoint)
    config.MODEL.TEST_WEIGHTS = str(checkpoint)
    config.OUTPUT_DIR = str(PROJECT_ROOT / "artifacts/runtime/sgtr")
    config.DUMP_INTERMEDITE = False

    model = build_model(config)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError(f"SGTR checkpoint has no model state_dict: {checkpoint}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict SGTR load failed: {incompatible}")
    return model.to(device).eval(), config


def prepare_input(batch: dict, device: torch.device, num_object_classes: int,
                  dataset: str):
    from cvpods.structures import Boxes, Instances
    from cvpods.structures.relationship import Relationships

    image = batch.get("image")
    if not isinstance(image, torch.Tensor):
        raise KeyError("SGTR export requires batch['image']")
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("SGTR export requires image batch_size=1")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected RGB image [3,H,W]")
    original_height, original_width = map(int, image.shape[-2:])
    scale = min(
        600.0 / min(original_height, original_width),
        1000.0 / max(original_height, original_width),
    )
    resized_height = max(1, int(round(original_height * scale)))
    resized_width = max(1, int(round(original_width * scale)))
    image = F.interpolate(
        image.float().unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    # The released cvpods config uses BGR images in the 0..255 range.
    image = image[[2, 1, 0]] * 255.0

    boxes = batch["boxes"].float().clone()
    boxes *= boxes.new_tensor([
        resized_width, resized_height, resized_width, resized_height,
    ])
    instances = Instances((resized_height, resized_width))
    instances.gt_boxes = Boxes(boxes)
    labels = batch["entity_labels"].long() - 1
    if bool((labels < 0).any()) or bool((labels >= num_object_classes).any()):
        raise ValueError(
            f"{dataset} object IDs do not align with SGTR's "
            f"0..{num_object_classes - 1} classes"
        )
    instances.gt_classes = labels
    instances.gt_classes_non_masked = labels.clone()

    pairs = batch["rel_pairs"].long()
    predicates = batch["rel_labels"].long()
    relationships = Relationships(
        instances,
        pairs,
        rel_label=predicates,
        rel_label_no_mask=predicates.clone(),
        relation_tuple=torch.cat((pairs, predicates[:, None]), dim=1),
    )
    cache_image_id = batch["image_id"]
    if isinstance(cache_image_id, torch.Tensor):
        cache_image_id = int(cache_image_id.item())
    elif dataset == "vg":
        cache_image_id = int(cache_image_id)
    else:
        cache_image_id = str(cache_image_id)
    model_image_id = (
        int(cache_image_id)
        if dataset == "vg"
        else int(cache_image_id, 16) % (2**63 - 1)
    )
    model_input = {
        "image": image.to(device),
        "height": original_height,
        "width": original_width,
        "image_id": model_image_id,
        "instances": instances,
        "relationships": relationships,
    }
    return [model_input], original_height, original_width, cache_image_id


def convert_output(output: dict, height: int, width: int,
                   num_object_classes: int,
                   num_predicate_classes: int) -> dict[str, np.ndarray]:
    instances = output["instances"].to("cpu")
    relationships = output["relationships"].to("cpu")
    object_prob = instances.pred_score_dist.float()
    if (
        object_prob.ndim != 2
        or object_prob.shape[1] != num_object_classes + 1
    ):
        raise ValueError(f"Unexpected SGTR object distribution: {object_prob.shape}")
    # SGTR's 151st column is no-object. Preserve its mass as objectness, then
    # encode the conditional 150-class distribution as logits. The common
    # evaluator reserves column zero for background.
    objectness = (1.0 - object_prob[:, -1]).clamp(0, 1)
    foreground = object_prob[:, :num_object_classes].clamp_min(1e-12)
    entity_scores = torch.cat(
        (foreground.new_full((foreground.shape[0], 1), -1e4), foreground.log()),
        dim=1,
    )
    boxes = instances.pred_boxes.tensor.float()
    boxes /= boxes.new_tensor([width, height, width, height])
    boxes = boxes.clamp(0, 1)

    relation_prob = relationships.pred_rel_dist.float()
    if relation_prob.ndim != 2 or relation_prob.shape[1] not in (
        num_predicate_classes, num_predicate_classes + 1
    ):
        raise ValueError(f"Unexpected SGTR predicate distribution: {relation_prob.shape}")
    # The VG focal head emits foreground probabilities only. The OI softmax
    # head already includes background in column zero, as used by SGTR's own
    # post-processor (`pred_rel_probs[:, 1:]`).
    relation_scores = (
        relation_prob
        if relation_prob.shape[1] == num_predicate_classes + 1
        else torch.cat(
            (relation_prob.new_zeros((relation_prob.shape[0], 1)), relation_prob),
            dim=1,
        )
    )
    pairs = relationships.rel_pair_tensor.long()
    if pairs.numel() and int(pairs.max()) >= boxes.shape[0]:
        raise ValueError("SGTR relation pair references a missing entity query")
    return {
        "pred_boxes": boxes.numpy(),
        "pred_entity_scores": entity_scores.numpy(),
        "pred_box_scores": objectness.numpy(),
        "pred_rel_pairs": pairs.numpy(),
        "pred_rel_scores": relation_scores.numpy(),
    }


def valid_existing_prediction(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return all(key in payload for key in (
                "pred_boxes", "pred_entity_scores", "pred_box_scores",
                "pred_rel_pairs", "pred_rel_scores",
            ))
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset", choices=("vg", "oi"), default="vg")
    parser.add_argument("--vg_root")
    parser.add_argument("--oi_root")
    parser.add_argument("--source_root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_samples", type=int, default=1_000_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    dataset = args.dataset.lower()
    data_root = Path(
        (args.vg_root or root / "data/vg/v1.4") if dataset == "vg"
        else (args.oi_root or root / "data/openimages/open-images-v6")
    ).expanduser().resolve()
    source_root = Path(
        args.source_root or root / "external/official_repos/SGTR"
    ).expanduser().resolve()
    checkpoint_root = (
        root / "checkpoints/sgg/weights/sgtr/vg/sgtr_vg_new_pth"
        if dataset == "vg"
        else root / "checkpoints/sgg/weights/sgtr/oi/sgtr_oiv6_new"
    )
    checkpoint = Path(
        args.checkpoint
        or checkpoint_root / (
            "model_0095999.pth" if dataset == "vg" else "model_0107999.pth"
        )
    ).expanduser().resolve()
    config_path = Path(args.config or checkpoint_root / "config.json").expanduser().resolve()
    output_dir = Path(
        args.output_dir or root / f"artifacts/prediction_cache/sgtr_{dataset}"
    ).expanduser().resolve()
    marker_path = source_root / ".official_source.json"
    for path in (data_root, source_root, checkpoint, config_path, marker_path):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, config = load_model(
        source_root, config_path, checkpoint, device, dataset
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    source_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint_digest = sha256(checkpoint)
    if dataset == "vg":
        loader = build_vg_test_loader(
            str(data_root), num_samples=args.eval_samples, batch_size=1, split=2,
            include_proxy_features=False, include_raw_images=True,
        )
        split = "test"
    else:
        loader = build_oi_loader(
            str(data_root), split="validation", num_samples=args.eval_samples,
            batch_size=1, include_proxy_features=False,
            include_raw_images=True,
        )
        split = "validation"
        if not loader.dataset.vocab.is_official_sgg_ontology:
            raise RuntimeError(
                "OI export requires annotations/oi_sgg_ontology.json; run "
                "scripts/build_openimages_sgg_ontology.py first"
            )
    num_object_classes = int(loader.dataset.num_entity_classes) - 1
    num_predicate_classes = int(loader.dataset.num_predicate_classes) - 1
    configured_objects = int(config.MODEL.DETR.NUM_CLASSES)
    configured_predicates = int(config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)
    if (configured_objects, configured_predicates) != (
        num_object_classes, num_predicate_classes
    ):
        raise RuntimeError(
            "SGTR checkpoint/loader ontology mismatch: "
            f"model={configured_objects}/{configured_predicates}, "
            f"loader={num_object_classes}/{num_predicate_classes}"
        )
    model_name = f"sgtr_{dataset}_official"
    writer = OfficialPredictionCacheWriter(
        output_dir,
        model_name=model_name,
        architecture_family="SGTR",
        source_commit=source_marker["commit"],
        parameter_count=parameter_count,
        checkpoint_sha256_by_task={"sgdet": checkpoint_digest},
        dataset=dataset,
        ontology_id=loader.dataset.ontology_id,
        split=split,
        tasks=("sgdet",),
    )
    export_state = {
        "model_name": model_name,
        "source_commit": source_marker["commit"],
        "checkpoint_sha256": checkpoint_digest,
        "ontology_id": loader.dataset.ontology_id,
        "split": split,
        "resize_short": 600,
        "resize_max": 1000,
        "relation_score_mode": "independent_probabilities",
    }
    state_path = output_dir / "export_state.json"
    existing = list((output_dir / "predictions/sgdet").glob("*.npz"))
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous != export_state:
            raise RuntimeError(
                f"Refusing to mix SGTR cache provenance: {previous} != {export_state}"
            )
    elif args.resume and existing:
        raise RuntimeError(f"Cannot resume {len(existing)} files without {state_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(export_state, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    processed = resumed = 0
    for index, batch in enumerate(loader, start=1):
        image_id = batch["image_id"]
        if isinstance(image_id, torch.Tensor):
            image_id = image_id.item()
        prediction_path = output_dir / "predictions/sgdet" / f"{image_id}.npz"
        if args.resume and valid_existing_prediction(prediction_path):
            writer.metadata["image_ids"].append(str(image_id))
            writer.metadata["images_by_task"]["sgdet"] += 1
            resumed += 1
        else:
            model_input, height, width, image_id = prepare_input(
                batch, device, num_object_classes, dataset
            )
            with torch.inference_mode():
                output = model(model_input)[0]
            writer.add(
                "sgdet", image_id,
                **convert_output(
                    output, height=height, width=width,
                    num_object_classes=num_object_classes,
                    num_predicate_classes=num_predicate_classes,
                ),
            )
            processed += 1
        if index % args.log_every == 0 or index == len(loader.dataset):
            writer.finalize()
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({
                "completed": index,
                "total": len(loader.dataset),
                "processed": processed,
                "resumed": resumed,
                "images_per_second": index / elapsed,
            }), flush=True)
    metadata = writer.finalize()
    print(f"metadata={metadata}")
    print(f"parameter_count={parameter_count}")
    print(f"checkpoint_sha256={checkpoint_digest}")


if __name__ == "__main__":
    main()
