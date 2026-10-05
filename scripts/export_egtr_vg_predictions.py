#!/usr/bin/env python3
"""Export official EGTR VG/OI SGDet predictions from its isolated runtime.

The released EGTR head uses independent sigmoid predicate probabilities. The
cache therefore records that score semantics explicitly and retains the union
of the official top graph-constrained pairs and top no-graph triplet pairs.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_transformers_compatibility() -> None:
    """Restore the one symbol moved after EGTR's Transformers 4.18 runtime."""
    import transformers.models.detr.feature_extraction_detr as legacy_detr
    if not hasattr(legacy_detr, "center_to_corners_format"):
        from transformers.image_transforms import center_to_corners_format
        legacy_detr.center_to_corners_format = center_to_corners_format


def load_model(source_root: Path, config_root: Path, checkpoint: Path,
               device: torch.device):
    install_transformers_compatibility()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import model.deformable_detr as deformable_detr
    from model.deformable_detr import DeformableDetrConfig
    from model.egtr import DetrForSceneGraphGeneration

    # The complete released checkpoint contains the backbone. Suppress timm's
    # redundant ImageNet network request before constructing the architecture.
    upstream_create_model = deformable_detr.create_model

    def offline_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return upstream_create_model(*args, **kwargs)

    deformable_detr.create_model = offline_create_model
    config = DeformableDetrConfig.from_pretrained(
        str(config_root), local_files_only=True
    )
    model = DetrForSceneGraphGeneration(config=config)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError(f"EGTR checkpoint has no state_dict: {checkpoint}")
    converted = {}
    for key, value in state.items():
        if not key.startswith("model."):
            raise ValueError(f"Unexpected EGTR state key without model. prefix: {key}")
        converted[key[len("model."):]] = value
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict EGTR load failed: {incompatible}")
    return model.to(device).eval(), config


def prepare_image(image: torch.Tensor, device: torch.device,
                  min_size: int, max_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("EGTR export requires image batch_size=1")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected RGB image tensor [3,H,W]")
    image = image.to(device=device, dtype=torch.float32)
    height, width = map(int, image.shape[-2:])
    scale = min(float(min_size) / min(height, width),
                float(max_size) / max(height, width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    image = F.interpolate(
        image.unsqueeze(0), size=(new_height, new_width), mode="bilinear",
        align_corners=False, antialias=True,
    )
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    image = (image - mean) / std
    pixel_mask = torch.ones(
        (1, new_height, new_width), dtype=torch.bool, device=device
    )
    return image, pixel_mask


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1).clamp(0, 1)


def scene_graph_output(outputs, num_labels: int, topk: int) -> dict[str, np.ndarray]:
    logits = outputs.logits[0]
    object_probs = logits.softmax(dim=-1)[:, :num_labels]
    object_confidence, _ = object_probs.max(dim=-1)
    relation = outputs.pred_rel[0].clamp(0, 1)
    connectivity = outputs.pred_connectivity[0].clamp(0, 1)
    relation = relation * connectivity

    pair_confidence = torch.outer(object_confidence, object_confidence)
    pair_confidence.fill_diagonal_(0)
    graph_scores = relation.max(dim=-1).values * pair_confidence
    triplet_scores = relation * pair_confidence.unsqueeze(-1)
    query_count = int(logits.shape[0])
    graph_indices = torch.topk(
        graph_scores.reshape(-1), k=min(topk, graph_scores.numel())
    ).indices
    triplet_indices = torch.topk(
        triplet_scores.reshape(-1), k=min(topk, triplet_scores.numel())
    ).indices
    relation_count = int(relation.shape[-1])
    triplet_pairs = torch.div(
        triplet_indices, relation_count, rounding_mode="floor"
    )
    pair_indices = torch.unique(
        torch.cat((graph_indices, triplet_pairs)), sorted=True
    )
    subjects = torch.div(pair_indices, query_count, rounding_mode="floor")
    objects = pair_indices.remainder(query_count)
    pairs = torch.stack((subjects, objects), dim=-1)

    # VG-150 reserves index 0 for background; EGTR predicts object IDs 0..149
    # and predicate IDs 0..49, so shift both ontologies with dummy columns.
    entity_scores = torch.cat(
        (object_probs.new_zeros((query_count, 1)), object_probs), dim=-1
    )
    selected_relations = relation[subjects, objects]
    relation_scores = torch.cat(
        (selected_relations.new_zeros((selected_relations.shape[0], 1)),
         selected_relations),
        dim=-1,
    )
    return {
        "pred_boxes": cxcywh_to_xyxy(outputs.pred_boxes[0]).cpu().numpy(),
        "pred_entity_scores": entity_scores.cpu().numpy(),
        "pred_rel_pairs": pairs.cpu().numpy(),
        "pred_rel_scores": relation_scores.cpu().numpy(),
    }


def valid_existing_prediction(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return all(key in payload for key in (
                "pred_boxes", "pred_entity_scores", "pred_rel_pairs",
                "pred_rel_scores",
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
    parser.add_argument("--config_root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_samples", type=int, default=1_000_000_000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
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
        args.source_root or root / "external/official_repos/egtr"
    ).expanduser().resolve()
    config_root = Path(
        args.config_root
        or root / f"checkpoints/sgg/weights/egtr/{dataset}/runtime"
    ).expanduser().resolve()
    checkpoint = Path(
        args.checkpoint or config_root / "model.ckpt"
    ).expanduser().resolve()
    output_dir = Path(
        args.output_dir or root / f"artifacts/prediction_cache/egtr_{dataset}"
    ).expanduser().resolve()
    marker_path = source_root / ".official_source.json"
    for path in (data_root, source_root, config_root, checkpoint, marker_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.topk < 1:
        raise ValueError("--topk must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, config = load_model(source_root, config_root, checkpoint, device)
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
    model_object_classes = int(config.num_labels)
    model_predicate_classes = int(config.num_rel_labels)
    expected_objects = int(loader.dataset.num_entity_classes) - 1
    expected_predicates = int(loader.dataset.num_predicate_classes) - 1
    if (model_object_classes, model_predicate_classes) != (
        expected_objects, expected_predicates
    ):
        raise RuntimeError(
            "EGTR checkpoint/loader ontology mismatch: "
            f"model={model_object_classes}/{model_predicate_classes}, "
            f"loader={expected_objects}/{expected_predicates}"
        )
    model_name = f"egtr_{dataset}_official"
    writer = OfficialPredictionCacheWriter(
        output_dir,
        model_name=model_name,
        architecture_family="EGTR",
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
        "topk": int(args.topk),
        "min_size": int(args.min_size),
        "max_size": int(args.max_size),
        "relation_score_mode": "independent_probabilities",
    }
    state_path = output_dir / "export_state.json"
    existing_predictions = list(
        (output_dir / "predictions/sgdet").glob("*.npz")
    )
    if state_path.is_file():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        if previous_state != export_state:
            raise RuntimeError(
                "Refusing to mix EGTR prediction-cache provenance: "
                f"existing={previous_state}, requested={export_state}"
            )
    elif args.resume and existing_predictions:
        raise RuntimeError(
            f"Cannot resume {len(existing_predictions)} predictions without "
            f"provenance lock: {state_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(export_state, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    processed = skipped = 0
    for index, batch in enumerate(loader, start=1):
        image_id = batch["image_id"]
        if isinstance(image_id, torch.Tensor):
            image_id = image_id.item()
        prediction_path = output_dir / "predictions/sgdet" / f"{image_id}.npz"
        if args.resume and valid_existing_prediction(prediction_path):
            writer.metadata["image_ids"].append(str(image_id))
            writer.metadata["images_by_task"]["sgdet"] += 1
            skipped += 1
        else:
            image = batch.get("image")
            if not isinstance(image, torch.Tensor):
                raise FileNotFoundError(
                    f"Raw {dataset} image missing for image_id={image_id}"
                )
            pixel_values, pixel_mask = prepare_image(
                image, device, args.min_size, args.max_size
            )
            # The upstream PyTorch fallback prints once per deformable-attention
            # layer when its optional CUDA extension is absent. Silence only
            # that stdout chatter; exceptions and stderr remain visible.
            with torch.inference_mode(), redirect_stdout(io.StringIO()):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    output_attention_states=True,
                )
            prediction = scene_graph_output(outputs, model_object_classes, args.topk)
            writer.add("sgdet", image_id, **prediction)
            processed += 1
        if index % args.log_every == 0 or index == len(loader.dataset):
            writer.finalize()
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({
                "completed": index,
                "total": len(loader.dataset),
                "processed": processed,
                "resumed": skipped,
                "images_per_second": index / elapsed,
            }), flush=True)
    metadata = writer.finalize()
    print(f"metadata={metadata}")
    print(f"parameter_count={parameter_count}")
    print(f"checkpoint_sha256={checkpoint_digest}")


if __name__ == "__main__":
    main()
