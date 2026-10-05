"""Experiment I-A: segmentation-conditioned object identity audit.

The frozen foundation backbone never receives object labels. Linear probes are
trained on image-disjoint PSG training objects and evaluated on the held-out
PSG split. Box ROI, ground-truth mask, and optional predicted-mask features are
compared under one object ontology.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from sgg_core.audits.object_grounding import (
    batched_logits,
    deterministic_image_split,
    evaluate_object_logits,
    fit_temperature,
    frequency_groups,
    paired_accuracy_delta,
    relationship_endpoint_summary,
    train_linear_probe,
)
from sgg_core.data.gqa_psg_data_utils import build_psg_loader
from sgg_core.experiments.experiment_1 import FrozenROIEncoder


FEATURE_CACHE_SCHEMA_VERSION = "experiment_1a_object_grounding_v1"
SUMMARY_SCHEMA_VERSION = "experiment_1a_object_grounding_v3"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_image_id(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _predicted_mask_path(root: Path, image_id) -> Path | None:
    name = _safe_image_id(image_id)
    candidates = [root / f"{name}.npz", root / "masks" / f"{name}.npz"]
    return next((path for path in candidates if path.is_file()), None)


def _validate_mask_root(root: Path | None, annotation: str) -> dict | None:
    if root is None:
        return None
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Predicted-mask manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "psg_sam_gt_box_prompt_v1":
        raise RuntimeError(f"Unexpected predicted-mask schema: {manifest_path}")
    expected = Path(annotation).expanduser().resolve()
    observed = Path(payload.get("annotation", "")).expanduser().resolve()
    if observed != expected:
        raise RuntimeError(
            f"Predicted-mask annotation mismatch: {observed} != {expected}"
        )
    expected_sha = hashlib.sha256(expected.read_bytes()).hexdigest()
    if payload.get("annotation_sha256") != expected_sha:
        raise RuntimeError(
            f"Predicted-mask annotation SHA256 mismatch: {manifest_path}"
        )
    if payload.get("images_ready") != payload.get("images_total"):
        raise RuntimeError(f"Predicted-mask cache is incomplete: {manifest_path}")
    if payload.get("prompt") != "ground_truth_box":
        raise RuntimeError(
            f"Experiment I-A requires ground-truth box prompted masks: {manifest_path}"
        )
    return {
        "schema": payload["schema"],
        "prompt": payload["prompt"],
        "class_labels_used": bool(payload.get("class_labels_used", False)),
        "images_total": int(payload["images_total"]),
        "images_ready": int(payload["images_ready"]),
        "annotation_sha256": payload["annotation_sha256"],
        "model_files": payload.get("model_files", {}),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _load_predicted_masks(root: Path | None, image_id, expected: int,
                          expected_segment_ids=None) -> torch.Tensor | None:
    if root is None:
        return None
    path = _predicted_mask_path(root, image_id)
    if path is None:
        raise FileNotFoundError(f"Predicted masks missing for image_id={image_id} in {root}")
    with np.load(path, allow_pickle=False) as payload:
        if "masks" not in payload:
            raise KeyError(f"Predicted-mask cache lacks 'masks': {path}")
        masks = torch.from_numpy(np.asarray(payload["masks"], dtype=np.uint8)).bool()
        cached_segment_ids = (
            np.asarray(payload["segment_ids"], dtype=np.int64)
            if "segment_ids" in payload else None
        )
    if masks.ndim != 3 or masks.size(0) != expected:
        raise ValueError(
            f"Predicted-mask/object mismatch for {path}: "
            f"masks={tuple(masks.shape)} expected_objects={expected}"
        )
    if expected_segment_ids is not None:
        expected_ids = np.asarray(expected_segment_ids, dtype=np.int64)
        if cached_segment_ids is None or not np.array_equal(
            cached_segment_ids, expected_ids
        ):
            raise ValueError(f"Predicted-mask segment order mismatch: {path}")
    return masks


def _mask_ious(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape[-2:] != target.shape[-2:]:
        predicted = F.interpolate(
            predicted[:, None].float(), size=target.shape[-2:], mode="nearest"
        )[:, 0].bool()
    target = target.bool()
    intersection = (predicted & target).flatten(1).sum(dim=1).float()
    union = (predicted | target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _cache_key(backbone: str, split: str, samples: int,
               predicted_mask_root: Path | None, backbone_sha256: str) -> str:
    marker = str(predicted_mask_root.resolve()) if predicted_mask_root else "none"
    manifest = predicted_mask_root / "manifest.json" if predicted_mask_root else None
    if manifest is not None and manifest.is_file():
        marker += ":" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    marker += ":" + str(backbone_sha256)
    digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:10]
    return (
        f"{FEATURE_CACHE_SCHEMA_VERSION}_{backbone}_{split}_{samples}_pred-{digest}.pt"
    )


@torch.no_grad()
def materialize_object_views(loader, encoder: FrozenROIEncoder, cache_path: Path,
                             predicted_mask_root: Path | None) -> dict:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") == FEATURE_CACHE_SCHEMA_VERSION:
            return payload
        raise RuntimeError(f"Stale Experiment I-A cache schema: {cache_path}")

    view_chunks = defaultdict(list)
    labels, image_ids, object_indices, areas, mask_ious = [], [], [], [], []
    graph_records = []
    skipped = []
    object_offset = 0
    for loader_index, batch in enumerate(loader):
        image = batch.get("image")
        boxes = batch.get("boxes")
        gt_masks = batch.get("masks")
        entity_labels = batch.get("entity_labels")
        image_id = batch.get("image_id", loader_index)
        if not all(isinstance(value, torch.Tensor) for value in (
            image, boxes, gt_masks, entity_labels
        )):
            skipped.append({
                "loader_index": loader_index,
                "image_id": str(image_id),
                "reason": "image_boxes_masks_or_labels_missing",
            })
            continue
        valid = entity_labels > 0
        if not bool(valid.any()):
            continue
        if not bool(valid.all()):
            raise ValueError(
                f"PSG instance labels must all be foreground for image_id={image_id}"
            )
        boxes = boxes[valid].float()
        gt_masks = gt_masks[valid].bool()
        entity_labels = entity_labels[valid].long()
        if boxes.size(0) != gt_masks.size(0):
            raise ValueError(f"PSG box/mask count mismatch for image_id={image_id}")

        feature_map = encoder.extract_feature_map(image)
        box_features = encoder._pool_boxes(
            feature_map, boxes, encoder.roi_chunk_size, output_size=7
        )
        gt_mask_features = encoder._pool_masks(feature_map, gt_masks, box_features)
        view_chunks["box"].append(box_features.to(torch.float16))
        view_chunks["gt_mask"].append(gt_mask_features.to(torch.float16))

        predicted_masks = _load_predicted_masks(
            predicted_mask_root, image_id, boxes.size(0),
            (
                batch["segment_ids"][valid]
                if isinstance(batch.get("segment_ids"), torch.Tensor)
                else None
            ),
        )
        if predicted_masks is not None:
            view_chunks["pred_mask"].append(
                encoder._pool_masks(feature_map, predicted_masks, box_features).to(torch.float16)
            )
            image_mask_iou = _mask_ious(predicted_masks, gt_masks)
        else:
            image_mask_iou = torch.full((boxes.size(0),), float("nan"))

        start = object_offset
        stop = start + boxes.size(0)
        graph_records.append({
            "image_id": str(image_id),
            "object_start": start,
            "object_stop": stop,
            "rel_pairs": batch.get("rel_pairs", torch.zeros(0, 2, dtype=torch.long)).cpu(),
            "rel_labels": batch.get("rel_labels", torch.zeros(0, dtype=torch.long)).cpu(),
        })
        object_offset = stop
        labels.append(entity_labels.cpu() - 1)
        image_ids.extend([str(image_id)] * boxes.size(0))
        object_indices.extend(range(boxes.size(0)))
        areas.append(
            ((boxes[:, 2] - boxes[:, 0]).clamp_min(0)
             * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)).cpu()
        )
        mask_ious.append(image_mask_iou.cpu())

    if not labels:
        raise RuntimeError(
            "No PSG object masks were materialized. Verify image_root and panoptic_root."
        )
    expected_objects = sum(value.size(0) for value in labels)
    views = {name: torch.cat(values) for name, values in view_chunks.items()}
    if any(value.size(0) != expected_objects for value in views.values()):
        raise RuntimeError("Experiment I-A feature views are not object-aligned")
    payload = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "views": views,
        "labels": torch.cat(labels),
        "image_ids": image_ids,
        "object_indices": torch.tensor(object_indices, dtype=torch.long),
        "areas": torch.cat(areas),
        "mask_iou": torch.cat(mask_ious),
        "graph_records": graph_records,
        "skipped_images": skipped,
        "backbone_provenance": encoder.provenance(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _aggregate_runs(runs: list[dict], metric_key: str = "metrics") -> dict:
    keys = (
        "top1_accuracy", "top5_accuracy", "macro_accuracy", "ece_15",
        "adaptive_ece_15", "nll", "brier",
    )
    return {
        key: {
            "mean": float(np.mean([run[metric_key][key] for run in runs])),
            "std": float(np.std([run[metric_key][key] for run in runs], ddof=0)),
        }
        for key in keys
    }


def _subset_image_ids(image_ids, selected: torch.Tensor) -> list[str]:
    return [
        str(image_id)
        for image_id, keep in zip(image_ids, selected.tolist())
        if keep
    ]


def _mask_quality_summary(mask_iou: torch.Tensor) -> dict:
    finite = mask_iou[torch.isfinite(mask_iou)].float()
    if not finite.numel():
        return {"support": 0, "available": False}
    return {
        "support": int(finite.numel()),
        "available": True,
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "quantiles": {
            "q25": float(torch.quantile(finite, 0.25)),
            "q75": float(torch.quantile(finite, 0.75)),
        },
        "threshold_support": {
            f"iou>={threshold:.2f}": int((finite >= threshold).sum())
            for threshold in (0.75, 0.85, 0.90, 0.95)
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psg_train_ann")
    parser.add_argument("--psg_eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--panoptic_root")
    parser.add_argument("--predicted_mask_train_dir")
    parser.add_argument("--predicted_mask_eval_dir")
    parser.add_argument("--train_cache")
    parser.add_argument("--eval_cache")
    parser.add_argument(
        "--source_summary",
        help="Formal source summary supplying ontology and mask provenance in cache-only mode.",
    )
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="data/derived/features/experiment_1a")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--feature_normalization", choices=("none", "zscore", "l2"),
        default="none",
        help="Fit normalization on probe-training objects only; apply unchanged to validation/eval.",
    )
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--roi_chunk_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _normalise_probe_features(train_features: torch.Tensor,
                              eval_features: torch.Tensor,
                              train_mask: torch.Tensor,
                              mode: str):
    """Normalize frozen features without using validation or evaluation data."""
    train = train_features.float()
    evaluation = eval_features.float()
    if mode == "none":
        return train, evaluation, {"mode": "none"}
    if mode == "l2":
        return (
            F.normalize(train, dim=1),
            F.normalize(evaluation, dim=1),
            {"mode": "l2", "fit_partition": "per_object_no_fitted_statistics"},
        )
    fitted = train[train_mask]
    mean = fitted.mean(dim=0)
    scale = fitted.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (
        (train - mean) / scale,
        (evaluation - mean) / scale,
        {
            "mode": "zscore",
            "fit_partition": "probe_training_objects_only",
            "mean": mean,
            "scale": scale,
        },
    )


def main():
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_only = any((args.train_cache, args.eval_cache, args.source_summary))
    if cache_only and not all((args.train_cache, args.eval_cache, args.source_summary)):
        raise ValueError(
            "Cache-only refit requires --train_cache, --eval_cache, and --source_summary"
        )
    if not cache_only and not all((
        args.psg_train_ann, args.psg_eval_ann, args.image_root, args.panoptic_root,
    )):
        raise ValueError(
            "Feature extraction requires PSG annotations plus image and panoptic roots"
        )
    predicted_mask_train_root = (
        Path(args.predicted_mask_train_dir).expanduser().resolve()
        if args.predicted_mask_train_dir else None
    )
    predicted_mask_eval_root = (
        Path(args.predicted_mask_eval_dir).expanduser().resolve()
        if args.predicted_mask_eval_dir else None
    )
    if (predicted_mask_train_root is None) != (predicted_mask_eval_root is None):
        raise ValueError(
            "Provide both --predicted_mask_train_dir and --predicted_mask_eval_dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seeds[0])

    if cache_only:
        source_summary = json.loads(
            Path(args.source_summary).expanduser().resolve().read_text(encoding="utf-8")
        )
        if source_summary.get("backbone") != args.backbone:
            raise RuntimeError("Source-summary backbone does not match --backbone")
        train_cache = torch.load(
            Path(args.train_cache).expanduser().resolve(),
            map_location="cpu", weights_only=False,
        )
        eval_cache = torch.load(
            Path(args.eval_cache).expanduser().resolve(),
            map_location="cpu", weights_only=False,
        )
        for split, payload in (("train", train_cache), ("eval", eval_cache)):
            if payload.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
                raise RuntimeError(f"Unexpected {split} feature-cache schema")
            if not {"box", "gt_mask", "pred_mask"}.issubset(payload.get("views", {})):
                raise RuntimeError(f"Incomplete {split} feature views")
        train_provenance = train_cache.get("backbone_provenance", {})
        eval_provenance = eval_cache.get("backbone_provenance", {})
        if train_provenance != eval_provenance:
            raise RuntimeError("Train/eval backbone provenance mismatch")
        if (
            source_summary.get("backbone_provenance", {}).get("state_dict_sha256")
            != train_provenance.get("state_dict_sha256")
        ):
            raise RuntimeError("Source-summary and feature-cache backbone hashes differ")
        ontology_id = str(source_summary["ontology_id"])
        predicted_mask_protocol = source_summary.get("predicted_mask_protocol")
        train_mask_provenance = (
            predicted_mask_protocol.get("train_manifest")
            if isinstance(predicted_mask_protocol, dict) else None
        )
        eval_mask_provenance = (
            predicted_mask_protocol.get("eval_manifest")
            if isinstance(predicted_mask_protocol, dict) else None
        )
    else:
        train_mask_provenance = _validate_mask_root(
            predicted_mask_train_root, args.psg_train_ann
        )
        eval_mask_provenance = _validate_mask_root(
            predicted_mask_eval_root, args.psg_eval_ann
        )
        train_loader = build_psg_loader(
            args.psg_train_ann, args.train_samples,
            exclude_annotation_path=args.psg_eval_ann,
            image_root=args.image_root, panoptic_root=args.panoptic_root,
            include_proxy_features=False, include_raw_images=True,
        )
        eval_loader = build_psg_loader(
            args.psg_eval_ann, args.eval_samples,
            image_root=args.image_root, panoptic_root=args.panoptic_root,
            include_proxy_features=False, include_raw_images=True,
        )
        if train_loader.dataset.ontology_id != eval_loader.dataset.ontology_id:
            raise RuntimeError("PSG train/eval ontology mismatch")
        ontology_id = str(eval_loader.dataset.ontology_id)
        encoder = FrozenROIEncoder(
            args.backbone, torch.device(args.device), args.roi_chunk_size
        )
        backbone_sha256 = encoder.provenance()["state_dict_sha256"]
        train_cache = materialize_object_views(
            train_loader, encoder,
            cache_dir / _cache_key(
                args.backbone, "train", args.train_samples,
                predicted_mask_train_root, backbone_sha256,
            ),
            predicted_mask_train_root,
        )
        eval_cache = materialize_object_views(
            eval_loader, encoder,
            cache_dir / _cache_key(
                args.backbone, "eval", args.eval_samples,
                predicted_mask_eval_root, backbone_sha256,
            ),
            predicted_mask_eval_root,
        )
        del encoder
        predicted_mask_protocol = (
            {
                "role": "oracle_box_conditioned_segmentation_control",
                "generator": "SAM ViT-B",
                "prompt": "ground_truth_box",
                "autonomous_segmentation": False,
                "paper_label": "GT-box-prompted SAM mask",
                "claim_restriction": (
                    "Do not describe this condition as autonomous segmentation."
                ),
                "train_manifest": train_mask_provenance,
                "eval_manifest": eval_mask_provenance,
            }
            if predicted_mask_train_root is not None else None
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_labels = train_cache["labels"].long()
    eval_labels = eval_cache["labels"].long()
    num_classes = max(
        int(train_labels.max().item()), int(eval_labels.max().item())
    ) + 1
    train_mask, validation_mask = deterministic_image_split(
        train_cache["image_ids"], args.validation_fraction, seed=997
    )
    groups = frequency_groups(train_labels[train_mask], num_classes)
    seen_training_classes = sorted(set(train_labels[train_mask].tolist()))
    seen_training_class_set = set(seen_training_classes)
    seen_eval_mask = torch.tensor(
        [int(value) in seen_training_class_set for value in eval_labels.tolist()],
        dtype=torch.bool,
    )
    unseen_eval_classes = sorted(
        set(eval_labels.tolist()) - set(seen_training_classes)
    )

    views = sorted(set(train_cache["views"]) & set(eval_cache["views"]))
    all_runs = {}
    predictions = {}
    for view in views:
        train_view, eval_view, normalizer = _normalise_probe_features(
            train_cache["views"][view], eval_cache["views"][view],
            train_mask, args.feature_normalization,
        )
        view_runs = []
        for seed in args.seeds:
            seed_everything(seed)
            model, history = train_linear_probe(
                train_view, train_labels,
                train_mask, validation_mask, num_classes,
                seed=seed, device=args.device, epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
            validation_logits = batched_logits(
                model, train_view[validation_mask],
                args.device, args.probe_batch_size,
            )
            temperature = fit_temperature(
                validation_logits, train_labels[validation_mask]
            )
            eval_logits = batched_logits(
                model, eval_view, args.device, args.probe_batch_size
            )
            eval_prediction = (eval_logits / temperature).argmax(dim=1)
            metrics = evaluate_object_logits(
                eval_logits, eval_labels, eval_cache["image_ids"], groups,
                areas=eval_cache["areas"], mask_iou=eval_cache["mask_iou"],
                temperature=temperature,
            )
            seen_metrics = evaluate_object_logits(
                eval_logits[seen_eval_mask], eval_labels[seen_eval_mask],
                _subset_image_ids(eval_cache["image_ids"], seen_eval_mask), groups,
                areas=eval_cache["areas"][seen_eval_mask],
                mask_iou=eval_cache["mask_iou"][seen_eval_mask],
                temperature=temperature,
            )
            checkpoint = output_dir / f"{args.backbone}_{view}_seed{seed}.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": args.backbone,
                "feature_view": view,
                "seed": seed,
                "num_classes": num_classes,
                "temperature": temperature,
                "ontology_id": ontology_id,
                "feature_normalization": {
                    key: value for key, value in normalizer.items()
                    if key not in {"mean", "scale"}
                },
                "feature_mean": normalizer.get("mean"),
                "feature_scale": normalizer.get("scale"),
            }, checkpoint)
            np.savez_compressed(
                output_dir / f"{args.backbone}_{view}_seed{seed}_predictions.npz",
                labels=eval_labels.numpy(),
                predictions=eval_prediction.numpy(),
                confidence=(eval_logits / temperature).softmax(dim=1).max(dim=1).values.numpy(),
                image_ids=np.asarray(
                    [str(value) for value in eval_cache["image_ids"]], dtype=str
                ),
                mask_iou=eval_cache["mask_iou"].numpy(),
                areas=eval_cache["areas"].numpy(),
            )
            run = {
                "seed": seed,
                "view": view,
                "metrics": metrics,
                "seen_class_metrics": seen_metrics,
                "relationship_endpoints": relationship_endpoint_summary(
                    eval_prediction, eval_labels, eval_cache["graph_records"],
                    mask_iou=(
                        eval_cache["mask_iou"] if view == "pred_mask" else None
                    ),
                    bootstrap_seed=seed + 3000,
                ),
                "history": history,
                "probe_training": {
                    "max_epochs": args.probe_epochs,
                    "epochs_ran": len(history),
                    "best_epoch": min(
                        history, key=lambda row: row["validation_cross_entropy"]
                    )["epoch"],
                    "stopped_early": len(history) < args.probe_epochs,
                    "early_stopping_patience": args.early_stopping_patience,
                    "early_stopping_min_delta": args.early_stopping_min_delta,
                },
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            view_runs.append(run)
            predictions[(view, seed)] = eval_prediction
            del model
        all_runs[view] = {
            "runs": view_runs,
            "aggregate": _aggregate_runs(view_runs),
            "seen_class_aggregate": _aggregate_runs(
                view_runs, metric_key="seen_class_metrics"
            ),
        }
        del train_view, eval_view

    paired = {}
    paired_views = {}
    for seed in args.seeds:
        if ("box", seed) in predictions and ("gt_mask", seed) in predictions:
            paired[str(seed)] = paired_accuracy_delta(
                predictions[("box", seed)], predictions[("gt_mask", seed)],
                eval_labels, eval_cache["image_ids"], seed=seed + 1000,
            )
        for baseline, candidate in (
            ("box", "gt_mask"),
            ("box", "pred_mask"),
            ("gt_mask", "pred_mask"),
        ):
            if (baseline, seed) not in predictions or (candidate, seed) not in predictions:
                continue
            key = f"{candidate}_minus_{baseline}"
            paired_views.setdefault(key, {})[str(seed)] = paired_accuracy_delta(
                predictions[(baseline, seed)], predictions[(candidate, seed)],
                eval_labels, eval_cache["image_ids"],
                seed=seed + 1000 + len(paired_views),
            )

    formal_protocol = (
        args.train_samples >= 5000
        and args.eval_samples >= 1000
        and len(args.seeds) >= 3
        and args.probe_epochs >= 100
        and args.early_stopping_patience >= 10
    )
    protocol_warnings = []
    if not formal_protocol:
        protocol_warnings.append(
            "Smoke-scale run: do not use these estimates as paper evidence."
        )
    if int(seen_eval_mask.sum()) != int(eval_labels.numel()):
        protocol_warnings.append(
            "Primary all-class metrics include evaluation classes without probe-training positives; "
            "use seen_class_metrics to separate closed-set decodability."
        )
    convergence = {
        view: {
            str(run["seed"]): run["probe_training"]
            for run in payload["runs"]
        }
        for view, payload in all_runs.items()
    }
    unconverged = [
        f"{view}/seed{seed}"
        for view, runs in convergence.items()
        for seed, details in runs.items()
        if not details["stopped_early"]
    ]
    if unconverged:
        protocol_warnings.append(
            "Probe reached max epochs without early stopping: "
            + ", ".join(unconverged)
        )
    summary = {
        "experiment": "I-A_segmentation_conditioned_object_identity",
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "research_question": (
            "Does high-quality object localization or segmentation suffice for "
            "correct object identity in frozen visual foundation features?"
        ),
        "claim_scope": (
            "Frozen-feature object decodability and calibration. Relationship "
            "reasoning depth is Experiment I-B; standard SGG is Experiment IV."
        ),
        "backbone": args.backbone,
        "backbone_provenance": train_cache["backbone_provenance"],
        "ontology_id": ontology_id,
        "num_classes": num_classes,
        "train_objects": int(train_labels.numel()),
        "eval_objects": int(eval_labels.numel()),
        "feature_views": views,
        "predicted_masks_available": "pred_mask" in views,
        "predicted_mask_protocol": predicted_mask_protocol,
        "paper_evidence_tier": (
            "full_mask_iou_conditioned" if "pred_mask" in views
            else "gt_mask_and_box_probe_only"
        ),
        "unseen_eval_classes": unseen_eval_classes,
        "seen_class_coverage": {
            "training_classes": len(seen_training_classes),
            "eval_objects_seen": int(seen_eval_mask.sum()),
            "eval_objects_total": int(eval_labels.numel()),
            "eval_object_fraction": float(seen_eval_mask.float().mean()),
        },
        "head_body_tail_from_training_frequency": groups,
        "image_disjoint_probe_split": {
            "train_objects": int(train_mask.sum()),
            "validation_objects": int(validation_mask.sum()),
            "validation_fraction": args.validation_fraction,
        },
        "views": all_runs,
        "paired_gt_mask_minus_box": paired,
        "paired_view_accuracy_deltas": paired_views,
        "predicted_mask_quality": _mask_quality_summary(eval_cache["mask_iou"]),
        "protocol_qualification": {
            "formal_scale": formal_protocol,
            "all_probes_early_stopped": not unconverged,
            "warnings": protocol_warnings,
        },
        "probe_convergence": convergence,
        "skipped_train_images": train_cache["skipped_images"],
        "skipped_eval_images": eval_cache["skipped_images"],
        "config": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Experiment I-A complete: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
