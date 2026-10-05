#!/usr/bin/env python3
"""Export released OpenPSG checkpoints to the strict PSG cache schema."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.models.prediction_cache_writer import OfficialPredictionCacheWriter


MODEL_SPECS = {
    "motifs": {
        "family": "Neural Motifs",
        "checkpoint": "openpsg/psg/motifs/epoch_12.pth",
    },
    "vctree": {
        "family": "VCTree",
        "checkpoint": "openpsg/psg/vctree/epoch_12.pth",
    },
    "psgtr": {
        "family": "PSGTR",
        "checkpoint": "openpsg/psg/psgtr/epoch_60.pth",
    },
    "psgformer": {
        "family": "PSGFormer",
        "checkpoint": "openpsg/psg/psgformer/epoch_60.pth",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ontology_id(annotation_path):
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    objects = payload["thing_classes"] + payload["stuff_classes"]
    predicates = payload["predicate_classes"]
    obj_vocab = {str(name).strip().lower(): i + 1 for i, name in enumerate(objects)}
    pred_vocab = {
        str(name).strip().lower(): i + 1 for i, name in enumerate(predicates)
    }
    canonical = json.dumps(
        {"objects": obj_vocab, "predicates": pred_vocab},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "psg:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_model(model_key, source_root, checkpoint_path, annotation_path, coco_root,
               native_panoptic=False, config_path=None):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import build_dataloader
    from mmdet.models import build_detector
    from openpsg.datasets import build_dataset

    if model_key in ("psgtr", "psgformer") and not native_panoptic:
        # Their default Result is panoptic-evaluation output: relation nodes are
        # mask-deduplicated while refine_bboxes remain pre-deduplication boxes.
        # Export the official non-panoptic branch so boxes, labels, and relation
        # pair indices share one entity axis for the unified box-IoU protocol.
        import openpsg.models.frameworks.psgtr as psgtr_framework

        original_triplet_to_result = psgtr_framework.triplet2Result

        def box_aligned_triplet_to_result(triplets, use_mask):
            return original_triplet_to_result(
                triplets, use_mask, eval_pan_rels=False
            )

        psgtr_framework.triplet2Result = box_aligned_triplet_to_result
    elif model_key in ("motifs", "vctree"):
        # Released checkpoints contain every object embedding parameter. Avoid
        # the legacy constructor's 862 MB GloVe download by supplying only a
        # shape-correct placeholder; strict checkpoint loading below must then
        # replace every placeholder parameter before inference.
        import openpsg.models.relation_heads.approaches.motif as motif_context
        import openpsg.models.relation_heads.approaches.vctree as vctree_context

        def checkpoint_embedding_placeholder(
            names, wv_dir, wv_type="glove.6B", wv_dim=300
        ):
            del wv_dir, wv_type
            return torch.zeros((len(names), int(wv_dim)), dtype=torch.float32)

        motif_context.obj_edge_vectors = checkpoint_embedding_placeholder
        vctree_context.obj_edge_vectors = checkpoint_embedding_placeholder

    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu")
    metadata = checkpoint_payload.get("meta", {})
    if config_path is None:
        config_text = metadata.get("config")
        if not config_text:
            raise RuntimeError("OpenPSG checkpoint does not contain its resolved config")
        cfg = Config.fromstring(config_text, ".py")
    else:
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg = Config.fromfile(str(config_path))
    del checkpoint_payload
    gc.collect()
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.workers_per_gpu = 0
    cfg.data.test.ann_file = str(annotation_path)
    cfg.data.test.img_prefix = str(coco_root)
    cfg.data.test.seg_prefix = str(coco_root)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(
        model, str(checkpoint_path), map_location="cpu", strict=True
    )
    if metadata.get("CLASSES"):
        model.CLASSES = metadata["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    model.PREDICATES = dataset.PREDICATES
    return model.cuda().eval(), dataset, data_loader


def as_numpy(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value.astype(dtype, copy=False) if dtype is not None else value


def convert_result(result, height, width):
    boxes_with_scores = as_numpy(result.refine_bboxes, np.float32)
    if boxes_with_scores is None:
        boxes_with_scores = np.zeros((0, 5), dtype=np.float32)
    if boxes_with_scores.ndim != 2 or boxes_with_scores.shape[1] < 4:
        raise ValueError("OpenPSG refine_bboxes must have shape [N,4+] ")
    boxes = boxes_with_scores[:, :4].copy()
    boxes /= np.asarray([width, height, width, height], dtype=np.float32)
    boxes = np.clip(boxes, 0.0, 1.0)
    box_scores = (
        np.clip(boxes_with_scores[:, 4], 0.0, 1.0)
        if boxes_with_scores.shape[1] > 4
        else np.ones((boxes.shape[0],), dtype=np.float32)
    )

    labels = result.refine_labels
    if labels is None:
        labels = result.labels
    labels = as_numpy(labels, np.int64)
    if labels is None:
        raise ValueError("OpenPSG result has no object labels")
    labels = labels.reshape(-1)
    if labels.shape[0] != boxes.shape[0]:
        raise ValueError("OpenPSG box/label count mismatch")
    if labels.size and (labels.min() < 1 or labels.max() > 133):
        raise ValueError("OpenPSG object labels are not PSG 1..133 IDs")

    entity_scores = as_numpy(result.refine_dists, np.float32)
    if entity_scores is None:
        # End-to-end OpenPSG outputs expose the selected class and a separate
        # detection confidence, but not the complete object class posterior.
        entity_scores = np.zeros((labels.shape[0], 134), dtype=np.float32)
        if labels.size:
            entity_scores[np.arange(labels.shape[0]), labels] = 1.0
    if entity_scores.shape != (labels.shape[0], 134):
        raise ValueError(
            "OpenPSG object distribution must have 134 columns, got "
            + str(entity_scores.shape)
        )

    rel_pairs = as_numpy(result.rel_pair_idxes, np.int64)
    rel_scores = as_numpy(result.rel_dists, np.float32)
    if rel_pairs is None:
        rel_pairs = np.zeros((0, 2), dtype=np.int64)
    if rel_scores is None:
        rel_scores = np.zeros((0, 57), dtype=np.float32)
    rel_pairs = rel_pairs.reshape(-1, 2)
    if rel_scores.ndim != 2 or rel_scores.shape != (rel_pairs.shape[0], 57):
        raise ValueError(
            "OpenPSG relation distribution must have 57 columns, got "
            + str(rel_scores.shape)
        )
    if rel_pairs.size and (rel_pairs.min() < 0 or rel_pairs.max() >= boxes.shape[0]):
        raise ValueError("OpenPSG relation pair references a missing object")
    for name, value in (
        ("boxes", boxes),
        ("entity_scores", entity_scores),
        ("box_scores", box_scores),
        ("rel_scores", rel_scores),
    ):
        if not np.isfinite(value).all():
            raise ValueError("Non-finite OpenPSG " + name)
    return {
        "pred_boxes": boxes,
        "pred_entity_scores": entity_scores,
        "pred_box_scores": box_scores,
        "pred_rel_pairs": rel_pairs,
        "pred_rel_scores": rel_scores,
    }


def valid_prediction(path):
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            return all(
                key in payload
                for key in (
                    "pred_boxes",
                    "pred_entity_scores",
                    "pred_box_scores",
                    "pred_rel_pairs",
                    "pred_rel_scores",
                )
            )
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--source_root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--annotation")
    parser.add_argument("--coco_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_samples", type=int, default=1_000_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    spec = MODEL_SPECS[args.model]
    source_root = Path(
        args.source_root or root / "external/official_repos/OpenPSG"
    ).expanduser().resolve()
    checkpoint = Path(
        args.checkpoint or root / "checkpoints/sgg/weights" / spec["checkpoint"]
    ).expanduser().resolve()
    annotation = Path(
        args.annotation or root / "data/psg/psg.json"
    ).expanduser().resolve()
    coco_root = Path(args.coco_root or root / "data/coco").expanduser().resolve()
    output_dir = Path(
        args.output_dir
        or root / ("artifacts/prediction_cache/openpsg_" + args.model + "_psg")
    ).expanduser().resolve()
    marker_path = source_root / ".official_source.json"
    for path in (source_root, checkpoint, annotation, coco_root, marker_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("OpenPSG export requires CUDA")

    model, dataset, loader = load_model(
        args.model, source_root, checkpoint, annotation, coco_root
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    source_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint_digest = sha256(checkpoint)
    ontology = ontology_id(annotation)
    model_name = "openpsg_" + args.model + "_psg_official"
    writer = OfficialPredictionCacheWriter(
        output_dir,
        model_name=model_name,
        architecture_family=spec["family"],
        source_commit=source_marker["commit"],
        parameter_count=parameter_count,
        checkpoint_sha256_by_task={"sgdet": checkpoint_digest},
        dataset="psg",
        ontology_id=ontology,
        split="test",
        tasks=("sgdet",),
    )
    limit = min(len(dataset), int(args.eval_samples))
    state = {
        "model_name": model_name,
        "source_commit": source_marker["commit"],
        "checkpoint_sha256": checkpoint_digest,
        "ontology_id": ontology,
        "split": "test",
        "images": limit,
        "relation_score_mode": "categorical",
        "prediction_protocol": (
            "official_non_panoptic_box_aligned"
            if args.model in ("psgtr", "psgformer")
            else "official_scene_graph_box_output"
        ),
        "word_vector_initialization": (
            "strict_checkpoint_embedded"
            if args.model in ("motifs", "vctree")
            else "not_applicable"
        ),
        "object_score_source": (
            "official_distribution"
            if args.model in ("motifs", "vctree")
            else "hard_label_plus_official_box_confidence"
        ),
    }
    state_path = output_dir / "export_state.json"
    existing = list((output_dir / "predictions/sgdet").glob("*.npz"))
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous != state:
            raise RuntimeError("Refusing to mix OpenPSG cache provenance")
    elif args.resume and existing:
        raise RuntimeError("Cannot resume OpenPSG predictions without export_state.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    from mmcv.parallel import MMDataParallel

    model = MMDataParallel(model, device_ids=[0])
    started = time.monotonic()
    processed = resumed = 0
    for index, data in enumerate(loader):
        if index >= limit:
            break
        info = dataset.data_infos[index]
        image_id = info["id"]
        prediction_path = output_dir / "predictions/sgdet" / (str(image_id) + ".npz")
        if args.resume and valid_prediction(prediction_path):
            writer.metadata["image_ids"].append(str(image_id))
            writer.metadata["images_by_task"]["sgdet"] += 1
            resumed += 1
        else:
            with torch.no_grad():
                outputs = model(return_loss=False, rescale=True, **data)
            if isinstance(outputs, (list, tuple)):
                if len(outputs) != 1:
                    raise ValueError("OpenPSG batch did not return one Result")
                result = outputs[0]
            else:
                result = outputs
            if not hasattr(result, "rel_pair_idxes"):
                raise ValueError("OpenPSG output is not a scene-graph Result")
            writer.add(
                "sgdet",
                image_id,
                **convert_result(result, info["height"], info["width"]),
            )
            processed += 1
        completed = index + 1
        if completed % args.log_every == 0 or completed == limit:
            writer.finalize()
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                json.dumps({
                    "model": args.model,
                    "completed": completed,
                    "total": limit,
                    "processed": processed,
                    "resumed": resumed,
                    "images_per_second": completed / elapsed,
                }),
                flush=True,
            )
    metadata_path = writer.finalize()
    print("metadata=" + str(metadata_path))
    print("parameter_count=" + str(parameter_count))
    print("checkpoint_sha256=" + checkpoint_digest)


if __name__ == "__main__":
    main()
