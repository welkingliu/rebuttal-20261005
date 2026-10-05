#!/usr/bin/env python3
"""Convert official Kaihua/KERN VG outputs to the unified NumPy cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.vg_ontology import assert_vg150_alignment


CORRUPT_IMAGE_IDS = {1592, 1722, 4616, 4617}
TASK_FLAGS = {
    "predcls": {"use_gt_box": True, "use_gt_object_label": True},
    "sgcls": {"use_gt_box": True, "use_gt_object_label": False},
    "sgdet": {"use_gt_box": False, "use_gt_object_label": False},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _parameter_count(checkpoint):
    payload = _torch_load(checkpoint)
    for key in ("model", "state_dict"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not expose a model/state_dict mapping")
    count = sum(int(value.numel()) for value in payload.values() if torch.is_tensor(value))
    if count <= 0:
        raise ValueError("Checkpoint contains no tensor parameters")
    return count


def _test_indices(vg_h5):
    with h5py.File(str(vg_h5), "r") as handle:
        split = np.asarray(handle["split"])
        first_box = np.asarray(handle["img_to_first_box"])
        first_rel = np.asarray(handle["img_to_first_rel"])
    return np.where((split == 2) & (first_box >= 0) & (first_rel >= 0))[0]


def _test_records(vg_h5, image_data):
    with h5py.File(str(vg_h5), "r") as handle:
        row_count = len(handle["split"])
    records = json.loads(Path(image_data).read_text(encoding="utf-8"))
    if len(records) != row_count:
        records = [
            record for record in records
            if int(record.get("image_id", -1)) not in CORRUPT_IMAGE_IDS
        ]
    if len(records) != row_count:
        raise ValueError(
            f"image_data/H5 row mismatch: {len(records)} != {row_count}"
        )
    indices = _test_indices(vg_h5)
    return [records[int(index)] for index in indices]


def _sparse_entity_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("Object label/score rows differ")
    if labels.size and (labels.min() < 1 or labels.max() >= 151):
        raise ValueError("Object labels are outside canonical VG-150 IDs")
    output = np.zeros((len(labels), 151), dtype=np.float32)
    if labels.size:
        output[:, 0] = np.clip(1.0 - scores, 0.0, 1.0)
        output[np.arange(len(labels)), labels] = scores
    return output


def _validate_arrays(arrays):
    boxes = np.asarray(arrays["pred_boxes"], dtype=np.float32)
    pairs = np.asarray(arrays["pred_rel_pairs"], dtype=np.int64)
    rel_scores = np.asarray(arrays["pred_rel_scores"], dtype=np.float32)
    entities = np.asarray(arrays["pred_entity_scores"], dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("pred_boxes must be [N,4]")
    if entities.shape != (len(boxes), 151):
        raise ValueError("pred_entity_scores must be [N,151]")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pred_rel_pairs must be [M,2]")
    if rel_scores.shape != (len(pairs), 51):
        raise ValueError("pred_rel_scores must be [M,51]")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= len(boxes)):
        raise ValueError("Relation pair references an invalid object row")
    for name, value in arrays.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Non-finite values in {name}")


def _kaihua_arrays(prediction):
    prediction = prediction.convert("xyxy")
    width, height = map(float, prediction.size)
    boxes = prediction.bbox.detach().cpu().numpy().astype(np.float32)
    boxes /= np.asarray([width, height, width, height], dtype=np.float32)
    boxes = np.clip(boxes, 0.0, 1.0)
    labels = prediction.get_field("pred_labels").detach().cpu().numpy()
    scores = prediction.get_field("pred_scores").detach().cpu().numpy()
    return {
        "pred_boxes": boxes,
        "pred_entity_scores": _sparse_entity_scores(labels, scores),
        "pred_box_scores": np.ones(len(labels), dtype=np.float32),
        "pred_rel_pairs": prediction.get_field("rel_pair_idxs").detach().cpu().numpy(),
        "pred_rel_scores": prediction.get_field("pred_rel_scores").detach().cpu().numpy(),
    }


def _kern_arrays(entry, image_record):
    width = float(image_record["width"])
    height = float(image_record["height"])
    scale = 1024.0 / max(width, height)
    denominator = np.asarray(
        [width * scale, height * scale, width * scale, height * scale],
        dtype=np.float32,
    )
    boxes = np.asarray(
        entry.get("pred_boxes", entry.get("boxes")), dtype=np.float32
    ) / denominator
    labels = np.asarray(
        entry.get("pred_classes", entry.get("objects")), dtype=np.int64
    )
    scores = np.asarray(
        entry.get("obj_scores", entry.get("object_scores")), dtype=np.float32
    )
    pairs = entry.get("pred_rel_inds", entry.get("predicates"))
    relation_scores = entry.get("rel_scores", entry.get("predicate_scores"))
    if pairs is None or relation_scores is None:
        raise KeyError("KERN entry lacks relation pairs or relation scores")
    return {
        "pred_boxes": np.clip(boxes, 0.0, 1.0),
        "pred_entity_scores": _sparse_entity_scores(labels, scores),
        "pred_box_scores": np.ones(len(labels), dtype=np.float32),
        "pred_rel_pairs": np.asarray(pairs, dtype=np.int64),
        "pred_rel_scores": np.asarray(relation_scores, dtype=np.float32),
    }


def _load_predictions(kind, path, source_root):
    if kind == "kaihua":
        sys.path.insert(0, str(source_root))
        payload = _torch_load(path)
        if not isinstance(payload, dict) or "predictions" not in payload:
            raise ValueError("Kaihua input must be eval_results.pytorch")
        return payload["predictions"]
    try:
        import dill
        with Path(path).open("rb") as handle:
            return dill.load(handle)
    except ImportError:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("kaihua", "kern"), required=True)
    parser.add_argument("--task", choices=tuple(TASK_FLAGS), required=True)
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--canonical_dict", required=True)
    parser.add_argument("--native_dict", required=True)
    parser.add_argument("--vg_h5", required=True)
    parser.add_argument("--native_vg_h5")
    parser.add_argument("--image_data", required=True)
    parser.add_argument("--effect_type", default="none")
    parser.add_argument("--expected_images", type=int, default=26446)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    prediction_file = Path(args.prediction_file).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    for path in (source_root, prediction_file, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint_sha256 = sha256(checkpoint)
    parameter_count = _parameter_count(checkpoint)
    ontology = assert_vg150_alignment(args.canonical_dict, args.native_dict)
    canonical_indices = _test_indices(args.vg_h5)
    native_indices = _test_indices(args.native_vg_h5 or args.vg_h5)
    if not np.array_equal(canonical_indices, native_indices):
        raise RuntimeError(
            "Native and canonical VG test split/order differ; refusing positional mapping"
        )
    records = _test_records(args.vg_h5, args.image_data)
    if int(args.expected_images) > len(records):
        raise RuntimeError(
            f"Canonical VG test split has only {len(records)} images, expected {args.expected_images}"
        )
    records = records[:int(args.expected_images)]
    predictions = _load_predictions(args.format, prediction_file, source_root)
    if len(predictions) != len(records):
        raise RuntimeError(
            f"Prediction/test ordering mismatch: {len(predictions)} != {len(records)}"
        )

    marker = json.loads((source_root / ".official_source.json").read_text())
    output = cache_root / "predictions" / args.task
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    image_ids = []
    for index, (prediction, record) in enumerate(zip(predictions, records)):
        image_id = str(record["image_id"])
        image_ids.append(image_id)
        destination = output / f"{image_id}.npz"
        if args.resume and destination.is_file():
            continue
        arrays = (
            _kaihua_arrays(prediction)
            if args.format == "kaihua"
            else _kern_arrays(prediction, record)
        )
        _validate_arrays(arrays)
        np.savez_compressed(str(destination), **arrays)
        completed = index + 1
        if completed % args.log_every == 0 or completed == len(records):
            print(json.dumps({
                "task": args.task,
                "completed": completed,
                "total": len(records),
                "images_per_second": completed / max(time.monotonic() - started, 1e-6),
            }), flush=True)

    state = {
        "schema": "legacy_vg_task_export_v1",
        "model_name": args.model_name,
        "architecture_family": args.family,
        "legacy_format": args.format,
        "task": args.task,
        "source_commit": marker["commit"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "native_prediction_file": str(prediction_file),
        "native_prediction_sha256": sha256(prediction_file),
        "parameter_count": parameter_count,
        "ontology_id": ontology["ontology_id"],
        "ontology_alignment": ontology,
        "vg_test_index_sha256": hashlib.sha256(
            canonical_indices.astype(np.int64).tobytes()
        ).hexdigest(),
        "images": len(image_ids),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(image_ids).encode("utf-8")
        ).hexdigest(),
        "effect_type": str(args.effect_type),
        "task_flags": TASK_FLAGS[args.task],
    }
    state_path = cache_root / f"state_{args.task}.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"state={state_path}")


if __name__ == "__main__":
    main()
