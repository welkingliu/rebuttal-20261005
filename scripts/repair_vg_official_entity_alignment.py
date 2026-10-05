#!/usr/bin/env python3
"""Record canonical VG entity rows retained by official PredCls/SGCls caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


CORRUPT_IMAGE_IDS = {1592, 1722, 4616, 4617}


def canonical_records(vg_h5: Path, image_data: Path):
    records = json.loads(image_data.read_text(encoding="utf-8"))
    with h5py.File(vg_h5, "r") as handle:
        row_count = len(handle["split"])
    if len(records) != row_count:
        records = [
            record for record in records
            if int(record.get("image_id", -1)) not in CORRUPT_IMAGE_IDS
        ]
    if len(records) != row_count:
        raise ValueError(
            f"image_data/H5 row mismatch: {len(records)} != {row_count}"
        )
    return records


def normalized_boxes(raw_boxes, width: int, height: int):
    values = np.asarray(raw_boxes, dtype=np.float32)
    size = np.maximum(values[:, 2:], 0.0)
    boxes = np.concatenate(
        (values[:, :2] - size / 2.0, values[:, :2] + size / 2.0), axis=1
    )
    boxes = np.clip(boxes, 0.0, 1024.0)
    boxes *= max(width, height) / 1024.0
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(height))
    boxes /= np.asarray([width, height, width, height], dtype=np.float32)
    return boxes.astype(np.float32, copy=False)


def retained_indices(canonical_boxes, predicted_boxes, tolerance: float):
    cost = np.abs(
        canonical_boxes[:, None, :] - predicted_boxes[None, :, :]
    ).max(axis=2)
    canonical_rows, prediction_rows = linear_sum_assignment(cost)
    mapping = np.full(len(predicted_boxes), -1, dtype=np.int64)
    mapping[prediction_rows] = canonical_rows
    if (mapping < 0).any():
        raise RuntimeError("Not every official prediction row was aligned")
    distances = cost[mapping, np.arange(len(predicted_boxes))]
    if distances.size and float(distances.max()) > tolerance:
        raise RuntimeError(
            "Official/canonical box alignment exceeds tolerance: "
            f"{float(distances.max()):.6f} > {tolerance:.6f}"
        )
    return mapping, distances


def replace_npz(path: Path, arrays: dict):
    temporary = path.with_name(path.name + ".alignment.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--vg_h5", required=True)
    parser.add_argument("--image_data", required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=("predcls", "sgcls"),
        default=["predcls", "sgcls"],
    )
    parser.add_argument("--box_tolerance", type=float, default=0.01)
    parser.add_argument("--manifest")
    parser.add_argument("--output_manifest")
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    vg_h5 = Path(args.vg_h5).expanduser().resolve()
    image_data = Path(args.image_data).expanduser().resolve()
    records = canonical_records(vg_h5, image_data)
    repairs = []

    with h5py.File(vg_h5, "r") as handle:
        split = np.asarray(handle["split"])
        first_box = np.asarray(handle["img_to_first_box"])
        last_box = np.asarray(handle["img_to_last_box"])
        first_rel = np.asarray(handle["img_to_first_rel"])
        test_rows = np.where(
            (split == 2) & (first_box >= 0) & (first_rel >= 0)
        )[0]

        for row in test_rows:
            record = records[int(row)]
            image_id = str(record["image_id"])
            first = int(first_box[row])
            last = int(last_box[row])
            canonical = normalized_boxes(
                handle["boxes_1024"][first:last + 1],
                int(record["width"]),
                int(record["height"]),
            )
            for task in args.tasks:
                path = cache_root / "predictions" / task / f"{image_id}.npz"
                with np.load(path, allow_pickle=False) as payload:
                    arrays = {
                        key: np.asarray(payload[key]) for key in payload.files
                    }
                entity_count = int(arrays["pred_entity_scores"].shape[0])
                if entity_count == len(canonical):
                    continue
                if entity_count > len(canonical):
                    raise RuntimeError(
                        f"{task}/{image_id}: official entities exceed canonical "
                        f"entities ({entity_count} > {len(canonical)})"
                    )
                mapping, distances = retained_indices(
                    canonical, np.asarray(arrays["pred_boxes"]),
                    float(args.box_tolerance),
                )
                arrays["gt_entity_indices"] = mapping
                replace_npz(path, arrays)
                repair = {
                    "task": task,
                    "image_id": image_id,
                    "canonical_entities": int(len(canonical)),
                    "official_entities": entity_count,
                    "filtered_canonical_rows": sorted(
                        set(range(len(canonical))) - set(mapping.tolist())
                    ),
                    "maximum_box_error": (
                        float(distances.max()) if distances.size else 0.0
                    ),
                }
                repairs.append(repair)
                print(json.dumps(repair), flush=True)

    expected = 3 * len(tuple(dict.fromkeys(args.tasks)))
    if len(repairs) != expected:
        raise RuntimeError(
            f"Expected {expected} task/image repairs, observed {len(repairs)}"
        )

    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["canonical_entity_alignment"] = {
        "status": "explicit",
        "field": "gt_entity_indices",
        "box_tolerance": float(args.box_tolerance),
        "repairs": repairs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    for task in dict.fromkeys(args.tasks):
        state_path = cache_root / f"state_{task}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["canonical_entity_alignment"] = {
            "status": "explicit",
            "field": "gt_entity_indices",
            "repaired_images": [
                repair["image_id"] for repair in repairs
                if repair["task"] == task
            ],
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    if bool(args.manifest) != bool(args.output_manifest):
        raise ValueError("--manifest and --output_manifest must be used together")
    if args.manifest:
        manifest = json.loads(
            Path(args.manifest).expanduser().read_text(encoding="utf-8")
        )
        manifest["config"]["test_prediction_cache_root"] = str(cache_root)
        manifest["config"]["test_cache_entity_alignment"] = (
            "explicit_gt_entity_indices"
        )
        output_manifest = Path(args.output_manifest).expanduser().resolve()
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"manifest={output_manifest}")

    print(f"repairs={len(repairs)}")


if __name__ == "__main__":
    main()
