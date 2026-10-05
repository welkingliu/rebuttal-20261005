#!/usr/bin/env python3
"""Compare the canonical VG-SGG H5 loader with RelTR's released COCO JSON."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.data.data_utils import VGTestDataset


def _official_boxes(annotations: list[dict], width: int, height: int) -> np.ndarray:
    boxes = np.asarray([item["bbox"] for item in annotations], dtype=np.float32)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, [0, 2]] /= float(width)
    boxes[:, [1, 3]] /= float(height)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vg_root", required=True)
    parser.add_argument("--reltr_protocol_root", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_examples", type=int, default=10)
    args = parser.parse_args()

    protocol_root = Path(args.reltr_protocol_root).expanduser().resolve()
    coco = json.loads((protocol_root / "test.json").read_text(encoding="utf-8"))
    relation_payload = json.loads(
        (protocol_root / "rel.json").read_text(encoding="utf-8")
    )["test"]
    official_images = coco["images"]
    official_annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        official_annotations[int(annotation["image_id"])].append(annotation)

    sample_count = min(int(args.samples), len(official_images))
    dataset = VGTestDataset(
        args.vg_root,
        num_samples=sample_count,
        split=2,
        include_proxy_features=False,
        require_relations=True,
        include_raw_images=False,
    )
    counters = defaultdict(int)
    examples = []
    max_box_error = 0.0
    total_h5_relations = 0
    total_official_relations = 0
    for index in range(sample_count):
        batch = dataset[index]
        official_image = official_images[index]
        image_id = int(batch["image_id"])
        official_id = int(official_image["id"])
        if image_id != official_id:
            counters["image_id"] += 1
            if len(examples) < args.max_examples:
                examples.append({
                    "index": index,
                    "kind": "image_id",
                    "h5": image_id,
                    "official": official_id,
                })
            continue

        annotations = official_annotations[official_id]
        labels = np.asarray(
            [item["category_id"] for item in annotations], dtype=np.int64
        )
        h5_labels = batch["entity_labels"].numpy()
        if labels.shape != h5_labels.shape or not np.array_equal(labels, h5_labels):
            counters["labels"] += 1

        boxes = _official_boxes(
            annotations, int(official_image["width"]), int(official_image["height"])
        )
        h5_boxes = batch["boxes"].numpy()
        if boxes.shape != h5_boxes.shape:
            counters["box_count"] += 1
        else:
            error = float(np.max(np.abs(boxes - h5_boxes))) if boxes.size else 0.0
            max_box_error = max(max_box_error, error)
            if error > 2e-3:
                counters["boxes"] += 1

        official_relations = np.asarray(
            relation_payload[str(official_id)], dtype=np.int64
        )
        h5_relations = np.column_stack((
            batch["rel_pairs"].numpy(), batch["rel_labels"].numpy()
        ))
        total_h5_relations += int(h5_relations.shape[0])
        total_official_relations += int(official_relations.shape[0])
        if (
            official_relations.shape != h5_relations.shape
            or not np.array_equal(official_relations, h5_relations)
        ):
            counters["relations"] += 1
            official_set = {tuple(map(int, row)) for row in official_relations}
            h5_set = {tuple(map(int, row)) for row in h5_relations}
            if official_set == h5_set:
                counters["relation_order_only"] += 1
            elif official_set.issubset(h5_set):
                counters["official_relations_subset_of_h5"] += 1
            else:
                counters["relation_content_difference"] += 1

        if any(counters[key] for key in ("labels", "box_count", "boxes", "relations")):
            if len(examples) < args.max_examples:
                examples.append({
                    "index": index,
                    "kind": "annotation",
                    "image_id": image_id,
                    "h5_objects": int(h5_labels.size),
                    "official_objects": int(labels.size),
                    "h5_relations": int(h5_relations.shape[0]),
                    "official_relations": int(official_relations.shape[0]),
                })

    report = {
        "samples": sample_count,
        "mismatch_images": dict(sorted(counters.items())),
        "total_h5_relations": total_h5_relations,
        "total_official_relations": total_official_relations,
        "max_box_absolute_error": max_box_error,
        "examples": examples,
    }
    print(json.dumps(report, indent=2))
    if counters:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
