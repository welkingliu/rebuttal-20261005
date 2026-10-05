"""Validate official SGG prediction-cache schema and exact image coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from sgg_core.models.prediction_cache import CACHE_SCHEMA, REQUIRED_ARRAYS, sha256_file
from sgg_core.protocol import build_loaders


def _dataset_image_ids(dataset) -> list[str] | None:
    """Read stable image IDs without materializing images, masks, or features."""
    items = getattr(dataset, "items", None)
    if items is not None:
        values = []
        for item in items:
            if isinstance(item, dict):
                value = item.get("image_id", item.get("img_id"))
            elif isinstance(item, (tuple, list)) and item:
                value = item[0]
            else:
                return None
            if value is None:
                return None
            values.append(str(value.item() if hasattr(value, "item") else value))
        if len(values) == len(dataset):
            return values

    index = getattr(dataset, "_index", None)
    if index is not None and hasattr(index, "image_id"):
        return [str(index.image_id(idx)) for idx in range(len(dataset))]

    image_indices = getattr(dataset, "image_indices", None)
    image_meta = getattr(dataset, "index_to_image_meta", None)
    if image_indices is not None and image_meta is not None:
        values = []
        for index_value in image_indices:
            index_value = int(index_value)
            meta = image_meta.get(index_value, {})
            values.append(str(meta.get("image_id", index_value)))
        if len(values) == len(dataset):
            return values
    return None


def _loader_image_ids(loader: Iterable[dict]) -> list[str]:
    values = _dataset_image_ids(loader.dataset)
    if values is not None:
        return values
    values = []
    for batch in loader:
        value = batch.get("image_id", batch.get("img_id"))
        if hasattr(value, "item"):
            value = value.item()
        values.append(str(value))
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--dataset", required=True, choices=("vg", "oi", "psg", "gqa", "vrd"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_ann")
    parser.add_argument("--eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--panoptic_root")
    parser.add_argument("--eval_samples", type=int, default=1_000_000_000)
    parser.add_argument("--report")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failures = []
    if metadata.get("schema") != CACHE_SCHEMA:
        failures.append(f"schema={metadata.get('schema')}")
    _, loader = build_loaders(
        args.dataset, args.data_root, 1, args.eval_samples,
        args.train_ann, args.eval_ann, args.image_root, args.panoptic_root,
        include_proxy_features=False,
        include_raw_images=False,
    )
    ontology_id = getattr(loader.dataset, "ontology_id", None)
    if metadata.get("ontology_id") != ontology_id:
        failures.append(
            f"ontology={metadata.get('ontology_id')} expected={ontology_id}"
        )
    expected_ids = _loader_image_ids(loader)
    invalid_files = []
    coverage = {}
    for task in metadata.get("tasks", []):
        missing = []
        for image_id in expected_ids:
            safe = image_id.replace("/", "_").replace("\\", "_")
            path = root / "predictions" / task / f"{safe}.npz"
            if not path.is_file():
                missing.append(image_id)
                continue
            try:
                with np.load(path, allow_pickle=False) as payload:
                    absent = [key for key in REQUIRED_ARRAYS if key not in payload]
                    if absent:
                        invalid_files.append(f"{path}:{absent}")
            except (OSError, ValueError) as exc:
                invalid_files.append(f"{path}:{exc}")
        coverage[task] = {
            "expected": len(expected_ids),
            "present": len(expected_ids) - len(missing),
            "missing": len(missing),
            "missing_examples": missing[:20],
        }
        if missing:
            failures.append(f"{task}_coverage={coverage[task]['present']}/{len(expected_ids)}")
    if invalid_files:
        failures.append(f"invalid_prediction_files={len(invalid_files)}")
    report = {
        "status": "ready" if not failures else "not_ready",
        "cache_root": str(root),
        "metadata_sha256": sha256_file(metadata_path),
        "dataset": args.dataset,
        "ontology_id": ontology_id,
        "coverage": coverage,
        "invalid_examples": invalid_files[:20],
        "failures": failures,
    }
    report_path = (
        Path(args.report).expanduser().resolve() if args.report
        else root / "validation_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
