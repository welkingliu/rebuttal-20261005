#!/usr/bin/env python3
"""Finalize task-specific legacy exports into one strict prediction cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgg_core.models.prediction_cache_writer import CACHE_SCHEMA


TASKS = ("predcls", "sgcls", "sgdet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--expected_images", type=int, default=26446)
    args = parser.parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    tasks = tuple(dict.fromkeys(args.tasks))
    states = {
        task: json.loads((root / f"state_{task}.json").read_text())
        for task in tasks
    }
    first = states[tasks[0]]
    identity = (
        "model_name", "architecture_family", "source_commit", "ontology_id",
        "legacy_format", "effect_type", "images", "image_ids_sha256",
        "vg_test_index_sha256",
    )
    image_sets = {}
    for task, state in states.items():
        if state["task"] != task:
            raise RuntimeError(f"Task-state mismatch for {task}")
        for field in identity:
            if state[field] != first[field]:
                raise RuntimeError(f"Legacy exports differ on {field}")
        expected_flags = {
            "use_gt_box": task in ("predcls", "sgcls"),
            "use_gt_object_label": task == "predcls",
        }
        if state["task_flags"] != expected_flags:
            raise RuntimeError(f"Invalid task flags for {task}")
        image_sets[task] = {
            path.stem for path in (root / "predictions" / task).glob("*.npz")
        }
        if len(image_sets[task]) != int(args.expected_images):
            raise RuntimeError(
                f"{task} coverage={len(image_sets[task])}/{args.expected_images}"
            )
    if len({frozenset(values) for values in image_sets.values()}) != 1:
        raise RuntimeError("Legacy task exports use different image IDs")

    metadata = {
        "schema": CACHE_SCHEMA,
        "model_name": first["model_name"],
        "architecture_family": first["architecture_family"],
        "source_commit": first["source_commit"],
        "parameter_count": max(int(state["parameter_count"]) for state in states.values()),
        "parameter_count_by_task": {
            task: int(states[task]["parameter_count"]) for task in tasks
        },
        "checkpoint_sha256_by_task": {
            task: states[task]["checkpoint_sha256"] for task in tasks
        },
        "dataset": "vg",
        "ontology_id": first["ontology_id"],
        "split": "test",
        "tasks": list(tasks),
        "image_ids": sorted(image_sets[tasks[0]]),
        "images_by_task": {task: int(args.expected_images) for task in tasks},
        "task_flags": {task: states[task]["task_flags"] for task in tasks},
        "legacy_format": first["legacy_format"],
        "effect_type": first["effect_type"],
        "native_prediction_sha256_by_task": {
            task: states[task]["native_prediction_sha256"] for task in tasks
        },
    }
    output = root / "metadata.json"
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"metadata={output}")


if __name__ == "__main__":
    main()
