#!/usr/bin/env python3
"""Finalize a complete PySGG VG cache for one or all three SGG tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgg_core.models.prediction_cache_writer import CACHE_SCHEMA

TASKS = ("predcls", "sgcls", "sgdet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--expected_images", type=int, default=26446)
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS),
        help="Task subset to finalize; the formal tri-task default uses all three.",
    )
    args = parser.parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    tasks = tuple(dict.fromkeys(args.tasks))
    states = {
        task: json.loads((root / ("state_" + task + ".json")).read_text())
        for task in tasks
    }
    identity_fields = (
        "model_name", "architecture_family", "source_commit",
        "ontology_id", "images",
    )
    first = states[tasks[0]]
    for task, state in states.items():
        if state.get("task") != task:
            raise RuntimeError("PySGG task-state mismatch")
        for field in identity_fields:
            if state.get(field) != first.get(field):
                raise RuntimeError("PySGG task exports differ on " + field)
        count = len(list((root / "predictions" / task).glob("*.npz")))
        if count != int(args.expected_images) or state["images"] != count:
            raise RuntimeError(
                "%s coverage=%d/%d" % (task, count, args.expected_images)
            )
    task_flags = {task: states[task]["task_flags"] for task in tasks}
    expected_flags = {
        "predcls": {"use_gt_box": True, "use_gt_object_label": True},
        "sgcls": {"use_gt_box": True, "use_gt_object_label": False},
        "sgdet": {"use_gt_box": False, "use_gt_object_label": False},
    }
    if task_flags != {task: expected_flags[task] for task in tasks}:
        raise RuntimeError("Invalid PredCls/SGCls/SGDet task flags")
    image_sets = {
        task: {path.stem for path in (root / "predictions" / task).glob("*.npz")}
        for task in tasks
    }
    if len({frozenset(values) for values in image_sets.values()}) != 1:
        raise RuntimeError("PySGG task caches use different image IDs")
    metadata = {
        "schema": CACHE_SCHEMA,
        "model_name": first["model_name"],
        "architecture_family": first["architecture_family"],
        "source_commit": first["source_commit"],
        # Some official PredCls implementations omit the object-classification
        # branch at construction time, so task-specific parameter counts may
        # legitimately differ even within one architecture family.
        "parameter_count": max(
            int(states[task]["parameter_count"]) for task in tasks
        ),
        "parameter_count_by_task": {
            task: int(states[task]["parameter_count"]) for task in tasks
        },
        "checkpoint_sha256_by_task": {
            task: states[task]["checkpoint_sha256"] for task in tasks
        },
        "config_sha256_by_task": {
            task: states[task]["config_sha256"] for task in tasks
        },
        "dataset": "vg",
        "ontology_id": first["ontology_id"],
        "split": "test",
        "tasks": list(tasks),
        "image_ids": sorted(image_sets[tasks[0]]),
        "images_by_task": {task: args.expected_images for task in tasks},
        "task_flags": task_flags,
    }
    path = root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("metadata=" + str(path))


if __name__ == "__main__":
    main()
