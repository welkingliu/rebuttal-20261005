#!/usr/bin/env python3
"""Strict preflight for the formal VG PredCls/SGCls/SGDet panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_TASKS = {"predcls", "sgcls", "sgdet"}
REQUIRED_FLAGS = {
    "predcls": {"use_gt_box": True, "use_gt_object_label": True},
    "sgcls": {"use_gt_box": True, "use_gt_object_label": False},
    "sgdet": {"use_gt_box": False, "use_gt_object_label": False},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--minimum_families", type=int, default=4)
    parser.add_argument("--expected_images", type=int, default=26446)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    rows = []
    failures = []
    for path in sorted((root / "checkpoints/sgg/manifests").glob("*.json")):
        payload = json.loads(path.read_text())
        datasets = {str(value).lower() for value in payload.get("supported_datasets", [])}
        tasks = {str(value).lower() for value in payload.get("supported_tasks", [])}
        if "vg" not in datasets or not REQUIRED_TASKS.issubset(tasks):
            continue
        cache = Path(str(payload.get("config", {}).get("prediction_cache_root", "")))
        metadata_path = cache / "metadata.json"
        problems = []
        if payload.get("execution_mode") != "prediction_cache":
            problems.append("execution_mode")
        if not metadata_path.is_file():
            problems.append("metadata")
        else:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("task_flags") != REQUIRED_FLAGS:
                problems.append("task_flags")
            counts = metadata.get("images_by_task", {})
            for task in REQUIRED_TASKS:
                if int(counts.get(task, 0)) != args.expected_images:
                    problems.append("%s_coverage" % task)
        row = {
            "manifest": str(path), "name": payload.get("name"),
            "family": payload.get("architecture_family"), "problems": problems,
        }
        rows.append(row)
        if problems:
            failures.append(row)
    valid_families = {row["family"] for row in rows if not row["problems"]}
    report = {
        "status": "ready" if len(valid_families) >= args.minimum_families else "not_ready",
        "required_tasks": sorted(REQUIRED_TASKS),
        "minimum_families": args.minimum_families,
        "valid_families": sorted(valid_families),
        "rows": rows,
        "failures": failures,
    }
    report_path = root / "artifacts/manifests/vg_tritask_readiness.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("report=" + str(report_path))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
