#!/usr/bin/env python3
"""Validate and summarize a completed six-backbone Experiment-I run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


EXPECTED_BACKBONES = {
    "resnet50", "dinov2_b", "siglip2_b", "radio_v25_b",
    "cradio_v4_so400m", "sam_vit_b",
}
EXPECTED_DEPTHS = (0, 2, 4, 8)
EXPECTED_SEEDS = (17, 23, 31)
RECALL_KS = (1, 5, 10, 20, 50, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_std(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--report")
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve() if args.report
        else root / "result_audit.json"
    )
    errors, warnings, rows, checkpoints = [], [], [], []
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    backbones = {path.name for path in directories}
    if backbones != EXPECTED_BACKBONES:
        errors.append({
            "kind": "backbone_panel",
            "missing": sorted(EXPECTED_BACKBONES - backbones),
            "unexpected": sorted(backbones - EXPECTED_BACKBONES),
        })

    expected_grid = {
        (depth, seed) for depth in EXPECTED_DEPTHS for seed in EXPECTED_SEEDS
    }
    for directory in directories:
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            errors.append({"kind": "missing_summary", "path": str(summary_path)})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = summary.get("runs", [])
        actual_grid = {(run.get("depth"), run.get("seed")) for run in runs}
        if actual_grid != expected_grid or len(runs) != len(expected_grid):
            errors.append({
                "kind": "run_grid", "backbone": directory.name,
                "missing": sorted(expected_grid - actual_grid),
                "unexpected": sorted(actual_grid - expected_grid),
                "run_count": len(runs),
            })
        for run in runs:
            run_name = str(run["run_name"])
            run_json = directory / f"{run_name}.json"
            checkpoint = directory / f"{run_name}.pth"
            if not run_json.is_file() or not checkpoint.is_file():
                errors.append({"kind": "missing_run_asset", "run": run_name})
                continue
            if json.loads(run_json.read_text(encoding="utf-8")) != run:
                errors.append({"kind": "run_summary_mismatch", "run": run_name})
            history = run.get("history", [])
            if len(history) != 10 or any(
                not math.isfinite(float(epoch["cross_entropy"])) for epoch in history
            ):
                errors.append({"kind": "invalid_training_history", "run": run_name})
            evaluation = run["evaluation"]
            metrics = evaluation["metrics"]
            for prefix in ("R", "mR", "zR"):
                values = [float(metrics[f"{prefix}@{k}"]) for k in RECALL_KS]
                if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                       for value in values):
                    errors.append({"kind": "metric_range", "run": run_name,
                                   "metric": prefix})
                if any(after < before for before, after in zip(values, values[1:])):
                    errors.append({"kind": "metric_monotonicity", "run": run_name,
                                   "metric": prefix})
            digest = _sha256(checkpoint)
            if run.get("checkpoint_sha256") and run["checkpoint_sha256"] != digest:
                errors.append({"kind": "checkpoint_digest", "run": run_name})
            if not args.skip_checkpoint_load:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (saved.get("seed"), saved.get("depth")) != (
                    run.get("seed"), run.get("depth")
                ) or not saved.get("state_dict"):
                    errors.append({"kind": "checkpoint_payload", "run": run_name})
            checkpoint_reference = Path(str(run.get("checkpoint", "")))
            if checkpoint_reference.is_absolute() and not checkpoint_reference.is_file():
                warnings.append({
                    "kind": "stale_absolute_checkpoint_path", "run": run_name,
                    "recorded": str(checkpoint_reference),
                    "local": str(checkpoint),
                })
            diagnostics = evaluation["layer_diagnostics"]
            final = diagnostics[str(max(map(int, diagnostics)))]
            physical = evaluation["physical_consistency"]
            rows.append({
                "backbone": directory.name,
                "depth": int(run["depth"]),
                "seed": int(run["seed"]),
                "R@50": float(metrics["R@50"]),
                "mR@50": float(metrics["mR@50"]),
                "zR@50": float(metrics["zR@50"]),
                "effective_rank": float(final["effective_rank"]),
                "dirichlet_energy": float(final["dirichlet_energy"]),
                "final_cross_entropy": float(history[-1]["cross_entropy"]),
                "pvr_status": physical["pvr_status"],
                "pvr_coverage": float(physical["coverage"]),
            })
            checkpoints.append({
                "run": run_name, "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size, "sha256": digest,
            })

    depth_summary = {}
    for depth in EXPECTED_DEPTHS:
        selected = [row for row in rows if row["depth"] == depth]
        depth_summary[str(depth)] = {
            key: _mean_std(selected, key) for key in (
                "R@50", "mR@50", "zR@50", "effective_rank",
                "dirichlet_energy", "final_cross_entropy",
            )
        }
    backbone_summary = {}
    for backbone in sorted(backbones):
        candidates = {}
        for depth in EXPECTED_DEPTHS:
            selected = [
                row for row in rows
                if row["backbone"] == backbone and row["depth"] == depth
            ]
            candidates[str(depth)] = {
                key: _mean_std(selected, key)
                for key in ("R@50", "mR@50", "zR@50")
            }
        best = max(candidates, key=lambda value: candidates[value]["R@50"]["mean"])
        backbone_summary[backbone] = {
            "by_depth": candidates, "best_mean_R@50_depth": int(best),
        }

    statuses = Counter(row["pvr_status"] for row in rows)
    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "run_root": str(root),
        "backbones": sorted(backbones),
        "run_count": len(rows),
        "checkpoint_count": len(checkpoints),
        "errors": errors,
        "warnings": warnings,
        "pvr_status_counts": dict(statuses),
        "pvr_coverage_range": [
            min((row["pvr_coverage"] for row in rows), default=None),
            max((row["pvr_coverage"] for row in rows), default=None),
        ],
        "depth_summary": depth_summary,
        "backbone_summary": backbone_summary,
        "checkpoints": checkpoints,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{report['status'].upper()}] backbones={len(backbones)} "
        f"runs={len(rows)} checkpoints={len(checkpoints)} "
        f"errors={len(errors)} warnings={len(warnings)}"
    )
    print(f"pvr_status={dict(statuses)} coverage={report['pvr_coverage_range']}")
    print(f"report={report_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
