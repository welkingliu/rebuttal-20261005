#!/usr/bin/env python3
"""Validate the separate I-A and Experiment V external-result contracts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.submission_protocol import (
    EXTERNAL_DIAGNOSTIC_DATASETS,
    EXTERNAL_INFERENCE_FAMILY_TARGETS,
    EXTERNAL_INFERENCE_MODES,
    MITIGATION_SEEDS,
)


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--require_mitigation", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    artifacts = root / "artifacts"
    report_path = Path(args.report).expanduser().resolve() if args.report else (
        artifacts / "manifests" / "external_result_readiness.json"
    )

    minimum_ia_images = {"gqa": 1000, "vrd": 955}
    ia = defaultdict(list)
    for path in (artifacts / "experiment_1a").glob("**/summary.json"):
        payload = _json(path)
        if not payload or payload.get("schema_version") != "experiment_1a_external_box_zero_shot_v1":
            continue
        dataset = str(payload.get("dataset", "")).lower()
        processed_images = int(payload.get("processed_images") or 0)
        if (
            dataset in EXTERNAL_DIAGNOSTIC_DATASETS
            and processed_images >= minimum_ia_images[dataset]
        ):
            ia[dataset].append({
                "path": str(path.resolve()),
                "images": processed_images,
                "objects": payload.get("processed_objects"),
            })

    mitigation = defaultdict(lambda: defaultdict(set))
    mitigation_paths = defaultdict(list)
    for path in (artifacts / "experiment_5").glob("**/summary.json"):
        payload = _json(path)
        if not payload or payload.get("schema_version") != "experiment_5_external_shared_vg_v1":
            continue
        metadata = payload.get("cache_metadata", {})
        state = payload.get("mitigation_state") or {}
        dataset = str(metadata.get("dataset", "")).lower()
        family = str(metadata.get("architecture_family", "")).strip()
        mode = state.get("training_mode")
        seed = state.get("seed")
        if (
            dataset in EXTERNAL_DIAGNOSTIC_DATASETS and family
            and mode in EXTERNAL_INFERENCE_MODES and seed in MITIGATION_SEEDS
        ):
            mitigation[dataset][family].add((mode, int(seed)))
            mitigation_paths[dataset].append(str(path.resolve()))

    failures = []
    for dataset in EXTERNAL_DIAGNOSTIC_DATASETS:
        if not ia[dataset]:
            failures.append(f"experiment_1a_external_{dataset}=0/1")
        if args.require_mitigation:
            complete_families = [
                family for family, runs in mitigation[dataset].items()
                if runs == {
                    (mode, seed)
                    for mode in EXTERNAL_INFERENCE_MODES
                    for seed in MITIGATION_SEEDS
                }
            ]
            target = EXTERNAL_INFERENCE_FAMILY_TARGETS[dataset]
            if len(complete_families) < target:
                failures.append(
                    f"experiment_5_external_{dataset}="
                    f"{len(complete_families)}/{target}_complete_families"
                )

    report = {
        "status": "ready" if not failures else "not_ready",
        "experiment_1a_external": dict(ia),
        "experiment_5_external_runs": {
            dataset: {
                family: sorted([list(item) for item in runs])
                for family, runs in families.items()
            }
            for dataset, families in mitigation.items()
        },
        "experiment_5_external_paths": dict(mitigation_paths),
        "mitigation_required": args.require_mitigation,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("External dataset result contract")
    print("=" * 72)
    for dataset in EXTERNAL_DIAGNOSTIC_DATASETS:
        print(
            f"  {dataset}: I-A={len(ia[dataset])} "
            f"V_families={len(mitigation[dataset])}"
        )
    print(f"  report={report_path}")
    if failures:
        print("[NOT READY] " + "; ".join(failures))
        raise SystemExit(1)
    print("[READY] Requested external result contract is satisfied.")


if __name__ == "__main__":
    main()
