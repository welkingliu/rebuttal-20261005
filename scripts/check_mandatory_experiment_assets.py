#!/usr/bin/env python3
"""Stage-aware preflight for the converged IV -> II -> V submission chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TRITASK_MODELS = ("motifs", "transformer")
LIVE_MODELS = TRITASK_MODELS
TASKS = ("predcls", "sgcls", "sgdet")


def _first(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def _check(path: Path, kind="file") -> dict:
    exists = path.is_file() if kind == "file" else path.is_dir()
    return {"path": str(path), "ready": bool(exists)}


def _check_registered_checkpoint(root: Path, model: str, task: str) -> dict:
    output = root / "checkpoints/sgg/trained/pysgg" / model / task
    pointer = output / "last_checkpoint"
    marker = output / ".sgg_training_complete.json"
    checkpoint = None
    if pointer.is_file():
        try:
            checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
        except OSError:
            checkpoint = None
    ready = bool(
        checkpoint
        and checkpoint.is_file()
        and marker.is_file()
    )
    return {
        "path": str(checkpoint or pointer),
        "pointer": str(pointer),
        "completion_marker": str(marker),
        "ready": ready,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--require", nargs="+",
        choices=("base", "iva", "tritask_checkpoints", "tritask_caches",
                 "external", "live_manifests", "all"),
        default=["base"],
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    manifests = root / "checkpoints/sgg/manifests"
    vg = _first(root / "data/vg/v1.4", root / "data/vg")
    source = root / "external/official_repos/PySGG"

    stages = {}
    stages["base"] = {
        "assets": {
            "vg_h5": _check(_first(vg / "VG-SGG.h5", vg / "VG-SGG-with-attri.h5")),
            "vg_dict": _check(_first(vg / "VG-SGG-dicts.json", vg / "VG-SGG-dicts-with-attri.json")),
            "seen_triplets": _check(root / "artifacts/manifests/seen_triplets_full.json"),
            "pysgg_source": _check(source, "dir"),
            "pysgg_source_marker": _check(source / ".official_source.json"),
            "pysgg_extension": {
                "path": str(source / "pysgg"),
                "ready": bool(list((source / "pysgg").glob("_C*.so"))),
            },
            "pysgg_worker_python": _check(Path(
                os.environ.get(
                    "PYSGG_PYTHON",
                    "python3",
                )
            )),
            "pysgg_worker_script": _check(root / "scripts/pysgg_live_worker.py"),
            "shared_detector": _check(root / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"),
            "glove": _check(root / "data/derived/glove/glove.6B.200d.txt"),
            "training_plan": _check(root / "artifacts/manifests/pysgg_vg_tritask_training_plan.json"),
        },
    }
    configs = [
        root / "configs/pysgg_vg_tritask" / f"{model}_{task}.yaml"
        for model in TRITASK_MODELS for task in TASKS
    ]
    stages["base"]["assets"]["tritask_configs"] = {
        "path": str(root / "configs/pysgg_vg_tritask"),
        "ready": all(path.is_file() for path in configs),
        "present": sum(path.is_file() for path in configs),
        "required": len(configs),
    }
    stages["iva"] = {
        "assets": {
            "bgnn_full_manifest": _check(manifests / "pysgg_bgnn_vg_sgdet.json"),
        },
    }

    checkpoint_assets = {}
    for model in TRITASK_MODELS:
        for task in TASKS:
            checkpoint_assets[f"{model}/{task}"] = _check_registered_checkpoint(
                root, model, task
            )
    stages["tritask_checkpoints"] = {"assets": checkpoint_assets}

    cache_assets = {}
    for model in TRITASK_MODELS:
        cache = root / "artifacts/prediction_cache" / f"pysgg_{model}_vg_tritask"
        metadata_path = cache / "metadata.json"
        ready = False
        detail = {"path": str(metadata_path), "ready": False}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
                counts = metadata.get("images_by_task", {})
                ready = (
                    set(metadata.get("tasks", [])) == set(TASKS)
                    and all(int(counts.get(task, 0)) == 26446 for task in TASKS)
                )
                detail["images_by_task"] = counts
            except (OSError, ValueError, TypeError):
                ready = False
        detail["ready"] = ready
        cache_assets[model] = detail
    stages["tritask_caches"] = {"assets": cache_assets}

    stages["external"] = {"assets": {
        "oi_egtr": _check(manifests / "egtr_oi.json"),
        "oi_sgtr": _check(manifests / "sgtr_oi.json"),
        "psg_motifs": _check(manifests / "openpsg_motifs_psg.json"),
        "psg_vctree": _check(manifests / "openpsg_vctree_psg.json"),
    }}
    stages["live_manifests"] = {"assets": {
        model: _check(manifests / f"pysgg_{model}_vg_live.json")
        for model in LIVE_MODELS
    }}

    for value in stages.values():
        value["ready"] = all(
            bool(asset["ready"]) for asset in value["assets"].values()
        )
    required = set(stages) if "all" in args.require else set(args.require)
    failures = [stage for stage in sorted(required) if not stages[stage]["ready"]]
    report = {
        "schema": "mandatory_experiment_assets_v1",
        "status": "ready" if not failures else "not_ready",
        "required_stages": sorted(required),
        "failed_stages": failures,
        "stages": stages,
        "next_stage": next(
            (name for name in (
                "base", "iva", "tritask_checkpoints", "tritask_caches",
                "external", "live_manifests",
            ) if not stages[name]["ready"]),
            "complete",
        ),
    }
    report_path = Path(
        args.report or root / "artifacts/manifests/mandatory_experiment_assets.json"
    ).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for stage, value in stages.items():
        print(f"[{'ok' if value['ready'] else 'wait'}] {stage}")
    print("next_stage=" + report["next_stage"])
    print("report=" + str(report_path))
    if failures:
        raise SystemExit("[NOT READY] " + ", ".join(failures))
    print("[READY] " + ", ".join(sorted(required)))


if __name__ == "__main__":
    main()
