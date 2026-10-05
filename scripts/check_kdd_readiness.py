#!/usr/bin/env python3
"""Report asset and integration readiness for the KDD Datasets & Benchmarks run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_official_integration import SUBMISSION_ASSETS
from official_model_catalog import MODELS, REPOSITORIES
from prepare_foundation_models import MAIN_MODELS, status_for
from sgg_core.submission_protocol import (
    DIAGNOSTIC_MODEL_RANGE,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_DATASET_FAMILY_TARGETS,
    STANDARD_TASK_FAMILY_TARGETS,
)


def _first(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_image_id(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _manifest_records(root: Path) -> tuple[list[dict], list[str]]:
    directory = root / "checkpoints" / "sgg" / "manifests"
    records, errors = [], []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = {
                "path": str(path),
                "name": str(payload["name"]),
                "family": str(payload["architecture_family"]),
                "datasets": sorted({
                    str(value).lower()
                    for value in payload.get("supported_datasets", [])
                }),
                "tasks": sorted({
                    str(value).lower()
                    for value in payload.get("supported_tasks", [])
                }),
                "execution_mode": str(
                    payload.get("execution_mode", "live_adapter")
                ),
                "perturbation_contract": payload.get(
                    "perturbation_contract", {}
                ),
                "diagnostic_contract": payload.get("diagnostic_contract", {}),
                "mitigation_contract": payload.get("mitigation_contract", {}),
            }
            if not record["datasets"] or not record["tasks"]:
                raise ValueError("empty dataset/task support")
            records.append(record)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return records, errors


def _families(records, dataset=None, task=None, mode=None):
    return {
        record["family"] for record in records
        if (dataset is None or dataset in record["datasets"])
        and (task is None or task in record["tasks"])
        and (mode is None or mode == record["execution_mode"])
    }


def _mask_status(path: Path, annotation: Path, verify_hashes=False) -> dict:
    manifest = path / "manifest.json"
    if not annotation.is_file():
        return {
            "ok": False, "path": str(manifest),
            "reason": f"annotation missing: {annotation}",
        }
    if not manifest.is_file():
        return {"ok": False, "path": str(manifest), "reason": "missing"}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        observed = Path(payload.get("annotation", "")).expanduser().resolve()
        hash_matches = None
        if verify_hashes:
            mask_files = sorted(
                path / "masks" / f"{_safe_image_id(row['image_id'])}.npz"
                for row in payload.get("records", [])
                if row.get("status") in {"ok", "cached"}
            )
            digest = hashlib.sha256()
            for mask_file in mask_files:
                if not mask_file.is_file():
                    hash_matches = False
                    break
                digest.update(mask_file.name.encode("utf-8"))
                digest.update(_sha256(mask_file).encode("ascii"))
            else:
                hash_matches = digest.hexdigest() == payload.get(
                    "mask_cache_sha256"
                )
        ok = (
            payload.get("schema") == "psg_sam_gt_box_prompt_v1"
            and observed == annotation.resolve()
            and payload.get("annotation_sha256") == _sha256(annotation)
            and payload.get("images_ready") == payload.get("images_total")
            and int(payload.get("images_total", 0)) > 0
            and hash_matches is not False
        )
        return {
            "ok": ok,
            "path": str(manifest),
            "images_ready": payload.get("images_ready"),
            "images_total": payload.get("images_total"),
            "annotation_matches": observed == annotation.resolve(),
            "mask_cache_hash_matches": hash_matches,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(manifest), "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--sam_mask_root")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify_mask_hashes", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    data = root / "data"
    sam_root = (
        Path(args.sam_mask_root).expanduser().resolve()
        if args.sam_mask_root else data / "derived" / "sam_psg"
    )

    data_paths = {
        "vg_h5": _first(
            data / "vg" / "v1.4" / "VG-SGG.h5",
            data / "vg" / "v1.4" / "VG-SGG-with-attri.h5",
        ),
        "vg_dict": _first(
            data / "vg" / "v1.4" / "VG-SGG-dicts.json",
            data / "vg" / "v1.4" / "VG-SGG-dicts-with-attri.json",
        ),
        "oi_train": data / "openimages" / "open-images-v6" / "annotations" / "oidv6-train-annotations-vrd.csv",
        "oi_validation": data / "openimages" / "open-images-v6" / "annotations" / "oidv6-validation-annotations-vrd.csv",
        "gqa_train": _first(
            data / "gqa" / "train_sceneGraphs.json",
            data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
        ),
        "gqa_validation": _first(
            data / "gqa" / "val_sceneGraphs.json",
            data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
        ),
        "psg_train": data / "psg" / "psg_train_val.json",
        "psg_eval": _first(
            data / "psg" / "psg_val_test.json", data / "psg" / "psg.json"
        ),
        "vrd_train": data / "vrd" / "json_dataset" / "annotations_train.json",
        "vrd_test": data / "vrd" / "json_dataset" / "annotations_test.json",
    }
    data_status = {
        key: {"ok": path.is_file(), "path": str(path)}
        for key, path in data_paths.items()
    }

    foundation = [status_for(model, root) for model in MAIN_MODELS]
    foundation_ok = all(item["ok"] for item in foundation)

    weight_rows = []
    for name in SUBMISSION_ASSETS:
        spec = MODELS[name]
        base = root / "checkpoints" / "sgg" / "weights"
        archive = base / spec["relative_path"]
        runtime = base / spec.get("runtime_checkpoint", spec["relative_path"])
        companions = [base / value for value in spec.get("required_paths", [])]
        if spec.get("runtime_config"):
            companions.append(base / spec["runtime_config"])
        ok = (
            archive.is_file() and archive.stat().st_size >= 1024 * 1024
            and runtime.is_file() and runtime.stat().st_size >= 1024 * 1024
            and all(path.is_file() for path in companions)
        )
        weight_rows.append({
            "name": name, "family": spec["architecture"], "ok": ok,
            "asset": str(archive), "runtime": str(runtime),
            "companions": [str(path) for path in companions],
        })

    repository_rows = []
    for key in sorted({MODELS[name]["repository"] for name in SUBMISSION_ASSETS}):
        path = root / "external" / "official_repos" / REPOSITORIES[key]["directory"]
        ok = path.is_dir() and (
            (path / ".git").is_dir() or (path / ".official_source.json").is_file()
        )
        repository_rows.append({"name": key, "ok": ok, "path": str(path)})

    manifests, manifest_errors = _manifest_records(root)
    family_counts = {
        dataset: len(_families(manifests, dataset=dataset))
        for dataset in STANDARD_DATASET_FAMILY_TARGETS
    }
    task_counts = {
        dataset: {
            task: len(_families(manifests, dataset=dataset, task=task))
            for task in targets
        }
        for dataset, targets in STANDARD_TASK_FAMILY_TARGETS.items()
    }
    live_vg = [
        record for record in manifests
        if "vg" in record["datasets"]
        and record["execution_mode"] == "live_adapter"
    ]
    diagnostic_live = {
        record["family"] for record in live_vg
        if record["diagnostic_contract"].get("gt_pair_predict") is True
        if all(record["perturbation_contract"].get(key) is True for key in (
            "full", "visual_noise", "union_attenuation",
            "on_manifold_replacement", "random_node_mask", "key_node_mask",
            "unrelated_node_mask",
        ))
    }
    graph_live = {
        record["family"] for record in live_vg
        if record["diagnostic_contract"].get("gt_node_features") is True
        and record["diagnostic_contract"].get("graph_intervention") is True
    }
    mitigation_live = {
        record["family"] for record in live_vg
        if all(record["mitigation_contract"].get(key) is True for key in (
            "forward_grounding", "trainable_grounding_parameters",
            "object_logits", "trainable_object_parameters",
        ))
        and record["mitigation_contract"].get(
            "relation_logit_alignment"
        ) == "gt_relations"
        and record["mitigation_contract"].get(
            "object_logit_alignment"
        ) == "gt_entities"
    }

    sam = {
        "train": _mask_status(
            sam_root / "train", data_paths["psg_train"],
            args.verify_mask_hashes,
        ),
        "eval": _mask_status(
            sam_root / "eval", data_paths["psg_eval"],
            args.verify_mask_hashes,
        ),
    }
    seen_path = root / "artifacts" / "manifests" / "seen_triplets_full.json"
    seen_datasets = []
    if seen_path.is_file():
        try:
            seen_payload = json.loads(seen_path.read_text(encoding="utf-8"))
            seen_datasets = sorted(set(seen_payload) - {"_metadata"})
        except (OSError, json.JSONDecodeError):
            pass

    standard_ready = (
        len(_families(manifests)) >= GLOBAL_MODEL_FAMILY_TARGET
        and all(
            family_counts[dataset] >= target
            for dataset, target in STANDARD_DATASET_FAMILY_TARGETS.items()
        )
        and all(
            task_counts[dataset][task] >= target
            for dataset, targets in STANDARD_TASK_FAMILY_TARGETS.items()
            for task, target in targets.items()
        )
        and all(dataset in seen_datasets for dataset in STANDARD_DATASET_FAMILY_TARGETS)
    )
    lower, _ = DIAGNOSTIC_MODEL_RANGE
    stages = {
        "experiment_1a": {
            "ready": foundation_ok and all(item["ok"] for item in sam.values())
            and data_status["psg_train"]["ok"] and data_status["psg_eval"]["ok"],
            "requires": "six foundation backbones, PSG RGB/panoptic data, train/eval SAM masks",
        },
        "experiment_1b": {
            "ready": foundation_ok and data_status["vg_h5"]["ok"]
            and data_status["vg_dict"]["ok"],
            "requires": "six foundation backbones and VG-150",
        },
        "experiment_2": {
            "ready": len(diagnostic_live) >= lower,
            "requires": "two VG live adapters with GT-pair-aligned perturbation support",
        },
        "experiment_3_appendix": {
            "ready": len(graph_live) >= lower,
            "requires": "optional: two VG live adapters with GT-node graph interventions",
        },
        "experiment_4": {
            "ready": standard_ready,
            "requires": "five-family SGDet breadth, two-family VG tri-task depth, full seen-triplet manifest",
        },
        "experiment_5": {
            "ready": len(mitigation_live) >= 2,
            "requires": "two live trainable families with GT-aligned object/relation logits",
        },
    }
    report = {
        "project_root": str(root),
        "data": data_status,
        "foundation": foundation,
        "official_weights": weight_rows,
        "official_repositories": repository_rows,
        "manifest_errors": manifest_errors,
        "manifests": manifests,
        "standard_family_counts": family_counts,
        "standard_task_counts": task_counts,
        "live_diagnostic_families": sorted(diagnostic_live),
        "live_graph_intervention_families": sorted(graph_live),
        "live_mitigation_families": sorted(mitigation_live),
        "sam_masks": sam,
        "seen_triplets": {"path": str(seen_path), "datasets": seen_datasets},
        "stages": stages,
        "all_ready": all(value["ready"] for value in stages.values()),
    }
    report_path = (
        Path(args.report).expanduser().resolve() if args.report else
        root / "artifacts" / "manifests" / "kdd_readiness.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("KDD Datasets & Benchmarks readiness")
    print("=" * 72)
    print(f"data entry points: {sum(row['ok'] for row in data_status.values())}/{len(data_status)}")
    print(f"foundation backbones: {sum(row['ok'] for row in foundation)}/{len(foundation)}")
    print(f"official runtime weights: {sum(row['ok'] for row in weight_rows)}/{len(weight_rows)}")
    print(f"official sources: {sum(row['ok'] for row in repository_rows)}/{len(repository_rows)}")
    print(f"manifests: {len(manifests)} families={len(_families(manifests))}")
    print(f"standard dataset families: {family_counts}")
    print(f"standard task families: {task_counts}")
    print(f"live diagnostic families: {sorted(diagnostic_live)}")
    print(f"live mitigation families: {sorted(mitigation_live)}")
    print("\nStage gates")
    for name, value in stages.items():
        print(f"  [{'ready' if value['ready'] else 'blocked'}] {name}: {value['requires']}")
    print(f"\nreport={report_path}")
    if args.strict and not report["all_ready"]:
        print("[NOT READY] Resolve blocked stage gates before the formal launcher.")
        return 1
    print("[READY]" if report["all_ready"] else "[PARTIAL] Use the stage gates above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
