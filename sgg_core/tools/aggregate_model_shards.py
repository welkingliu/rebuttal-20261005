"""Aggregate isolated Experiment-IV model shards into one formal result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sgg_core.audits.evidence_chain import EvidenceChainAnalyzer
from sgg_core.experiments.experiment_4 import _cross_dataset_rows
from sgg_core.protocol import write_json
from sgg_core.submission_protocol import (
    EXTERNAL_DATASET_MODEL_TARGETS,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_BENCHMARK_DATASETS,
    STANDARD_DATASET_FAMILY_TARGETS,
    STANDARD_TASK_FAMILY_TARGETS,
    parse_dataset_targets,
)


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_native_reference(provenance: dict) -> dict | None:
    """Read a fresh, hash-pinned native report from the recorded manifest."""
    manifest_path = Path(str(provenance.get("manifest_path", "")))
    if not manifest_path.is_file():
        return None
    manifest = _read(manifest_path)
    declaration = manifest.get("native_reference_validation")
    if not isinstance(declaration, dict) or declaration.get("status") != "pass":
        return None
    report_path = Path(str(declaration.get("report", "")))
    if not report_path.is_file() or _sha256(report_path) != declaration.get("sha256"):
        return None
    report = _read(report_path)
    checkpoint_sha = provenance.get("checkpoint_status", {}).get("sha256")
    if (
        report.get("status") != "pass"
        or report.get("checkpoint_sha256") != checkpoint_sha
        or int(report.get("eval_images", 0)) <= 0
    ):
        return None
    return {
        "status": "pass_external_native_protocol",
        "protocol": declaration.get("protocol"),
        "report": str(report_path),
        "sha256": declaration.get("sha256"),
    }


def _merge_mapping(target: dict, source: dict, label: str) -> None:
    for name, value in source.items():
        if name in target:
            old_sha = (
                target[name].get("checkpoint_status", {}).get("sha256")
                if isinstance(target[name], dict) else None
            )
            new_sha = (
                value.get("checkpoint_status", {}).get("sha256")
                if isinstance(value, dict) else None
            )
            if old_sha and new_sha and old_sha != new_sha:
                raise RuntimeError(f"Checkpoint collision in {label}/{name}: {old_sha} != {new_sha}")
            raise RuntimeError(f"Duplicate model result in {label}: {name}")
        target[name] = value


def _merge_provenance(target: dict, source: dict, label: str) -> None:
    for name, value in source.items():
        if name not in target:
            target[name] = value
            continue
        old_sha = target[name].get("checkpoint_status", {}).get("sha256")
        new_sha = value.get("checkpoint_status", {}).get("sha256")
        if old_sha != new_sha:
            raise RuntimeError(
                f"Checkpoint collision in {label}/{name}: {old_sha} != {new_sha}"
            )


def _load_diagnostics(roots: list[Path]) -> tuple[dict, dict, list[str]]:
    merged: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    sources = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Diagnostic root not found: {root}")
        paths = sorted(set(root.rglob("experiment_2.json")))
        paths.extend(sorted(set(root.rglob("experiment_3.json"))))
        for path in paths:
            payload = _read(path)
            dataset = str(payload.get("dataset", "")).lower()
            if not dataset:
                raise RuntimeError(f"Diagnostic result has no dataset: {path}")
            target = merged.setdefault(dataset, {})
            mappings = {}
            if path.name == "experiment_2.json":
                mappings = {
                    "pair_audit": payload.get("pair_audit") or {},
                    "perturbation_sweep": (
                        payload.get("dose_response_and_controls") or {}
                    ),
                    "physical_consistency": (
                        payload.get("physical_consistency") or {}
                    ),
                    "object_error_propagation": payload.get(
                        "object_error_propagation", {}
                    ) or {},
                }
            elif path.name == "experiment_3.json":
                mappings = {"graph_audit": payload.get("graph_audit", {})}
            for audit_name, values in mappings.items():
                if not isinstance(values, dict):
                    raise TypeError(f"{path}: {audit_name} must be a mapping")
                target.setdefault(audit_name, {})
                _merge_mapping(target[audit_name], values, f"{dataset}/{audit_name}")
            model_info = payload.get("model_provenance", {})
            if isinstance(model_info, dict):
                target.setdefault("model_provenance", {})
                _merge_provenance(
                    target["model_provenance"], model_info,
                    f"{dataset}/model_provenance",
                )
                _merge_provenance(provenance, model_info, "diagnostic_provenance")
            sources.append(str(path))
    return merged, provenance, sources


def aggregate(shard_root: Path, datasets: list[str], minimum_families: int,
              minimum_per_dataset: int | dict[str, int],
              diagnostic_roots: list[Path] | None = None,
              diagnostic_targets: dict[str, int] | None = None,
              task_contract: str = "full") -> dict:
    summaries = sorted(shard_root.glob("*/*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No model shards found under {shard_root}")

    merged = {dataset: {} for dataset in datasets}
    provenance = {}
    reproduction_by_model = {}
    sources = []
    for summary_path in summaries:
        shard_summary = _read(summary_path)
        shard_datasets = shard_summary.get("datasets", [])
        if len(shard_datasets) != 1:
            raise RuntimeError(f"Shard must contain exactly one dataset: {summary_path}")
        dataset = shard_datasets[0]
        if dataset not in merged:
            continue
        result_path = summary_path.parent / dataset / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = _read(result_path)
        for audit_name, payload in result.items():
            if audit_name == "dataset_metadata":
                continue
            if not isinstance(payload, dict):
                continue
            merged[dataset].setdefault(audit_name, {})
            _merge_mapping(merged[dataset][audit_name], payload, f"{dataset}/{audit_name}")
        for name, value in shard_summary.get("model_provenance", {}).items():
            if name in provenance:
                old_sha = provenance[name].get("checkpoint_status", {}).get("sha256")
                new_sha = value.get("checkpoint_status", {}).get("sha256")
                if old_sha != new_sha:
                    raise RuntimeError(f"Model {name} changed checkpoint across datasets")
            else:
                provenance[name] = value
        for name, value in result.get("reproduction_validation", {}).items():
            reproduction_by_model.setdefault(name, []).append(value.get("status"))
        sources.append(str(summary_path))

    families = {
        value.get("architecture_family") for value in provenance.values()
        if value.get("architecture_family")
    }
    if len(families) < minimum_families:
        raise RuntimeError(
            f"Formal aggregate has {len(families)}/{minimum_families} model families"
        )
    dataset_family_counts = {}
    task_family_counts = {}
    dataset_targets = (
        {dataset: int(minimum_per_dataset) for dataset in datasets}
        if isinstance(minimum_per_dataset, int)
        else {dataset: int(minimum_per_dataset.get(dataset, 0)) for dataset in datasets}
    )
    if task_contract == "sgdet_only":
        task_targets = {
            dataset: {"sgdet": int(dataset_targets.get(dataset, 0))}
            for dataset in datasets
        }
    elif task_contract == "tritask_depth":
        task_targets = {
            dataset: {
                task: int(dataset_targets.get(dataset, 0))
                for task in ("predcls", "sgcls", "sgdet")
            }
            for dataset in datasets
        }
    else:
        task_targets = {
            dataset: dict(STANDARD_TASK_FAMILY_TARGETS.get(dataset, {}))
            for dataset in datasets
        }
    for dataset, result in merged.items():
        dataset_provenance = result.get("model_provenance", {})
        dataset_families = {
            value.get("architecture_family") for value in dataset_provenance.values()
            if value.get("architecture_family")
        }
        dataset_family_counts[dataset] = len(dataset_families)
        minimum = dataset_targets[dataset]
        if len(dataset_families) < minimum:
            raise RuntimeError(
                f"Dataset {dataset} has {len(dataset_families)}/{minimum} families"
            )
        task_family_counts[dataset] = {}
        standard_results = result.get("standard_sgg", {})
        for task, target in task_targets.get(dataset, {}).items():
            task_families = {
                dataset_provenance[name].get("architecture_family")
                for name, value in standard_results.items()
                if name in dataset_provenance
                and value.get("tasks", {}).get(task, {}).get("status") == "ok"
            }
            task_families.discard(None)
            task_family_counts[dataset][task] = len(task_families)
            if (
                task_contract == "tritask_depth"
                or minimum_families >= GLOBAL_MODEL_FAMILY_TARGET
            ) and len(task_families) < target:
                raise RuntimeError(
                    f"Dataset/task {dataset}/{task} has "
                    f"{len(task_families)}/{target} successful families"
                )
    missing_reproduction = [
        name for name in provenance
        if not {
            "pass", "pass_with_protocol_qualification",
        }.intersection(reproduction_by_model.get(name, []))
        and _validated_native_reference(provenance[name]) is None
    ]
    if missing_reproduction:
        raise RuntimeError(
            "Every checkpoint must reproduce a native reference metric once; missing="
            f"{missing_reproduction}"
        )

    diagnostic_results, diagnostic_provenance, diagnostic_sources = _load_diagnostics(
        diagnostic_roots or []
    )
    diagnostic_family_counts = {}
    for dataset, results in diagnostic_results.items():
        families_for_dataset = {
            value.get("architecture_family")
            for value in results.get("model_provenance", {}).values()
            if value.get("architecture_family")
        }
        diagnostic_family_counts[dataset] = len(families_for_dataset)
    for dataset, target in (diagnostic_targets or {}).items():
        available = diagnostic_family_counts.get(dataset, 0)
        if available < int(target):
            raise RuntimeError(
                f"External diagnostic dataset {dataset} has "
                f"{available}/{target} ontology-compatible families"
            )

    analysis_results = {dataset: dict(result) for dataset, result in merged.items()}
    for dataset, diagnostics in diagnostic_results.items():
        target = analysis_results.setdefault(dataset, {})
        for audit_name, values in diagnostics.items():
            if audit_name == "model_provenance":
                target.setdefault(audit_name, {})
                _merge_provenance(
                    target[audit_name], values, f"{dataset}/joined_provenance"
                )
            else:
                target.setdefault(audit_name, {})
                _merge_mapping(target[audit_name], values, f"{dataset}/{audit_name}")

    analysis_provenance = dict(provenance)
    _merge_provenance(
        analysis_provenance, diagnostic_provenance, "joined_model_provenance"
    )
    load_report = {
        name: {
            **value,
            "parameter_count": value.get("checkpoint_status", {}).get("parameter_count"),
            "paradigm": value.get("checkpoint_status", {}).get("metadata", {}).get("paradigm"),
        }
        for name, value in analysis_provenance.items()
    }
    evidence = EvidenceChainAnalyzer(performance_k=50, diagnostic_k=5).analyze(
        analysis_results, load_report
    )
    return {
        "experiment": "IV_standard_isolated_official_model_benchmark",
        "status": "formal_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "model_runs": len(provenance),
        "model_families": sorted(families),
        "model_family_count": len(families),
        "dataset_family_counts": dataset_family_counts,
        "dataset_family_targets": dataset_targets,
        "task_contract": task_contract,
        "task_family_counts": task_family_counts,
        "task_family_targets": task_targets,
        "diagnostic_reuse_contract": {
            "pair_and_perturbation": "Experiment II",
            "motif_intervention": "Experiment III",
            "rerun_inside_experiment_4": False,
        },
        "cross_dataset_rows": _cross_dataset_rows(analysis_results),
        "evidence_chain": evidence,
        "model_provenance": provenance,
        "external_native_reference_validation": {
            name: value for name in provenance
            if (value := _validated_native_reference(provenance[name])) is not None
        },
        "diagnostic_model_provenance": diagnostic_provenance,
        "diagnostic_family_counts": diagnostic_family_counts,
        "diagnostic_targets": diagnostic_targets or {},
        "diagnostic_sources": diagnostic_sources,
        "shard_summaries": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=list(STANDARD_BENCHMARK_DATASETS)
    )
    parser.add_argument(
        "--minimum_model_families", type=int,
        default=GLOBAL_MODEL_FAMILY_TARGET,
    )
    parser.add_argument(
        "--dataset_family_targets", nargs="*",
        help="Per-dataset family requirements such as vg=4 oi=2 psg=2.",
    )
    parser.add_argument(
        "--task_contract", choices=("full", "sgdet_only", "tritask_depth"),
        default="full",
        help="Task coverage contract used by the shard-producing matrix.",
    )
    parser.add_argument("--experiment2_root")
    parser.add_argument("--experiment3_root")
    parser.add_argument(
        "--external_diagnostic_targets", nargs="*",
        help="External diagnostic requirements such as gqa=1 vrd=1.",
    )
    args = parser.parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    targets = parse_dataset_targets(
        args.dataset_family_targets,
        {
            dataset: STANDARD_DATASET_FAMILY_TARGETS.get(dataset, 0)
            for dataset in datasets
        },
    )
    diagnostic_targets = parse_dataset_targets(
        args.external_diagnostic_targets,
        EXTERNAL_DATASET_MODEL_TARGETS if (
            args.experiment2_root or args.experiment3_root
        ) else {},
    )
    diagnostic_roots = [
        Path(value).expanduser().resolve()
        for value in (args.experiment2_root, args.experiment3_root)
        if value
    ]
    payload = aggregate(
        Path(args.shard_root).expanduser().resolve(),
        datasets,
        args.minimum_model_families,
        targets,
        diagnostic_roots=diagnostic_roots,
        diagnostic_targets=diagnostic_targets,
        task_contract=args.task_contract,
    )
    write_json(Path(args.output), payload)
    print(f"Formal aggregate: {args.output}")


if __name__ == "__main__":
    main()
