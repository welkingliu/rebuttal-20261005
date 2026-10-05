#!/usr/bin/env python3
"""Validate the compute-converged submission manifest matrix before GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.submission_protocol import (
    DIAGNOSTIC_MODEL_RANGE,
    EXTERNAL_DATASET_MODEL_TARGETS,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_DATASET_FAMILY_TARGETS,
    STANDARD_TASK_FAMILY_TARGETS,
)


DEFAULT_FAMILIES = ["Neural Motifs", "SGG Transformer"]


def _read_manifests(path: Path) -> list[dict]:
    records = []
    for manifest in sorted(path.glob("*.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        family = str(payload.get("architecture_family", "")).strip()
        name = str(payload.get("name", "")).strip()
        datasets = {
            str(dataset).lower() for dataset in payload.get("supported_datasets", [])
        }
        ontology_ids = payload.get("ontology_ids", {})
        if not family or not name or not datasets or not isinstance(ontology_ids, dict):
            raise ValueError(f"Incomplete submission identity: {manifest}")
        inexact = sorted(
            dataset for dataset in datasets
            if not ontology_ids.get(dataset) or ontology_ids.get(dataset) == "*"
        )
        if inexact:
            raise ValueError(
                f"Formal manifest requires exact ontology IDs for {inexact}: {manifest}"
            )
        records.append({
            "path": str(manifest.resolve()),
            "name": name,
            "family": family,
            "datasets": sorted(datasets),
            "mitigation_contract": payload.get("mitigation_contract", {}),
            "perturbation_contract": payload.get("perturbation_contract", {}),
            "diagnostic_contract": payload.get("diagnostic_contract", {}),
            "execution_mode": payload.get("execution_mode", "live_adapter"),
            "supported_tasks": sorted({
                str(task).lower() for task in payload.get("supported_tasks", [])
            }),
        })
    return records


def _families_for(records: list[dict], dataset: str,
                  selected: set[str] | None = None) -> set[str]:
    return {
        record["family"] for record in records
        if dataset in record["datasets"]
        and (selected is None or record["family"] in selected)
    }


def _supports_diagnostic(record: dict, experiment: int) -> bool:
    if record["execution_mode"] != "live_adapter":
        return False
    diagnostic = record["diagnostic_contract"]
    perturbation = record["perturbation_contract"]
    if experiment == 2:
        required = {
            "full", "visual_noise", "union_attenuation",
            "on_manifold_replacement", "random_node_mask",
            "key_node_mask", "unrelated_node_mask",
        }
        return (
            diagnostic.get("gt_pair_predict") is True
            and all(perturbation.get(name) is True for name in required)
        )
    return (
        diagnostic.get("gt_node_features") is True
        and diagnostic.get("graph_intervention") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest_dir")
    parser.add_argument("--report")
    parser.add_argument("--exp2_families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--exp3_families", nargs="*", default=[])
    parser.add_argument("--mitigation_classic_family", default="Neural Motifs")
    parser.add_argument("--mitigation_transformer_family", default="SGG Transformer")
    parser.add_argument("--mitigation_dataset", default="vg")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve()
        if args.manifest_dir else root / "checkpoints" / "sgg" / "manifests"
    )
    report_path = (
        Path(args.report).expanduser().resolve() if args.report
        else root / "artifacts" / "manifests" / "submission_contract.json"
    )
    records = _read_manifests(manifest_dir)
    all_families = {record["family"] for record in records}
    standard_families = {
        record["family"] for record in records
        if set(record["datasets"]) & set(STANDARD_DATASET_FAMILY_TARGETS)
    }
    failures = []

    if len(standard_families) < GLOBAL_MODEL_FAMILY_TARGET:
        failures.append(
            "standard_matrix_families="
            f"{len(standard_families)}/{GLOBAL_MODEL_FAMILY_TARGET}"
        )
    standard_counts = {}
    for dataset, target in STANDARD_DATASET_FAMILY_TARGETS.items():
        families = _families_for(records, dataset)
        standard_counts[dataset] = len(families)
        if len(families) < target:
            failures.append(f"{dataset}_standard={len(families)}/{target}")
    task_counts = {}
    for dataset, task_targets in STANDARD_TASK_FAMILY_TARGETS.items():
        task_counts[dataset] = {}
        for task, target in task_targets.items():
            families = {
                record["family"] for record in records
                if dataset in record["datasets"] and task in record["supported_tasks"]
            }
            task_counts[dataset][task] = len(families)
            if len(families) < target:
                failures.append(
                    f"{dataset}_{task}_families={len(families)}/{target}"
                )
    external_counts = {}
    for dataset, target in EXTERNAL_DATASET_MODEL_TARGETS.items():
        families = _families_for(records, dataset)
        external_counts[dataset] = len(families)
        if len(families) < target:
            failures.append(f"{dataset}_diagnostic={len(families)}/{target}")

    exp2 = set(args.exp2_families)
    exp3 = set(args.exp3_families)
    selected_diagnostics = [("exp2", exp2)]
    if exp3:
        selected_diagnostics.append(("exp3", exp3))
    for label, selected in selected_diagnostics:
        lower, upper = DIAGNOSTIC_MODEL_RANGE
        if not lower <= len(selected) <= upper:
            failures.append(
                f"{label}_live_diagnostic_families={len(selected)}/{lower}-{upper}"
            )
        missing = sorted(selected - all_families)
        if missing:
            failures.append(f"{label}_missing_families={missing}")
    exp2_records = [record for record in records if _supports_diagnostic(record, 2)]
    exp3_records = [record for record in records if _supports_diagnostic(record, 3)]
    exp2_targets = {"vg": 2}
    exp3_targets = {"vg": 2}
    for dataset, target in exp2_targets.items():
        count = len(_families_for(exp2_records, dataset, exp2))
        if count < target:
            failures.append(f"exp2_{dataset}={count}/{target}")
    if exp3:
        for dataset, target in exp3_targets.items():
            count = len(_families_for(exp3_records, dataset, exp3))
            if count < target:
                failures.append(f"exp3_{dataset}={count}/{target}")

    mitigation_families = {
        args.mitigation_classic_family, args.mitigation_transformer_family,
    }
    if len(mitigation_families) != 2:
        failures.append("mitigation_requires_two_distinct_families")
    for family in mitigation_families:
        candidates = [
            record for record in records
            if record["family"] == family
            and args.mitigation_dataset in record["datasets"]
        ]
        if not candidates:
            failures.append(
                f"mitigation_{args.mitigation_dataset}_missing={family}"
            )
            continue
        declared = any(
            record["execution_mode"] == "live_adapter"
            and
            record["mitigation_contract"].get("forward_grounding") is True
            and record["mitigation_contract"].get(
                "trainable_grounding_parameters"
            ) is True
            and record["mitigation_contract"].get("object_logits") is True
            and record["mitigation_contract"].get(
                "trainable_object_parameters"
            ) is True
            and record["mitigation_contract"].get(
                "relation_logit_alignment"
            ) == "gt_relations"
            and record["mitigation_contract"].get(
                "object_logit_alignment"
            ) == "gt_entities"
            for record in candidates
        )
        if not declared:
            failures.append(f"mitigation_contract_missing={family}")

    report = {
        "status": "ready" if not failures else "not_ready",
        "global_family_count": len(standard_families),
        "global_family_target": GLOBAL_MODEL_FAMILY_TARGET,
        "families": sorted(all_families),
        "standard_matrix_families": sorted(standard_families),
        "standard_family_counts": standard_counts,
        "standard_family_targets": STANDARD_DATASET_FAMILY_TARGETS,
        "standard_task_family_counts": task_counts,
        "standard_task_family_targets": STANDARD_TASK_FAMILY_TARGETS,
        "external_family_counts": external_counts,
        "external_family_targets": EXTERNAL_DATASET_MODEL_TARGETS,
        "experiment_2_families": args.exp2_families,
        "experiment_3_families": args.exp3_families,
        "mitigation_families": sorted(mitigation_families),
        "failures": failures,
        "manifests": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Submission manifest contract")
    print("=" * 72)
    print(
        "  standard-matrix families: "
        f"{len(standard_families)}/{GLOBAL_MODEL_FAMILY_TARGET}"
    )
    print(f"  standard coverage: {standard_counts}")
    print(f"  task coverage: {task_counts}")
    print(f"  external coverage: {external_counts}")
    print(f"  report={report_path}")
    if failures:
        print("[NOT READY] " + "; ".join(failures))
        raise SystemExit(1)
    print("[READY] Compute-converged submission contract is satisfied.")


if __name__ == "__main__":
    main()
