#!/usr/bin/env python3
"""Validate and combine the converged Experiment IV, II, and V outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXP2_FAMILIES = {"Neural Motifs", "SGG Transformer"}
EXP5_FAMILIES = {"Neural Motifs", "SGG Transformer"}


def _load(path: str) -> tuple[Path, dict]:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        value = value / "summary.json"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value, json.loads(value.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment4", required=True)
    parser.add_argument("--experiment2", required=True)
    parser.add_argument("--experiment5", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    exp4_path, exp4 = _load(args.experiment4)
    exp2_path, exp2 = _load(args.experiment2)
    exp5_path, exp5 = _load(args.experiment5)
    failures = []
    if exp2.get("status") != "complete":
        failures.append("experiment_2_incomplete")
    if set(exp2.get("families", [])) != EXP2_FAMILIES:
        failures.append("experiment_2_family_contract")
    if len(exp2.get("jobs", [])) != 2:
        failures.append("experiment_2_job_count")
    if exp5.get("status") != "complete":
        failures.append("experiment_5_incomplete")
    exp5_families = {
        str(row.get("family")) for row in exp5.get("result_aggregate", {}).get("rows", [])
    }
    if exp5_families != EXP5_FAMILIES:
        failures.append("experiment_5_family_contract")
    if len(exp5.get("jobs", [])) != 12:
        failures.append("experiment_5_job_count")

    coverage = exp4.get("coverage", {})
    if not coverage:
        coverage = exp4.get("dataset_coverage", {})
    rows = exp4.get("cross_dataset_rows", [])
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "experiment4_cross_dataset.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema": "mandatory_submission_aggregate_v1",
        "status": "complete" if not failures else "failed_contract",
        "failures": failures,
        "scope": {
            "experiment_4": (
                "two-family VG tri-task depth; broad native SGDet is retained "
                "as a separately provenance-validated result slice"
            ),
            "experiment_2": "two-family five-dose intervention matrix",
            "experiment_5": "two-family two-mode three-seed mitigation",
        },
        "sources": {
            "experiment_4": str(exp4_path),
            "experiment_2": str(exp2_path),
            "experiment_5": str(exp5_path),
        },
        "experiment_4_coverage": coverage,
        "experiment_4_cross_dataset_rows": rows,
        "experiment_2_jobs": exp2.get("jobs", []),
        "experiment_5_result_aggregate": exp5.get("result_aggregate", {}),
    }
    path = output / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("summary=" + str(path))
    if failures:
        raise SystemExit("[FAILED CONTRACT] " + ", ".join(failures))
    print("[COMPLETE] mandatory submission aggregate")


if __name__ == "__main__":
    main()
