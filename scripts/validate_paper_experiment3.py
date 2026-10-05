#!/usr/bin/env python3
"""Validate the strict two-model Experiment III result contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_graph_result(payload: dict) -> dict:
    audit = payload.get("graph_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("Missing graph_audit")
    if "models" in audit and isinstance(audit["models"], dict):
        values = list(audit["models"].values())
    else:
        values = [
            value for value in audit.values()
            if isinstance(value, dict) and "error_count" in value
        ]
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one model result per Experiment III file, found {len(values)}"
        )
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    paths = [
        root / "neural_motifs" / "experiment_3.json",
        root / "sgg_transformer" / "experiment_3.json",
    ]
    report = {"status": "pass", "models": {}}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = _find_graph_result(payload)
        checks = {
            "error_count_zero": model.get("error_count") == 0,
            "twenty_motifs": model.get("motifs_used") == 20,
            "full_pair_set": model.get("total_motif_pairs") == 1411,
        }
        if not all(checks.values()):
            report["status"] = "fail"
        report["models"][path.parent.name] = {
            "path": str(path),
            "checks": checks,
            "error_count": model.get("error_count"),
            "motif_count": model.get("motifs_used"),
            "eligible_pairs": model.get("total_motif_pairs"),
        }
    destination = root / "validation_report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{report['status'].upper()}] Experiment III contract: {destination}")
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
