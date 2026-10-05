#!/usr/bin/env python3
"""Recover an Experiment V result that failed only during JSON serialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sgg_core.mitigation.run_mitigation import _acceptance, _json_default


def _load_completed_prefix(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        marker = '\n  "acceptance":'
        prefix, separator, _ = text.partition(marker)
        if not separator:
            raise RuntimeError(
                "Partial result does not contain complete before/after evaluations"
            )
        return json.loads(prefix.rstrip().rstrip(",") + "\n}")


def _log_records(path: Path):
    history = []
    early_stopping = None
    for line in path.read_text(encoding="utf-8", errors="replace").replace(
        "\r", "\n"
    ).splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in record and "epoch" in record:
            history.append(record)
        if record.get("event") == "early_stopping":
            early_stopping = record
    return history, early_stopping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    result_path = run_dir / "mitigation_results.json"
    checkpoint_path = run_dir / "mitigated_state_dict.pth"
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Partial result and mitigated checkpoint are required")

    payload = _load_completed_prefix(result_path)
    if "acceptance" in payload:
        print(f"[ok] result already valid: {result_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_args = checkpoint.get("training_args")
    if not isinstance(training_args, dict):
        raise RuntimeError("Checkpoint does not contain training_args")
    history, early_event = _log_records(Path(args.log).expanduser().resolve())
    if not history:
        raise RuntimeError("Training history was not recoverable from the log")

    namespace = SimpleNamespace(**training_args)
    payload.update({
        "acceptance": _acceptance(
            payload["before_validation"], payload["after_validation"], namespace
        ),
        "history": history,
        "selection_history": [],
        "selection_history_recovery": (
            "Per-epoch selection metrics were not logged before the serialization "
            "failure; selected_epoch is recovered from the saved checkpoint."
        ),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "epochs_completed": len(history),
        "early_stopping": {
            "enabled": True,
            "maximum_epochs": int(namespace.epochs),
            "minimum_epochs": int(namespace.minimum_epochs),
            "patience": int(namespace.early_stopping_patience),
            "stopped_early": early_event is not None,
            "reason": (
                early_event.get("reason")
                if early_event is not None else "maximum_epochs_reached"
            ),
        },
        "checkpoint": str(checkpoint_path),
        "claim_scope": (
            "Validation selects the mitigation; VG split-2 is an untouched "
            "within-dataset test. OOD requires a separate ontology-compatible run."
        ),
        "recovered_after_serialization_failure": True,
    })
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(result_path)
    print(f"[recovered] {result_path}")


if __name__ == "__main__":
    main()
