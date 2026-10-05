#!/usr/bin/env python3
"""Run one cache-free mitigation pilot before expanding the formal matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment5_matrix import _dataset_args, _select_manifest


PROTOCOL_VERSION = "exp5-live-validation-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--family", default="SGG Transformer")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minimum_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--minimum_validation_objects", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve()
        if args.manifest_dir
        else root / "checkpoints" / "sgg" / "manifests"
    )
    output = Path(args.output_dir).expanduser().resolve()
    run_output = output / "grounding" / args.family.replace(" ", "_") / f"seed_{args.seed}"
    log_path = output / "logs" / f"gate_{args.family.replace(' ', '_')}_seed_{args.seed}.log"
    report_path = output / "gate_report.json"
    run_output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = _select_manifest(manifest_dir, args.family, "vg")

    command = [
        record["python"], "-m", "sgg_core.experiments.experiment_5",
        "--manifest", str(record["path"]),
        "--dataset", "vg",
        "--output_dir", str(run_output),
        "--epochs", str(args.epochs),
        "--minimum_epochs", str(args.minimum_epochs),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--train_samples", str(args.train_samples),
        "--eval_samples", str(args.eval_samples),
        "--skip_test",
        "--minimum_validation_objects", str(args.minimum_validation_objects),
        "--seed", str(args.seed),
        "--training_mode", "grounding",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--device", "cuda",
        *_dataset_args(root, "vg"),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(root) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else ""
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=root, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )

    result_path = run_output / "mitigation_results.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if completed.returncode == 0 and result_path.is_file()
        else {}
    )
    acceptance = result.get("acceptance", {})
    checks = {
        "process_completed": completed.returncode == 0,
        "result_written": bool(result),
        "live_validation_protocol_valid": (
            acceptance.get("protocol_valid") is True
        ),
        "object_parameters_updated": (
            acceptance.get("object_parameters_updated") is True
        ),
        "object_top1_improved": (
            acceptance.get("object_top1", {}).get("passed") is True
        ),
        "object_calibration_preserved": (
            acceptance.get("object_ece_15", {}).get("passed") is True
        ),
        "sgcls_non_degraded": (
            acceptance.get("standard_task_non_degradation", {})
            .get("sgcls", {}).get("passed") is True
        ),
        "acceptance_passed": acceptance.get("passed") is True,
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
        "passed": all(checks.values()),
        "dataset": "vg",
        "family": args.family,
        "seed": args.seed,
        "training_mode": "grounding",
        "checks": checks,
        "acceptance": acceptance,
        "result": str(result_path),
        "log": str(log_path),
        "command": command,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{report['status'].upper()}] Experiment V gate: {report_path}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
