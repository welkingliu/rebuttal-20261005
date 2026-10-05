#!/usr/bin/env python3
"""Run mitigation on one classic and one transformer family with three seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.submission_protocol import MITIGATION_SEEDS

GATE_PROTOCOL_VERSION = "exp5-live-validation-v2"


def _first(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def _dataset_args(root: Path, dataset: str) -> list[str]:
    data = root / "data"
    if dataset == "vg":
        return ["--data_root", str(_first(data / "vg" / "v1.4", data / "vg"))]
    if dataset == "oi":
        return ["--data_root", str(data / "openimages" / "open-images-v6")]
    if dataset == "gqa":
        return [
            "--data_root", str(data / "gqa"),
            "--train_ann", str(_first(
                data / "gqa" / "train_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
            )),
            "--eval_ann", str(_first(
                data / "gqa" / "val_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
            )),
            "--image_root", str(data / "gqa" / "images"),
        ]
    if dataset == "psg":
        return [
            "--data_root", str(data / "psg"),
            "--train_ann", str(data / "psg" / "psg_train_val.json"),
            "--eval_ann", str(_first(
                data / "psg" / "psg_val_test.json", data / "psg" / "psg.json"
            )),
            "--image_root", str(data / "coco"),
            "--panoptic_root", str(data / "coco"),
        ]
    return ["--data_root", str(data / "vrd")]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _nested(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _result_row(outcome: dict) -> dict:
    path = Path(outcome["output"]) / "mitigation_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    before = payload.get("before_test") or payload["before_validation"]
    after = payload.get("after_test") or payload["after_validation"]

    def metrics(result):
        return {
            "object_top1": _nested(
                result, "grounding_error_decomposition", "object_identity",
                "top1_accuracy_given_localized",
            ),
            "object_ece_15": _nested(
                result, "grounding_error_decomposition", "object_identity", "ece_15"
            ),
            "SGCls_mR@50": _nested(
                result, "standard_sgg", "tasks", "sgcls", "metrics", "mR@50"
            ),
            "SGCls_R@50": _nested(
                result, "standard_sgg", "tasks", "sgcls", "metrics", "R@50"
            ),
            "SGDet_mR@50": _nested(
                result, "standard_sgg", "tasks", "sgdet", "metrics", "mR@50"
            ),
            "SGDet_R@50": _nested(
                result, "standard_sgg", "tasks", "sgdet", "metrics", "R@50"
            ),
            "SGDet_zR@50": _nested(
                result, "standard_sgg", "tasks", "sgdet", "metrics", "zR@50"
            ),
        }

    before_metrics = metrics(before)
    after_metrics = metrics(after)
    changes = {
        key: (
            float(after_metrics[key]) - float(before_metrics[key])
            if isinstance(after_metrics[key], (int, float))
            and isinstance(before_metrics[key], (int, float))
            and np.isfinite(after_metrics[key]) and np.isfinite(before_metrics[key])
            else None
        )
        for key in after_metrics
    }
    return {
        **{key: outcome[key] for key in (
            "model", "family", "training_mode", "seed", "output"
        )},
        "selected_epoch": payload.get("selected_epoch"),
        "acceptance_passed": payload.get("acceptance", {}).get("passed"),
        "before": before_metrics,
        "after": after_metrics,
        "change": changes,
    }


def _mean_std(values) -> dict:
    finite = np.asarray([
        float(value) for value in values
        if isinstance(value, (int, float)) and np.isfinite(value)
    ], dtype=np.float64)
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std(ddof=0)) if finite.size else None,
    }


def _aggregate_results(outcomes: list[dict]) -> dict:
    rows = [_result_row(outcome) for outcome in outcomes]
    metrics = (
        "object_top1", "object_ece_15", "SGCls_R@50", "SGCls_mR@50",
        "SGDet_R@50", "SGDet_mR@50", "SGDet_zR@50",
    )
    groups = {}
    for family in sorted({row["family"] for row in rows}):
        groups[family] = {}
        for mode in ("supervised_control", "grounding"):
            subset = [
                row for row in rows
                if row["family"] == family and row["training_mode"] == mode
            ]
            groups[family][mode] = {
                "runs": len(subset),
                "acceptance_passes": sum(bool(row["acceptance_passed"]) for row in subset),
                "after": {
                    metric: _mean_std(row["after"][metric] for row in subset)
                    for metric in metrics
                },
                "change_from_pretrained": {
                    metric: _mean_std(row["change"][metric] for row in subset)
                    for metric in metrics
                },
            }
        paired = {}
        for metric in metrics:
            differences = []
            for seed in sorted({row["seed"] for row in rows if row["family"] == family}):
                by_mode = {
                    row["training_mode"]: row for row in rows
                    if row["family"] == family and row["seed"] == seed
                }
                if set(by_mode) != {"supervised_control", "grounding"}:
                    continue
                control = by_mode["supervised_control"]["after"][metric]
                grounding = by_mode["grounding"]["after"][metric]
                if (
                    isinstance(control, (int, float))
                    and isinstance(grounding, (int, float))
                    and np.isfinite(control) and np.isfinite(grounding)
                ):
                    differences.append(float(grounding) - float(control))
            paired[metric] = _mean_std(differences)
        groups[family]["grounding_minus_supervised_control"] = paired
    return {"rows": rows, "by_family": groups}


def _select_manifest(manifest_dir: Path, family: str, dataset: str) -> dict:
    candidates = []
    for path in sorted(manifest_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("architecture_family") != family:
            continue
        supported = {str(value).lower() for value in payload.get("supported_datasets", [])}
        supported_tasks = {
            str(value).lower() for value in payload.get("supported_tasks", [])
        }
        if dataset not in supported:
            continue
        mitigation = payload.get("mitigation_contract", {})
        if not (
            payload.get("execution_mode", "live_adapter") == "live_adapter"
            and
            mitigation.get("forward_grounding") is True
            and mitigation.get("trainable_grounding_parameters") is True
            and mitigation.get("object_logits") is True
            and mitigation.get("trainable_object_parameters") is True
            and mitigation.get("relation_logit_alignment") == "gt_relations"
            and mitigation.get("object_logit_alignment") == "gt_entities"
            and "sgdet" in supported_tasks
        ):
            continue
        candidates.append({
            "path": path.resolve(),
            "name": str(payload.get("name", "")).strip(),
            "family": family,
            "python": str(payload.get("environment_python") or sys.executable),
            "training_seed": int(payload.get("training_seed", 2**31 - 1)),
        })
    if not candidates:
        raise RuntimeError(
            f"No {dataset}-compatible mitigation manifest for family {family!r}"
        )
    return sorted(
        candidates,
        key=lambda item: (item["training_seed"] != 17, item["training_seed"], item["name"]),
    )[0]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_family(gpu: str, record: dict, args, root: Path,
                output: Path, wait_pid: int | None = None) -> list[dict]:
    if wait_pid is not None:
        print(
            f"[gpu={gpu}] waiting for upstream_pid={wait_pid} before "
            f"family={record['family']}",
            flush=True,
        )
        while _pid_exists(wait_pid):
            time.sleep(args.wait_poll_seconds)
        print(
            f"[gpu={gpu}] upstream complete; starting family={record['family']}",
            flush=True,
        )
    python_path = Path(record["python"]).expanduser()
    if not python_path.is_file():
        raise FileNotFoundError(
            f"environment_python for {record['name']} not found: {python_path}"
        )
    outcomes = []
    for training_mode in args.training_modes:
        for seed in args.seeds:
            family_name = _safe_name(record["family"])
            run_output = output / training_mode / family_name / f"seed_{seed}"
            before_test_cache = (
                output / "shared_baselines" / family_name / "before_test.json"
            )
            log_path = (
                output / "logs" / f"{training_mode}_{family_name}_seed_{seed}.log"
            )
            run_output.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            result_path = run_output / "mitigation_results.json"
            if args.resume and result_path.is_file():
                outcomes.append({
                    "model": record["name"],
                    "family": record["family"],
                    "training_mode": training_mode,
                    "seed": seed,
                    "gpu": str(gpu),
                    "returncode": 0,
                    "output": str(run_output),
                    "log": str(log_path),
                    "resumed": True,
                })
                print(
                    f"[gpu={gpu}] resume-skip experiment=5 mode={training_mode} "
                    f"family={record['family']} seed={seed}",
                    flush=True,
                )
                continue
            command = [
                record["python"], "-m", "sgg_core.experiments.experiment_5",
                "--manifest", str(record["path"]),
                "--dataset", args.dataset,
                "--output_dir", str(run_output),
                "--epochs", str(args.epochs),
                "--minimum_epochs", str(args.minimum_epochs),
                "--early_stopping_patience",
                str(args.early_stopping_patience),
                "--train_samples", str(args.train_samples),
                "--eval_samples", str(args.eval_samples),
                "--test_samples", str(args.test_samples),
                "--before_test_cache", str(before_test_cache),
                "--seed", str(seed),
                "--training_mode", training_mode,
                "--gradient_accumulation_steps",
                str(args.gradient_accumulation_steps),
                "--device", "cuda",
                *_dataset_args(root, args.dataset),
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = str(root) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            print(
                f"[gpu={gpu}] experiment=5 mode={training_mode} "
                f"family={record['family']} seed={seed}",
                flush=True,
            )
            completed = None
            for attempt in range(1, args.max_run_attempts + 1):
                log_mode = "w" if attempt == 1 else "a"
                with log_path.open(log_mode, encoding="utf-8") as log:
                    if attempt > 1:
                        log.write(
                            f"\n[retry] attempt={attempt}/"
                            f"{args.max_run_attempts}\n"
                        )
                        log.flush()
                    completed = subprocess.run(
                        command, cwd=root, env=env,
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                if completed.returncode == 0:
                    break
                print(
                    f"[gpu={gpu}] run failed mode={training_mode} "
                    f"family={record['family']} seed={seed} "
                    f"attempt={attempt}/{args.max_run_attempts}; "
                    f"see {log_path}",
                    flush=True,
                )
                if attempt < args.max_run_attempts:
                    time.sleep(args.retry_delay_seconds)
            assert completed is not None
            outcome = {
                "model": record["name"],
                "family": record["family"],
                "training_mode": training_mode,
                "seed": seed,
                "gpu": str(gpu),
                "returncode": completed.returncode,
                "attempts": attempt,
                "output": str(run_output),
                "log": str(log_path),
            }
            outcomes.append(outcome)
            if completed.returncode:
                raise RuntimeError(
                    f"Mitigation failed: mode={training_mode} "
                    f"family={record['family']} seed={seed}; "
                    f"see {log_path}"
                )
            print(
                f"[gpu={gpu}] {record['family']} {training_mode} "
                f"seed {seed} 实验结束",
                flush=True,
            )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--gate_report",
        help="Passing report from run_experiment5_gate.py required for execution.",
    )
    parser.add_argument(
        "--allow_failed_gate",
        action="store_true",
        help=(
            "Run the full matrix after a protocol-valid failed gate. The override "
            "and original gate result remain recorded in the matrix contract."
        ),
    )
    parser.add_argument("--classic_family", default="Neural Motifs")
    parser.add_argument("--transformer_family", default="SGG Transformer")
    parser.add_argument("--dataset", choices=("vg",), default="vg")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(MITIGATION_SEEDS))
    parser.add_argument(
        "--training_modes", nargs="+",
        choices=("supervised_control", "grounding"),
        default=["supervised_control", "grounding"],
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["0", "1"],
        help="One GPU runs the two families sequentially; two GPUs run them in parallel.",
    )
    parser.add_argument(
        "--wait_gpu", action="append", default=[], metavar="GPU=PID",
        help="Delay the family assigned to GPU until the upstream PID exits.",
    )
    parser.add_argument("--wait_poll_seconds", type=float, default=30.0)
    parser.add_argument("--max_run_attempts", type=int, default=1)
    parser.add_argument("--retry_delay_seconds", type=float, default=60.0)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minimum_epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1_000_000_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    if args.classic_family == args.transformer_family:
        raise ValueError("Mitigation requires two distinct architecture families")
    if not 1 <= args.minimum_epochs <= args.epochs:
        raise ValueError("--minimum_epochs must be between 1 and --epochs")
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be positive")
    if len(set(args.seeds)) != 3:
        raise ValueError("Formal mitigation requires exactly three distinct seeds")
    if set(args.training_modes) != {"supervised_control", "grounding"}:
        raise ValueError(
            "Formal mitigation requires supervised_control and grounding modes"
        )
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Experiment V requires one or two distinct GPU identifiers")
    if args.wait_poll_seconds <= 0:
        raise ValueError("--wait_poll_seconds must be positive")
    if args.max_run_attempts < 1:
        raise ValueError("--max_run_attempts must be positive")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry_delay_seconds must be non-negative")
    wait_by_gpu = {}
    for value in args.wait_gpu:
        gpu, separator, pid = str(value).partition("=")
        if not separator or gpu not in args.gpus or not pid.isdigit():
            raise ValueError(
                f"Invalid --wait_gpu {value!r}; expected one of "
                f"{args.gpus} followed by '=PID'"
            )
        if gpu in wait_by_gpu:
            raise ValueError(f"Duplicate --wait_gpu entry for GPU {gpu}")
        wait_by_gpu[gpu] = int(pid)
    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve()
        if args.manifest_dir else root / "checkpoints" / "sgg" / "manifests"
    )
    records = [
        _select_manifest(manifest_dir, args.classic_family, args.dataset),
        _select_manifest(manifest_dir, args.transformer_family, args.dataset),
    ]
    gate = None
    if not args.check_only:
        if not args.gate_report:
            raise RuntimeError(
                "Formal Experiment V is gated. Run scripts/run_experiment5_gate.py "
                "and provide --gate_report before expanding the 12-run matrix."
            )
        gate_path = Path(args.gate_report).expanduser().resolve()
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_protocol_valid = (
            gate.get("protocol_version") == GATE_PROTOCOL_VERSION
            and gate.get("dataset") == args.dataset
        )
        if not gate_protocol_valid:
            raise RuntimeError(
                f"Experiment V gate report has an invalid protocol: {gate_path}"
            )
        gate_passed = (
            gate.get("passed") is True and gate.get("status") == "pass"
        )
        if not gate_passed and not args.allow_failed_gate:
            raise RuntimeError(f"Experiment V gate did not pass: {gate_path}")
        if not gate_passed:
            print(
                "[WARNING] Experiment V gate failed; continuing under the "
                "explicit --allow_failed_gate override.",
                flush=True,
            )
    contract = {
        "experiment": "V_grounding_dependency_mitigation",
        "dataset": args.dataset,
        "classic_family": args.classic_family,
        "transformer_family": args.transformer_family,
        "seeds": args.seeds,
        "training_modes": args.training_modes,
        "early_stopping": {
            "maximum_epochs": args.epochs,
            "minimum_epochs": args.minimum_epochs,
            "patience": args.early_stopping_patience,
            "selection_split": "validation",
        },
        "gpus": args.gpus,
        "wait_gpu": wait_by_gpu,
        "models": [{**record, "path": str(record["path"])} for record in records],
        "gate_report": (
            str(Path(args.gate_report).expanduser().resolve())
            if args.gate_report else None
        ),
        "gate": gate,
        "gate_enforcement": {
            "passed": bool(gate and gate.get("passed") is True),
            "failed_gate_override": bool(
                gate and gate.get("passed") is not True
                and args.allow_failed_gate
            ),
        },
    }
    (output / "matrix_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    if args.check_only:
        print(
            "[READY] Experiment 5: "
            f"dataset={args.dataset} families="
            f"{[record['family'] for record in records]} runs="
            f"{len(records) * len(args.training_modes) * len(args.seeds)}"
        )
        return

    assignments = [
        (args.gpus[index % len(args.gpus)], record)
        for index, record in enumerate(records)
    ]
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                _run_family, gpu, record, args, root, output,
                wait_by_gpu.get(gpu),
            )
            for gpu, record in assignments
        ]
        for future in futures:
            outcomes.extend(future.result())
    result_aggregate = _aggregate_results(outcomes)
    summary = {
        **contract,
        "status": "complete",
        "jobs": outcomes,
        "result_aggregate": result_aggregate,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Experiment V mitigation matrix complete: {output}")


if __name__ == "__main__":
    main()
