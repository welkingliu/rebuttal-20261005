#!/usr/bin/env python3
"""Run the paper's fixed one-family TDE-Motifs Experiment V protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment5_matrix import _aggregate_results, _dataset_args


FAMILY = "TDE-Motifs"
MODES = ("supervised_control", "grounding")
SEEDS = (17, 23, 31)


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TDE-Motifs live manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "architecture_family": FAMILY,
        "execution_mode": "live_adapter",
        "reference_dataset": "vg",
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    contract = payload.get("mitigation_contract", {})
    for key in (
        "forward_grounding",
        "object_logits",
        "trainable_object_parameters",
        "trainable_grounding_parameters",
    ):
        if contract.get(key) is not True:
            mismatches[f"mitigation_contract.{key}"] = {
                "expected": True,
                "observed": contract.get(key),
            }
    if mismatches:
        raise RuntimeError(
            "Manifest does not satisfy the paper Experiment V contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def _stream(command: list[str], cwd: Path, env: dict[str, str],
            log_path: Path, prefix: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{prefix}] {line}", end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"Command failed with exit code {returncode}; see {log_path}"
        )


def _job_command(args, python: str, output: Path, mode: str,
                 seed: int) -> list[str]:
    return [
        python,
        "-m",
        "sgg_core.experiments.experiment_5",
        "--manifest",
        str(args.manifest),
        "--dataset",
        "vg",
        "--data_root",
        str(args.data_root),
        "--output_dir",
        str(output),
        "--epochs",
        "5",
        "--minimum_epochs",
        "3",
        "--early_stopping_patience",
        "1",
        "--train_samples",
        str(args.train_samples),
        "--eval_samples",
        str(args.eval_samples),
        "--test_samples",
        str(args.test_samples),
        "--skip_test",
        "--learning_rate",
        "3e-5",
        "--max_mr_drop",
        "0.005",
        "--stop_on_mr_drop",
        "--minimum_object_top1_gain",
        "0.005",
        "--max_object_ece_increase",
        "0.01",
        "--minimum_validation_objects",
        str(args.minimum_validation_objects),
        "--gradient_accumulation_steps",
        "4",
        "--object_weight",
        "2.25",
        "--task_driven_object_focus",
        "--task_object_weight",
        "3.0",
        "--freeze_relation_parameters",
        "--object_weight_delta_scale",
        "1.1",
        "--object_bias_delta_scale",
        "0.5",
        "--progress_interval_images",
        str(args.progress_interval_images),
        "--seed",
        str(seed),
        "--training_mode",
        mode,
        "--device",
        "cuda",
    ]


def _verify_result(path: Path, mode: str, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") or []
    if not history:
        raise RuntimeError(f"No training history in {path}")
    checks = {
        "mode": payload.get("training_mode") == mode,
        "seed": payload.get("seed") == seed,
        "epochs_at_least_three": payload.get("epochs_completed", 0) >= 3,
        "epochs_at_most_five": payload.get("epochs_completed", 99) <= 5,
        "relation_frozen": all(
            row.get("freeze_relation_parameters") is True for row in history
        ),
        "task_focus": all(
            row.get("task_driven_object_focus") is True for row in history
        ),
        "task_weight": all(
            row.get("task_object_weight") == 3.0 for row in history
        ),
        "object_only": all(
            row.get("optimized_parameter_groups") == ["object"] for row in history
        ),
        "delta_weight": (
            payload.get("object_delta_scaling", {}).get("weight") == 1.1
        ),
        "delta_bias": (
            payload.get("object_delta_scaling", {}).get("bias") == 0.5
        ),
        "validation_only_selection": (
            payload.get("acceptance", {}).get("selection_split") == "validation"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Experiment V protocol mismatch in {path}: "
            + json.dumps(checks, sort_keys=True)
        )
    return {"checks": checks, "acceptance": payload.get("acceptance")}


def _run_job(args, root: Path, python: str, output_root: Path,
             gpu: str, mode: str, seed: int) -> dict:
    family_key = FAMILY.replace("-", "_")
    output = output_root / mode / family_key / f"seed_{seed}"
    result_path = output / "mitigation_results.json"
    log_path = output_root / "logs" / f"{mode}_seed_{seed}.log"
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and result_path.is_file():
        print(f"[resume mode={mode} seed={seed}] {result_path}", flush=True)
    else:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        command = _job_command(args, python, output, mode, seed)
        _stream(
            command,
            root,
            env,
            log_path,
            f"gpu={gpu} mode={mode} seed={seed}",
        )
    verification = _verify_result(result_path, mode, seed)
    print(f"[complete] TDE-Motifs {mode} seed {seed} 实验结束", flush=True)
    return {
        "model": "pysgg_tde_motifs_vg_live",
        "family": FAMILY,
        "training_mode": mode,
        "seed": seed,
        "gpu": gpu,
        "returncode": 0,
        "output": str(output),
        "log": str(log_path),
        "verification": verification,
    }


def _ordered_jobs() -> Iterable[tuple[str, int]]:
    for mode in MODES:
        for seed in SEEDS:
            yield mode, seed


def _run_gpu_queue(args, root: Path, python: str, output_root: Path,
                   gpu: str, jobs: list[tuple[str, int]]) -> list[dict]:
    return [
        _run_job(args, root, python, output_root, gpu, mode, seed)
        for mode, seed in jobs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints/sgg/manifests/pysgg_tde_motifs_vg_live.json",
    )
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--python")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=26446)
    parser.add_argument("--minimum_validation_objects", type=int, default=1000)
    parser.add_argument("--progress_interval_images", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else Path(_dataset_args(root, "vg")[1]).resolve()
    )
    if len(args.gpus) not in (1, 2) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus requires one or two distinct GPU identifiers")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("Sample counts must be positive")
    manifest = _load_manifest(args.manifest)
    python = args.python or manifest.get("environment_python") or sys.executable
    if not Path(python).expanduser().is_file():
        raise FileNotFoundError(
            f"Python recorded by the manifest does not exist: {python}. "
            "Pass --python with the configured main environment."
        )
    for path, label in ((args.data_root, "VG data root"),):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    contract = {
        "protocol": "paper_experiment5_tde_motifs_v1",
        "family": FAMILY,
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "fixed_hyperparameters": {
            "learning_rate": 3e-5,
            "object_weight": 2.25,
            "task_driven_object_focus": True,
            "task_object_weight": 3.0,
            "freeze_relation_parameters": True,
            "object_weight_delta_scale": 1.1,
            "object_bias_delta_scale": 0.5,
            "maximum_epochs": 5,
            "minimum_epochs": 3,
            "early_stopping_patience": 1,
            "max_mR_drop": 0.005,
        },
        "gpus": args.gpus,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.check_only:
        print(
            "[READY] Experiment V TDE-Motifs: "
            f"6 runs on {len(args.gpus)} GPU(s); manifest={args.manifest}"
        )
        return

    outcomes = []
    queues = {gpu: [] for gpu in args.gpus}
    for index, job in enumerate(_ordered_jobs()):
        queues[args.gpus[index % len(args.gpus)]].append(job)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue,
                args,
                root,
                python,
                args.output_dir,
                gpu,
                queues[gpu],
            )
            for gpu in args.gpus
        ]
        for future in futures:
            outcomes.extend(future.result())
    outcomes.sort(key=lambda row: (MODES.index(row["training_mode"]), row["seed"]))
    summary = {
        **contract,
        "status": "complete",
        "jobs": outcomes,
        "result_aggregate": _aggregate_results(outcomes),
    }
    destination = args.output_dir / "summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[complete] Experiment V TDE-Motifs matrix: {destination}")


if __name__ == "__main__":
    main()
