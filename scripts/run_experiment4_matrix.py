#!/usr/bin/env python3
"""Run official-model shards in isolated environments and aggregate them."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.submission_protocol import (
    EXTERNAL_DATASET_MODEL_TARGETS,
    EXPERIMENT_4_STEPS,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_BENCHMARK_DATASETS,
    STANDARD_DATASET_FAMILY_TARGETS,
    STANDARD_TASK_FAMILY_TARGETS,
    format_dataset_targets,
    parse_dataset_targets,
)


DATASETS = ("vg", "oi", "gqa", "psg", "vrd")


def _first(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def _dataset_args(root: Path, dataset: str) -> list[str]:
    data = root / "data"
    if dataset == "vg":
        return ["--vg_root", str(_first(data / "vg" / "v1.4", data / "vg"))]
    if dataset == "oi":
        return ["--oi_root", str(data / "openimages" / "open-images-v6")]
    if dataset == "gqa":
        train = _first(
            data / "gqa" / "train_sceneGraphs.json",
            data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
        )
        val = _first(
            data / "gqa" / "val_sceneGraphs.json",
            data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
        )
        return [
            "--gqa_train_ann", str(train),
            "--gqa_eval_ann", str(val),
            "--gqa_image_root", str(data / "gqa" / "images"),
        ]
    if dataset == "psg":
        return [
            "--psg_train_ann", str(data / "psg" / "psg_train_val.json"),
            "--psg_eval_ann", str(_first(
                data / "psg" / "psg_val_test.json",
                data / "psg" / "psg.json",
            )),
            "--psg_image_root", str(data / "coco"),
            "--psg_panoptic_root", str(data / "coco"),
        ]
    return ["--vrd_root", str(data / "vrd")]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _read_manifests(manifest_dir: Path) -> list[dict]:
    records = []
    names = set()
    for path in sorted(manifest_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload.get("name", "")).strip()
        family = str(payload.get("architecture_family", "")).strip()
        supported = list(dict.fromkeys(
            str(value).lower() for value in payload.get("supported_datasets", [])
        ))
        if not name or not family or not supported:
            raise ValueError(f"Incomplete model identity in {path}")
        if name in names:
            raise ValueError(f"Duplicate model name across manifests: {name}")
        names.add(name)
        records.append({
            "path": path.resolve(),
            "name": name,
            "family": family,
            "datasets": supported,
            "python": str(payload.get("environment_python") or sys.executable),
            "tasks": sorted({
                str(task).lower() for task in payload.get("supported_tasks", [])
            }),
        })
    if not records:
        raise FileNotFoundError(f"No manifests in {manifest_dir}")
    return records


def _validate_matrix(records: list[dict], datasets: list[str], minimum_families: int,
                     dataset_targets: dict[str, int], task_contract: str) -> None:
    families = {record["family"] for record in records}
    if len(families) < minimum_families:
        raise RuntimeError(f"Manifest matrix has {len(families)}/{minimum_families} families")
    for dataset in datasets:
        minimum = int(dataset_targets.get(dataset, 0))
        native = {
            record["family"] for record in records if dataset in record["datasets"]
        }
        if len(native) < minimum:
            raise RuntimeError(
                f"Manifest matrix has {len(native)}/{minimum} families for {dataset}"
            )
        if task_contract == "sgdet_only":
            required_tasks = {"sgdet": int(dataset_targets.get(dataset, 0))}
        elif task_contract == "tritask_depth":
            required_tasks = {
                task: int(dataset_targets.get(dataset, 0))
                for task in ("predcls", "sgcls", "sgdet")
            }
        else:
            required_tasks = STANDARD_TASK_FAMILY_TARGETS.get(dataset, {})
        for task, target in required_tasks.items():
            task_families = {
                record["family"] for record in records
                if dataset in record["datasets"] and task in record["tasks"]
            }
            if len(task_families) < target:
                raise RuntimeError(
                    f"Manifest matrix has {len(task_families)}/{target} "
                    f"{dataset}-{task} families"
                )


def _run_gpu_queue(gpu: str, jobs: list[dict], args, project_root: Path,
                   shard_root: Path, log_root: Path) -> list[dict]:
    outcomes = []
    for job in jobs:
        model_dir = _safe_name(job["record"]["name"])
        output_dir = shard_root / model_dir / job["dataset"]
        log_path = log_root / f"{model_dir}_{job['dataset']}.log"
        summary_path = output_dir / "summary.json"
        result_path = output_dir / job["dataset"] / "results.json"
        if args.resume and summary_path.is_file() and result_path.is_file():
            print(
                f"[gpu={gpu}] resume-skip {job['record']['name']} / "
                f"{job['dataset']} -> {summary_path}",
                flush=True,
            )
            outcomes.append({
                "model": job["record"]["name"],
                "family": job["record"]["family"],
                "dataset": job["dataset"],
                "gpu": str(gpu),
                "returncode": 0,
                "log": str(log_path),
                "resumed": True,
            })
            continue
        command = [
            job["record"]["python"], "-m", "sgg_core.experiments.experiment_4",
            "--datasets", job["dataset"],
            "--official_manifest", str(job["record"]["path"]),
            "--model_panel", str(project_root / "sgg_core" / "models" / "model_panel.json"),
            "--output_dir", str(output_dir),
            "--minimum_model_families", "1",
            "--minimum_models_per_dataset", "1",
            "--steps", *args.steps,
            "--seen_triplets_manifest", str(args.seen_triplets_manifest),
            "--train_samples", str(args.train_samples),
            "--eval_samples", str(args.eval_samples),
            "--device", "cuda",
            *_dataset_args(project_root, job["dataset"]),
        ]
        python_path = Path(job["record"]["python"]).expanduser()
        if not python_path.is_file():
            raise FileNotFoundError(
                f"environment_python for {job['record']['name']} not found: {python_path}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(project_root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        print(f"[gpu={gpu}] {job['record']['name']} / {job['dataset']} -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=project_root, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        outcome = {
            "model": job["record"]["name"],
            "family": job["record"]["family"],
            "dataset": job["dataset"],
            "gpu": str(gpu),
            "returncode": completed.returncode,
            "log": str(log_path),
        }
        outcomes.append(outcome)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Shard failed: {job['record']['name']}/{job['dataset']}; see {log_path}"
            )
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seen_triplets_manifest")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS,
        default=list(STANDARD_BENCHMARK_DATASETS),
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip only shards that already contain both summary.json and results.json.",
    )
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument(
        "--eval_samples", type=int, default=1_000_000_000,
        help="Formal standard metrics use the complete official evaluation split.",
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
        help=(
            "Validation contract for the model matrix. 'full' enforces the "
            "submission PredCls/SGCls/SGDet coverage; 'sgdet_only' creates an "
            "explicitly scoped SGDet panel; 'tritask_depth' requires every "
            "selected family to provide all three VG tasks."
        ),
    )
    parser.add_argument(
        "--steps", nargs="+", choices=("standard", "feature", "pair", "graph", "grounding"),
        default=list(EXPERIMENT_4_STEPS),
    )
    parser.add_argument(
        "--experiment2_root",
        help="Completed Experiment-II root whose perturbation JSON is reused.",
    )
    parser.add_argument(
        "--experiment3_root",
        help="Completed Experiment-III root whose motif JSON is reused.",
    )
    parser.add_argument(
        "--external_diagnostic_targets", nargs="*",
        default=format_dataset_targets(EXTERNAL_DATASET_MODEL_TARGETS),
    )
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    if args.seen_triplets_manifest:
        args.seen_triplets_manifest = str(
            Path(args.seen_triplets_manifest).expanduser().resolve()
        )
    else:
        args.seen_triplets_manifest = str(
            root / "artifacts" / "manifests" / "seen_triplets_full.json"
        )
    if not Path(args.seen_triplets_manifest).is_file():
        raise FileNotFoundError(
            "Build the full seen-triplet manifest before Experiment IV: "
            f"{args.seen_triplets_manifest}"
        )
    output = Path(args.output_dir).expanduser().resolve()
    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir
        else root / "checkpoints" / "sgg" / "manifests"
    )
    datasets = list(dict.fromkeys(args.datasets))
    targets = parse_dataset_targets(
        args.dataset_family_targets,
        {
            dataset: STANDARD_DATASET_FAMILY_TARGETS.get(dataset, 0)
            for dataset in datasets
        },
    )
    unknown_targets = sorted(set(targets) - set(datasets))
    if unknown_targets:
        raise ValueError(
            f"Dataset targets were supplied for datasets not being run: {unknown_targets}"
        )
    records = _read_manifests(manifest_dir)
    _validate_matrix(
        records, datasets, args.minimum_model_families,
        targets, args.task_contract,
    )
    jobs = [
        {"record": record, "dataset": dataset}
        for record in records for dataset in datasets
        if dataset in record["datasets"]
    ]
    shard_root = output / "shards"
    log_root = output / "logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    queues = [jobs[index::len(args.gpus)] for index in range(len(args.gpus))]
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue, gpu, queue, args, root, shard_root, log_root
            )
            for gpu, queue in zip(args.gpus, queues)
        ]
        for future in futures:
            outcomes.extend(future.result())

    aggregate_path = output / "summary.json"
    aggregate_command = [
        sys.executable, "-m", "sgg_core.tools.aggregate_model_shards",
        "--shard_root", str(shard_root),
        "--output", str(aggregate_path),
        "--datasets", *datasets,
        "--minimum_model_families", str(args.minimum_model_families),
        "--dataset_family_targets", *format_dataset_targets(targets),
        "--task_contract", args.task_contract,
    ]
    if args.experiment2_root:
        aggregate_command.extend(["--experiment2_root", args.experiment2_root])
    if args.experiment3_root:
        aggregate_command.extend(["--experiment3_root", args.experiment3_root])
    if args.experiment2_root or args.experiment3_root:
        aggregate_command.extend([
            "--external_diagnostic_targets", *args.external_diagnostic_targets,
        ])
    subprocess.run(aggregate_command, cwd=root, check=True)
    (output / "job_outcomes.json").write_text(
        json.dumps(outcomes, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Experiment IV model matrix complete: {aggregate_path}")


if __name__ == "__main__":
    main()
