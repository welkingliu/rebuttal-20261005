#!/usr/bin/env python3
"""Run the compute-bounded Experiment-II or Experiment-III model matrix."""

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
    ALL_DATASETS,
    EXPERIMENT_2_DATASETS,
    EXPERIMENT_2_FULL_DATASET,
    EXPERIMENT_2_FULL_LEVELS,
    EXPERIMENT_2_LIGHT_LEVELS,
    EXPERIMENT_2_SWEEP_STRATEGIES,
    EXPERIMENT_3_DATASETS,
    DIAGNOSTIC_MODEL_FAMILIES,
    DIAGNOSTIC_MODEL_RANGE,
    format_dataset_targets,
    parse_dataset_targets,
)


DEFAULT_FAMILIES = DIAGNOSTIC_MODEL_FAMILIES


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
                data / "psg" / "psg_val_test.json",
                data / "psg" / "psg.json",
            )),
            "--image_root", str(data / "coco"),
            "--panoptic_root", str(data / "coco"),
        ]
    return ["--data_root", str(data / "vrd")]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _read_manifests(manifest_dir: Path) -> list[dict]:
    records = []
    names = set()
    for path in sorted(manifest_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload.get("name", "")).strip()
        family = str(payload.get("architecture_family", "")).strip()
        supported = tuple(dict.fromkeys(
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
            "training_seed": int(payload.get("training_seed", 2**31 - 1)),
            "execution_mode": payload.get("execution_mode", "live_adapter"),
            "supported_tasks": sorted({
                str(task).lower() for task in payload.get("supported_tasks", [])
            }),
            "perturbation_contract": payload.get("perturbation_contract", {}),
            "diagnostic_contract": payload.get("diagnostic_contract", {}),
        })
    if not records:
        raise FileNotFoundError(f"No manifests in {manifest_dir}")
    return records


def _capability_gaps(record: dict, experiment: int,
                     analysis_scope: str = "both") -> list[str]:
    gaps = []
    diagnostic = record["diagnostic_contract"]
    perturbation = record["perturbation_contract"]
    if experiment == 2:
        if analysis_scope == "observational":
            if not ({"sgcls", "sgdet"} & set(record["supported_tasks"])):
                gaps.append("supported_tasks.sgcls_or_sgdet")
            return gaps
        if record["execution_mode"] != "live_adapter":
            gaps.append("live_adapter")
        if diagnostic.get("gt_pair_predict") is not True:
            gaps.append("diagnostic_contract.gt_pair_predict")
        required = {
            "full", "visual_noise", "union_attenuation",
            "on_manifold_replacement", "random_node_mask",
            "key_node_mask", "unrelated_node_mask", "color_jitter",
        }
        gaps.extend(
            f"perturbation_contract.{name}"
            for name in sorted(required)
            if perturbation.get(name) is not True
        )
    else:
        if record["execution_mode"] != "live_adapter":
            gaps.append("live_adapter")
        for name in ("gt_node_features", "graph_intervention"):
            if diagnostic.get(name) is not True:
                gaps.append(f"diagnostic_contract.{name}")
    return gaps


def _select_for_dataset(records: list[dict], families: list[str], dataset: str,
                        experiment: int, analysis_scope: str = "both") -> list[dict]:
    selected = []
    for family in families:
        family_records = [
            record for record in records
            if record["family"] == family and dataset in record["datasets"]
        ]
        candidates = sorted(
            (
                record for record in family_records
                if not _capability_gaps(record, experiment, analysis_scope)
            ),
            key=lambda record: (record["training_seed"] != 17, record["training_seed"], record["name"]),
        )
        if not candidates:
            details = {
                record["name"]: _capability_gaps(record, experiment, analysis_scope)
                for record in family_records
            }
            raise RuntimeError(
                f"No Experiment-{experiment} capable manifest for "
                f"family={family!r}, dataset={dataset!r}; candidates={details or 'none'}"
            )
        selected.append(candidates[0])
    return selected


def _write_contract(output: Path, args, selected: list[dict],
                    targets: dict[str, int], gaps: dict[str, dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment,
        "analysis_scope": args.analysis_scope if args.experiment == 2 else None,
        "datasets": args.datasets,
        "selected_models": selected,
        "dataset_model_targets": targets,
        "coverage_gaps": gaps,
        "full_perturbation_dataset": (
            args.full_dataset if args.experiment == 2 else None
        ),
        "full_levels": args.full_levels if args.experiment == 2 else None,
        "light_levels": args.light_levels if args.experiment == 2 else None,
        "perturbation_strategies": (
            args.perturbation_strategies if args.experiment == 2 else None
        ),
        "skip_pair_audit": (
            args.skip_pair_audit if args.experiment == 2 else None
        ),
        "skip_physical_consistency": (
            args.skip_physical_consistency if args.experiment == 2 else None
        ),
    }
    (output / "matrix_contract.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _run_queue(gpu: str, jobs: list[dict], args, root: Path,
               output: Path) -> list[dict]:
    outcomes = []
    for job in jobs:
        record = job["record"]
        dataset = job["dataset"]
        model_name = _safe_name(record["name"])
        model_output = output / dataset / model_name
        log_path = output / "logs" / f"{dataset}_{model_name}.log"
        result_path = model_output / f"experiment_{args.experiment}.json"
        if args.resume and result_path.is_file():
            outcomes.append({
                "dataset": dataset,
                "model": record["name"],
                "family": record["family"],
                "gpu": str(gpu),
                "returncode": 0,
                "output": str(model_output),
                "log": str(log_path),
                "resumed": True,
            })
            print(
                f"[gpu={gpu}] resume-skip experiment={args.experiment} "
                f"dataset={dataset} model={record['name']}",
                flush=True,
            )
            continue
        command = [
            record["python"], "-m", f"sgg_core.experiments.experiment_{args.experiment}",
            "--dataset", dataset,
            "--official_manifest", str(record["path"]),
            "--output_dir", str(model_output),
            "--train_samples", str(args.train_samples),
            "--eval_samples", str(args.eval_samples),
            "--device", args.device,
            *_dataset_args(root, dataset),
        ]
        if args.experiment == 2:
            levels = args.full_levels if dataset == args.full_dataset else args.light_levels
            command.extend([
                "--analysis_scope", args.analysis_scope,
                "--perturbation_levels", *(str(level) for level in levels),
                "--perturbation_strategies", *args.perturbation_strategies,
            ])
            if args.skip_pair_audit:
                command.append("--skip_pair_audit")
            if args.skip_physical_consistency:
                command.append("--skip_physical_consistency")
        python_path = Path(record["python"]).expanduser()
        if not python_path.is_file():
            raise FileNotFoundError(
                f"environment_python for {record['name']} not found: {python_path}"
            )
        model_output.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if args.device.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONPATH"] = str(root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        print(
            f"[gpu={gpu}] experiment={args.experiment} "
            f"dataset={dataset} model={record['name']}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        outcome = {
            "dataset": dataset,
            "model": record["name"],
            "family": record["family"],
            "gpu": str(gpu),
            "returncode": completed.returncode,
            "output": str(model_output),
            "log": str(log_path),
        }
        outcomes.append(outcome)
        if completed.returncode:
            raise RuntimeError(
                f"Diagnostic shard failed: experiment={args.experiment} "
                f"{record['name']}/{dataset}; see {log_path}"
            )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(2, 3), required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda",
        help="Execution device passed to each diagnostic shard. Prediction-cache "
             "observational runs can use CPU.",
    )
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--analysis_scope", choices=("observational", "interventional", "both"),
        default="both",
        help="Experiment II only; observational accepts formal prediction caches.",
    )
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--minimum_families", type=int, default=DIAGNOSTIC_MODEL_RANGE[0])
    parser.add_argument("--maximum_families", type=int, default=DIAGNOSTIC_MODEL_RANGE[1])
    parser.add_argument("--dataset_model_targets", nargs="*")
    parser.add_argument("--full_dataset", choices=ALL_DATASETS, default=EXPERIMENT_2_FULL_DATASET)
    parser.add_argument("--full_levels", nargs="+", type=float, default=list(EXPERIMENT_2_FULL_LEVELS))
    parser.add_argument("--light_levels", nargs="+", type=float, default=list(EXPERIMENT_2_LIGHT_LEVELS))
    parser.add_argument(
        "--perturbation_strategies", nargs="+",
        default=list(EXPERIMENT_2_SWEEP_STRATEGIES),
    )
    parser.add_argument("--skip_pair_audit", action="store_true")
    parser.add_argument("--skip_physical_consistency", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve()
        if args.manifest_dir else root / "checkpoints" / "sgg" / "manifests"
    )
    args.datasets = list(dict.fromkeys(
        args.datasets or (
            EXPERIMENT_2_DATASETS if args.experiment == 2 else EXPERIMENT_3_DATASETS
        )
    ))
    args.families = list(dict.fromkeys(args.families))
    if not args.minimum_families <= len(args.families) <= args.maximum_families:
        raise ValueError(
            "Representative panel must contain between "
            f"{args.minimum_families} and {args.maximum_families} families; "
            f"received={len(args.families)}"
        )
    defaults = {
        dataset: (args.minimum_families if dataset == "vg" else 1)
        for dataset in args.datasets
    }
    targets = parse_dataset_targets(args.dataset_model_targets, defaults)
    unknown_targets = sorted(set(targets) - set(args.datasets))
    if unknown_targets:
        raise ValueError(f"Targets supplied for datasets not being run: {unknown_targets}")

    records = _read_manifests(manifest_dir)
    coverage = {
        dataset: _select_for_dataset(
            records, args.families, dataset, args.experiment, args.analysis_scope
        )
        for dataset in args.datasets
    }
    selected = list({
        record["path"]: record
        for records_for_dataset in coverage.values()
        for record in records_for_dataset
    }.values())
    gaps = {
        dataset: {
            "available": len(coverage[dataset]),
            "required": int(targets.get(dataset, 0)),
        }
        for dataset in args.datasets
        if len(coverage[dataset]) < int(targets.get(dataset, 0))
    }
    _write_contract(output, args, selected, targets, gaps)
    if gaps:
        raise RuntimeError(
            "Diagnostic matrix lacks ontology-declared model coverage: "
            f"{gaps}. Add real compatible manifests; do not relabel a VG head."
        )
    if args.check_only:
        print(
            f"[READY] Experiment {args.experiment}: "
            f"datasets={args.datasets} families={args.families} jobs="
            f"{sum(len(items) for items in coverage.values())}"
        )
        return

    jobs = [
        {"record": record, "dataset": dataset}
        for dataset in args.datasets for record in coverage[dataset]
    ]
    queues = [jobs[index::len(args.gpus)] for index in range(len(args.gpus))]
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(_run_queue, gpu, queue, args, root, output)
            for gpu, queue in zip(args.gpus, queues)
        ]
        for future in futures:
            outcomes.extend(future.result())
    summary = {
        "experiment": args.experiment,
        "analysis_scope": args.analysis_scope if args.experiment == 2 else None,
        "status": "complete",
        "datasets": args.datasets,
        "families": args.families,
        "dataset_model_targets": targets,
        "jobs": outcomes,
        "matrix_contract": str(output / "matrix_contract.json"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Experiment {args.experiment} diagnostic matrix complete: {output}")


if __name__ == "__main__":
    main()
