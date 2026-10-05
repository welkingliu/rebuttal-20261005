#!/usr/bin/env python3
"""Create a strict official-model manifest from a validated prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PERTURBATIONS = (
    "full", "noise", "swap", "union_zero", "boxes_only", "visual_noise",
    "color_jitter", "union_attenuation", "on_manifold_replacement",
    "random_node_mask", "key_node_mask", "unrelated_node_mask",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key_values(values, separator="="):
    result = {}
    for value in values:
        key, token, raw = value.partition(separator)
        if not token or not key or not raw:
            raise ValueError(f"Expected key{separator}value, got {value!r}")
        result[key] = raw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--paradigm", required=True)
    parser.add_argument("--source_url", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--training_dataset", required=True)
    parser.add_argument("--reference_dataset", required=True)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="task=/absolute/path/to/checkpoint")
    parser.add_argument("--reference_metric", action="append", default=[],
                        help="Task/metric=value, for example SGDet/R@50=0.301")
    parser.add_argument("--metric_scale", choices=("fraction", "percent"), default="fraction")
    parser.add_argument("--reproduction_tolerance", type=float, default=0.02)
    parser.add_argument("--reference_eval_images", type=int)
    parser.add_argument("--training_seed", type=int, default=17)
    parser.add_argument("--baseline_mr", type=float, default=0.0)
    parser.add_argument(
        "--relation_score_mode",
        choices=("categorical", "independent_probabilities"),
        default="categorical",
        help="Use independent_probabilities for multi-label sigmoid relation heads.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_values = parse_key_values(args.checkpoint)
    tasks = [str(value).lower() for value in metadata.get("tasks", [])]
    if set(checkpoint_values) != set(tasks):
        raise ValueError(
            f"Checkpoint tasks must match cache tasks: {sorted(checkpoint_values)} != {sorted(tasks)}"
        )
    checkpoints = {}
    for task, value in checkpoint_values.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[task] = {"path": str(path), "sha256": sha256(path)}
    references = {
        key: float(value) for key, value in parse_key_values(args.reference_metric).items()
    }
    if not references:
        raise ValueError("At least one --reference_metric is required")
    manifest = {
        "name": args.name,
        "architecture_family": args.family,
        "paradigm": args.paradigm,
        "execution_mode": "prediction_cache",
        "factory": "sgg_core.models.prediction_cache:create_adapter",
        "environment_python": sys.executable,
        "checkpoints": checkpoints,
        "supported_tasks": tasks,
        "diagnostic_task": "sgdet" if "sgdet" in tasks else tasks[0],
        "source_url": args.source_url,
        "source_root": str(Path(args.source_root).expanduser().resolve()),
        "source_commit": args.source_commit,
        "training_dataset": args.training_dataset,
        "reference_dataset": args.reference_dataset.lower(),
        "metric_scale": args.metric_scale,
        "reproduction_tolerance": args.reproduction_tolerance,
        "input_source": "official_prediction_cache",
        "training_seed": args.training_seed,
        "parameter_count": int(metadata["parameter_count"]),
        "parameter_count_by_task": {
            str(task): int(value) for task, value in
            metadata.get("parameter_count_by_task", {}).items()
        },
        "baseline_mR": args.baseline_mr,
        "supported_datasets": [str(metadata["dataset"]).lower()],
        "ontology_ids": {
            str(metadata["dataset"]).lower(): str(metadata["ontology_id"]),
        },
        "perturbation_contract": {key: False for key in PERTURBATIONS},
        "mitigation_contract": {
            "forward_grounding": False,
            "trainable_grounding_parameters": False,
            "object_logits": False,
            "trainable_object_parameters": False,
            "mask_object_logits": False,
            "relation_logit_alignment": "unavailable",
            "object_logit_alignment": "unavailable",
        },
        "config": {
            "prediction_cache_root": str(cache_root),
            "prediction_cache_metadata_sha256": sha256(metadata_path),
            "relation_score_mode": args.relation_score_mode,
        },
        "reference_metrics": references,
    }
    if args.reference_eval_images is not None:
        if args.reference_eval_images <= 0:
            raise ValueError("--reference_eval_images must be positive")
        manifest["reference_eval_images"] = int(args.reference_eval_images)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prediction-cache manifest: {output}")


if __name__ == "__main__":
    main()
