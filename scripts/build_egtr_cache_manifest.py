#!/usr/bin/env python3
"""Build a formal EGTR VG/OI manifest from a validated cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", choices=("vg", "oi"), default="vg")
    parser.add_argument("--cache_root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source_root")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    dataset = args.dataset.lower()
    cache = Path(
        args.cache_root or root / f"artifacts/prediction_cache/egtr_{dataset}"
    ).expanduser().resolve()
    checkpoint = Path(
        args.checkpoint
        or root / f"checkpoints/sgg/weights/egtr/{dataset}/runtime/model.ckpt"
    ).expanduser().resolve()
    source = Path(
        args.source_root or root / "external/official_repos/egtr"
    ).expanduser().resolve()
    output = Path(
        args.output or root / f"checkpoints/sgg/manifests/egtr_{dataset}.json"
    ).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    validation_path = cache / "validation_report.json"
    for path in (checkpoint, source, metadata_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    marker = json.loads((source / ".official_source.json").read_text(encoding="utf-8"))
    if validation.get("status") != "ready":
        raise RuntimeError(f"EGTR cache is not ready: {validation.get('failures')}")
    expected_images = 26446 if dataset == "vg" else 1813
    if metadata.get("images_by_task", {}).get("sgdet") != expected_images:
        raise RuntimeError(
            f"Formal EGTR-{dataset} manifest requires all {expected_images:,} images"
        )
    if metadata.get("dataset") != dataset:
        raise RuntimeError(
            f"EGTR cache dataset mismatch: {metadata.get('dataset')} != {dataset}"
        )
    checkpoint_digest = sha256(checkpoint)
    if metadata["checkpoint_sha256_by_task"].get("sgdet") != checkpoint_digest:
        raise RuntimeError("EGTR cache/checkpoint SHA256 mismatch")

    reference_metrics = (
        {"SGDet/R@50": 0.302, "SGDet/imR@50": 0.079}
        if dataset == "vg"
        else {"SGDet/R@50": 0.7915673176482114,
              "SGDet/mR@50": 0.3999715109055298}
    )
    manifest = {
        "name": f"egtr_{dataset}_official",
        "architecture_family": "EGTR",
        "paradigm": "end_to_end_transformer",
        "execution_mode": "prediction_cache",
        "factory": "sgg_core.models.prediction_cache:create_adapter",
        "environment_python": "python3",
        "checkpoints": {
            "sgdet": {"path": str(checkpoint), "sha256": checkpoint_digest}
        },
        "supported_tasks": ["sgdet"],
        "diagnostic_task": "sgdet",
        "source_url": marker["repository_url"],
        "source_root": str(source),
        "source_commit": marker["commit"],
        "training_dataset": "VG-150" if dataset == "vg" else "Open Images V6",
        "reference_dataset": dataset,
        "reference_eval_images": expected_images,
        "metric_scale": "fraction",
        "reproduction_tolerance": 0.02 if dataset == "vg" else 0.05,
        "input_source": "official_prediction_cache",
        "training_seed": 42,
        "parameter_count": int(metadata["parameter_count"]),
        "baseline_mR": 0.079 if dataset == "vg" else 0.3999715109055298,
        "supported_datasets": [dataset],
        "ontology_ids": {dataset: metadata["ontology_id"]},
        "perturbation_contract": {
            key: False for key in (
                "full", "noise", "swap", "union_zero", "boxes_only",
                "visual_noise", "color_jitter", "union_attenuation",
                "on_manifold_replacement", "random_node_mask",
                "key_node_mask", "unrelated_node_mask",
            )
        },
        "mitigation_contract": {
            "forward_grounding": False,
            "trainable_grounding_parameters": False,
            "object_logits": False,
            "trainable_object_parameters": False,
            "mask_object_logits": False,
            "relation_logit_alignment": "sparse_queries",
            "object_logit_alignment": "sparse_queries",
        },
        "config": {
            "prediction_cache_root": str(cache),
            "prediction_cache_metadata_sha256": sha256(metadata_path),
            "relation_score_mode": "independent_probabilities",
        },
        "reference_metric_policies": (
            {
                "SGDet/R@50": "required",
                # The EGTR archive reports OI ``(single)mR``.  The unified
                # benchmark computes class-mean mR on the public validation
                # ontology, so retain the native value without gating on it.
                "SGDet/mR@50": "report_only_protocol_mismatch",
            }
            if dataset == "oi" else {}
        ),
        "reference_metrics": reference_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
