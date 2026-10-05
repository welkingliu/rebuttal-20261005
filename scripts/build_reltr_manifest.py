#!/usr/bin/env python3
"""Build a provenance-complete manifest for the official RelTR VG checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.models.adapters.reltr import create_adapter


PERTURBATION_CONTRACT = {
    "full": False,
    "noise": False,
    "swap": False,
    "union_zero": False,
    "boxes_only": False,
    "visual_noise": False,
    "color_jitter": True,
    "union_attenuation": False,
    "on_manifold_replacement": False,
    "random_node_mask": False,
    "key_node_mask": False,
    "unrelated_node_mask": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--environment_python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference_r50", type=float, default=0.275)
    parser.add_argument("--reference_mr50", type=float, default=0.108)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    source_root = root / "external" / "official_repos" / "RelTR"
    checkpoint = root / "checkpoints" / "sgg" / "weights" / "reltr" / "vg" / "checkpoint0149.pth"
    seen_path = root / "artifacts" / "manifests" / "seen_triplets_full.json"
    for path in (source_root, checkpoint, seen_path):
        if not path.exists():
            raise FileNotFoundError(path)

    marker = json.loads((source_root / ".official_source.json").read_text(encoding="utf-8"))
    seen = json.loads(seen_path.read_text(encoding="utf-8"))
    ontology_id = seen["_metadata"]["vg"]["ontology_id"]

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    adapter = create_adapter(
        checkpoint=str(checkpoint),
        checkpoints={"sgdet": str(checkpoint)},
        device=args.device,
        config={"dataset": "vg"},
        diagnostic_task="sgdet",
    )
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters())

    payload = {
        "name": "reltr_vg_official",
        "architecture_family": "RelTR",
        "paradigm": "end_to_end_transformer",
        "factory": "sgg_core.models.adapters.reltr:create_adapter",
        "environment_python": str(Path(args.environment_python).expanduser().resolve()),
        "checkpoints": {
            "sgdet": {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
        },
        "supported_tasks": ["sgdet"],
        "diagnostic_task": "sgdet",
        "source_url": marker["repository_url"],
        "source_root": str(source_root),
        "source_commit": marker["commit"],
        "training_dataset": "VG-150",
        "reference_dataset": "vg",
        "reference_eval_images": 26446,
        "metric_scale": "fraction",
        "reproduction_tolerance": 0.02,
        "input_source": "raw_images",
        "training_seed": 42,
        "parameter_count": parameter_count,
        "baseline_mR": float(args.reference_mr50),
        "supported_datasets": ["vg"],
        "ontology_ids": {"vg": ontology_id},
        "perturbation_contract": PERTURBATION_CONTRACT,
        "mitigation_contract": {
            "forward_grounding": False,
            "trainable_grounding_parameters": False,
            "object_logits": True,
            "trainable_object_parameters": False,
            "mask_object_logits": False,
            "relation_logit_alignment": "sparse_queries",
            "object_logit_alignment": "sparse_queries",
        },
        "config": {"dataset": "vg"},
        "reference_metrics": {
            "SGDet/R@50": float(args.reference_r50),
            "SGDet/mR@50": float(args.reference_mr50),
        },
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else root / "checkpoints" / "sgg" / "manifests" / "reltr_vg.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={output}")
    print(f"parameter_count={parameter_count}")
    print(f"ontology_id={ontology_id}")


if __name__ == "__main__":
    main()
