#!/usr/bin/env python3
"""Build a formal Experiment-IV manifest from a complete OpenPSG cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_SPECS = {
    "motifs": {
        "family": "Neural Motifs",
        "paradigm": "rnn_context",
        "checkpoint": "openpsg/psg/motifs/epoch_12.pth",
        "r50": 0.217,
        "mr50": 0.0957,
    },
    "vctree": {
        "family": "VCTree",
        "paradigm": "tree_context",
        "checkpoint": "openpsg/psg/vctree/epoch_12.pth",
        "r50": 0.221,
        "mr50": 0.102,
    },
    "psgtr": {
        "family": "PSGTR",
        "paradigm": "end_to_end_transformer",
        "checkpoint": "openpsg/psg/psgtr/epoch_60.pth",
        "r50": 0.344,
        "mr50": 0.208,
    },
    "psgformer": {
        "family": "PSGFormer",
        "paradigm": "end_to_end_transformer",
        "checkpoint": "openpsg/psg/psgformer/epoch_60.pth",
        "r50": 0.196,
        "mr50": 0.170,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--cache_root")
    parser.add_argument(
        "--native_report",
        help="Optional native-reference report; defaults to the canonical model directory.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    spec = MODEL_SPECS[args.model]
    cache = Path(
        args.cache_root
        or root / ("artifacts/prediction_cache/openpsg_" + args.model + "_psg")
    ).expanduser().resolve()
    checkpoint = (
        root / "checkpoints/sgg/weights" / spec["checkpoint"]
    ).resolve()
    source = (root / "external/official_repos/OpenPSG").resolve()
    output = Path(
        args.output
        or root / ("checkpoints/sgg/manifests/openpsg_" + args.model + "_psg.json")
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
        raise RuntimeError("OpenPSG cache validation is not ready")
    if metadata.get("images_by_task", {}).get("sgdet") != 2177:
        raise RuntimeError("Formal OpenPSG manifest requires all 2,177 PSG test images")
    checkpoint_digest = sha256(checkpoint)
    if metadata["checkpoint_sha256_by_task"].get("sgdet") != checkpoint_digest:
        raise RuntimeError("OpenPSG cache/checkpoint SHA256 mismatch")

    model_name = "openpsg_" + args.model + "_psg_official"
    if metadata.get("model_name") != model_name:
        raise RuntimeError("OpenPSG cache model name mismatch")
    export_state = json.loads(
        (cache / "export_state.json").read_text(encoding="utf-8")
    )
    native_report_path = Path(
        args.native_report
        or root / "artifacts/native_reference" / ("openpsg_" + args.model + "_psg")
        / "report.json"
    ).expanduser().resolve()
    manifest = {
        "name": model_name,
        "architecture_family": spec["family"],
        "paradigm": spec["paradigm"],
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
        "training_dataset": "PSG",
        # The paper numbers use panoptic-mask matching. The unified benchmark
        # uses box IoU, so reproduction is validated in a separate native report.
        "reference_dataset": "psg_official_panoptic",
        "reference_eval_images": 2177,
        "metric_scale": "fraction",
        "reproduction_tolerance": 0.02,
        "input_source": "official_prediction_cache",
        "training_seed": 0,
        "parameter_count": int(metadata["parameter_count"]),
        "baseline_mR": spec["mr50"],
        # ``psg`` is the executable unified box-IoU benchmark.  The protocol
        # label keeps the paper's panoptic-mask numbers out of direct box-IoU
        # reproduction checks while satisfying the manifest reference-domain
        # contract.
        "supported_datasets": ["psg", "psg_official_panoptic"],
        # The ontology is shared; only box-IoU versus panoptic-mask matching differs.
        "ontology_ids": {
            "psg": metadata["ontology_id"],
            "psg_official_panoptic": metadata["ontology_id"],
        },
        "perturbation_contract": {
            key: False
            for key in (
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
            "relation_logit_alignment": "sparse_predictions",
            "object_logit_alignment": "sparse_predictions",
        },
        "config": {
            "prediction_cache_root": str(cache),
            "prediction_cache_metadata_sha256": sha256(metadata_path),
            "relation_score_mode": "categorical",
            "unified_sgdet_matching": "box_iou_0.5",
            "official_reference_protocol": "panoptic_mask_iou_0.5",
            "prediction_protocol": export_state["prediction_protocol"],
            "object_score_source": export_state["object_score_source"],
        },
        "reference_metrics": {
            "SGDet/R@50": spec["r50"],
            "SGDet/mR@50": spec["mr50"],
        },
    }
    if native_report_path.is_file():
        native_report = json.loads(native_report_path.read_text(encoding="utf-8"))
        if native_report.get("status") != "pass":
            raise RuntimeError(f"Native OpenPSG reference did not pass: {native_report_path}")
        if native_report.get("model") != model_name:
            raise RuntimeError("Native OpenPSG report model mismatch")
        if native_report.get("checkpoint_sha256") != checkpoint_digest:
            raise RuntimeError("Native OpenPSG report checkpoint mismatch")
        if int(native_report.get("eval_images", 0)) != 2177:
            raise RuntimeError("Native OpenPSG report must evaluate all 2,177 images")
        manifest["native_reference_validation"] = {
            "status": "pass",
            "protocol": "official_panoptic_mask_iou_0.5",
            "report": str(native_report_path),
            "sha256": sha256(native_report_path),
            "config_source": native_report.get("config_source"),
            "config_sha256": native_report.get("config_sha256"),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("manifest=" + str(output))


if __name__ == "__main__":
    main()
