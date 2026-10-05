#!/usr/bin/env python3
"""Register the official Causal Motifs-SUM checkpoints as live TDE-Motifs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


CHECKPOINTS = {
    "predcls": "predcls/model_0030000.pth",
    "sgcls": "sgcls/model_final.pth",
    "sgdet": "sgdet/model_0028000.pth",
}
REFERENCE_METRICS = {
    "PredCls/R@50": 0.4588,
    "PredCls/mR@50": 0.2475,
    "SGCls/R@50": 0.2631,
    "SGCls/mR@50": 0.1321,
    "SGDet/R@50": 0.1656,
    "SGDet/mR@50": 0.0894,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--main_python")
    parser.add_argument("--worker_python")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    source = root / "external/official_repos/PySGG"
    marker_path = source / ".official_source.json"
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Missing pinned PySGG source marker: {marker_path}. "
            "Run scripts/prepare_official_models.py first."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    weights = root / "checkpoints/sgg/weights/causal_motifs_sum/vg"
    checkpoints = {}
    for task, relative in CHECKPOINTS.items():
        path = weights / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing official Causal Motifs {task} checkpoint: {path}"
            )
        checkpoints[task] = {"path": str(path), "sha256": _sha256(path)}

    config = root / "configs/pysgg_vg_tritask/tde_motifs_sgcls.yaml"
    worker = root / "scripts/pysgg_live_worker.py"
    for path in (config, worker):
        if not path.is_file():
            raise FileNotFoundError(path)
    main_python = str(
        Path(
            args.main_python
            or os.environ.get("SGG_PYTHON")
            or sys.executable
        ).expanduser().resolve()
    )
    worker_python = str(
        Path(
            args.worker_python
            or os.environ.get("PYSGG_PYTHON")
            or sys.executable
        ).expanduser().resolve()
    )
    manifest = {
        "name": "pysgg_tde_motifs_vg_live",
        "architecture_family": "TDE-Motifs",
        "paradigm": "two_stage_causal_context",
        "execution_mode": "live_adapter",
        "factory": "sgg_core.models.adapters.pysgg_live:create_adapter",
        "environment_python": main_python,
        "checkpoints": checkpoints,
        "supported_tasks": ["predcls", "sgcls", "sgdet"],
        "diagnostic_task": "sgcls",
        "source_url": marker["repository_url"],
        "source_root": str(source),
        "source_commit": marker["commit"],
        "training_dataset": "VG-150",
        "reference_dataset": "vg",
        "metric_scale": "fraction",
        "reproduction_tolerance": 0.02,
        "input_source": "raw_images",
        "training_seed": 17,
        "parameter_count": 369_959_784,
        "baseline_mR": REFERENCE_METRICS["SGDet/mR@50"],
        "supported_datasets": ["vg"],
        "ontology_ids": {"vg": "vg150:27af736e1c912d3f"},
        "perturbation_contract": {
            key: True for key in (
                "full",
                "noise",
                "swap",
                "union_zero",
                "boxes_only",
                "visual_noise",
                "color_jitter",
                "union_attenuation",
                "on_manifold_replacement",
                "random_node_mask",
                "key_node_mask",
                "unrelated_node_mask",
            )
        },
        "diagnostic_contract": {
            "gt_pair_predict": True,
            "gt_node_features": False,
            "graph_intervention": True,
            "graph_intervention_space": "raw_image_gt_box_conditioned",
            "consumed_input_fingerprint": True,
        },
        "mitigation_contract": {
            "forward_grounding": True,
            "trainable_grounding_parameters": True,
            "object_logits": True,
            "trainable_object_parameters": True,
            "mask_object_logits": False,
            "object_score_semantics": "full_refined_logits",
            "relation_logit_alignment": "gt_relations",
            "object_logit_alignment": "gt_entities",
            "method": "shared_identity_initialised_logit_calibrator",
        },
        "config": {
            "source_root": str(source),
            "worker_python": worker_python,
            "worker_script": str(worker),
            "diagnostic_config": str(config),
            "prediction_cache_root": str(
                root
                / "artifacts/prediction_cache/"
                "causal_motifs_sum_tde_live_validation_stub"
            ),
            "test_prediction_cache_root": str(
                root / "artifacts/prediction_cache/causal_motifs_sum_tde_vg"
            ),
            "official_parameter_count": 369_934_180,
            "checkpoint_tensor_parameter_count": 367_174_438,
            "parameter_count_semantics": (
                "runtime_constructed_model_plus_shared_calibrators"
            ),
            "relation_score_mode": "categorical",
            "mitigation_scope": (
                "post_hoc_calibration_shared_across_predcls_sgcls_sgdet"
            ),
        },
        "reference_metrics": REFERENCE_METRICS,
        "reference_eval_images": 26_446,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root / "checkpoints/sgg/manifests/pysgg_tde_motifs_vg_live.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={output}")
    print(
        "checkpoint_sha256="
        + json.dumps(
            {task: value["sha256"] for task, value in checkpoints.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
