#!/usr/bin/env python3
"""Register the converged live PySGG models required by Experiments II and V."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


CALIBRATOR_PARAMETERS = 51 * 51 + 51 + 151 * 151 + 151
RUNTIME_PARAMETER_COUNTS = {
    # Measured from the pinned PySGG worker after full model construction and
    # checkpoint loading. Export metadata counts only checkpoint tensors and
    # omits runtime-created modules.
    "motifs": 367_594_769,
    "transformer": 331_076_267,
}
SPECS = {
    "motifs": {
        "name": "pysgg_motifs_vg_live",
        "family": "Neural Motifs",
        "paradigm": "sequential_context",
        "references": {
            "PredCls/R@50": 0.6518, "PredCls/mR@50": 0.1479,
            "SGCls/R@50": 0.3892, "SGCls/mR@50": 0.0828,
            "SGDet/R@50": 0.3278, "SGDet/mR@50": 0.0675,
        },
    },
    "bgnn": {
        "name": "pysgg_bgnn_vg_live",
        "family": "BGNN",
        "paradigm": "bipartite_graph_message_passing",
        "references": {"SGDet/R@50": 0.2980, "SGDet/mR@50": 0.1090},
    },
    "transformer": {
        "name": "pysgg_transformer_vg_live",
        "family": "SGG Transformer",
        "paradigm": "transformer_context",
        "references": {
            "PredCls/R@50": 0.6555, "PredCls/mR@50": 0.1630,
            "SGCls/R@50": 0.4018, "SGCls/mR@50": 0.1009,
            "SGDet/R@50": 0.3304, "SGDet/mR@50": 0.0813,
        },
    },
}
TASKS = ("predcls", "sgcls", "sgdet")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--models", nargs="+", choices=sorted(SPECS),
                        default=["motifs", "transformer"])
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    source = root / "external/official_repos/PySGG"
    marker = json.loads((source / ".official_source.json").read_text())
    output_dir = Path(
        args.output_dir or root / "checkpoints/sgg/manifests"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for model_key in tuple(dict.fromkeys(args.models)):
        spec = SPECS[model_key]
        cache = root / "artifacts/prediction_cache" / (
            "pysgg_" + model_key + "_vg_tritask"
        )
        metadata = json.loads((cache / "metadata.json").read_text())
        if set(metadata.get("tasks", [])) != set(TASKS):
            raise RuntimeError(f"Incomplete tri-task cache: {cache}")
        if any(int(metadata["images_by_task"].get(task, 0)) != 26446 for task in TASKS):
            raise RuntimeError(f"Incomplete full-VG cache coverage: {cache}")

        checkpoints = {}
        states = {}
        for task in TASKS:
            state_path = cache / f"state_{task}.json"
            state = json.loads(state_path.read_text())
            states[task] = state
            checkpoint = Path(state["checkpoint"]).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            coverage = state.get("checkpoint_load_coverage", {})
            if min(
                float(coverage.get("parameter_coverage", 0.0)),
                float(coverage.get("relation_parameter_coverage", 0.0)),
            ) < 0.98:
                raise RuntimeError(f"Insufficient checkpoint coverage: {state_path}")
            checkpoints[task] = {
                "path": str(checkpoint), "sha256": sha256(checkpoint),
            }
        ontology_id = str(metadata["ontology_id"])
        references = spec["references"]
        manifest = {
            "name": spec["name"],
            "architecture_family": spec["family"],
            "paradigm": spec["paradigm"],
            "execution_mode": "live_adapter",
            "factory": "sgg_core.models.adapters.pysgg_live:create_adapter",
            "environment_python": sys.executable,
            "checkpoints": checkpoints,
            "supported_tasks": list(TASKS),
            "diagnostic_task": "sgcls",
            "source_url": marker["repository_url"],
            "source_root": str(source),
            "source_commit": marker["commit"],
            "training_dataset": "VG-150",
            "reference_dataset": "vg",
            "metric_scale": "fraction",
            "reproduction_tolerance": 0.02,
            "input_source": "raw_images",
            "training_seed": 666,
            # Only the SGCls network is resident for live interventions. The
            # other two checkpoints are consumed through their verified cache.
            "parameter_count": (
                int(RUNTIME_PARAMETER_COUNTS[model_key])
                + CALIBRATOR_PARAMETERS
            ),
            "baseline_mR": float(references.get("SGDet/mR@50", 0.0)),
            "supported_datasets": ["vg"],
            "ontology_ids": {"vg": ontology_id},
            "perturbation_contract": {key: True for key in PERTURBATIONS},
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
                "worker_python": str(Path(
                    os.environ.get(
                        "PYSGG_PYTHON",
                        "python3",
                    )
                ).expanduser().resolve()),
                "worker_script": str(root / "scripts/pysgg_live_worker.py"),
                "diagnostic_config": str(
                    root / "configs/pysgg_vg_tritask"
                    / f"{model_key}_sgcls.yaml"
                ),
                "prediction_cache_root": str(cache),
                "official_parameter_count": int(
                    RUNTIME_PARAMETER_COUNTS[model_key]
                ),
                "checkpoint_tensor_parameter_count": int(
                    states["sgcls"]["parameter_count"]
                ),
                "parameter_count_semantics": (
                    "runtime_constructed_model_plus_shared_calibrators"
                ),
                "relation_score_mode": "categorical",
                "mitigation_scope": (
                    "post_hoc_calibration_shared_across_predcls_sgcls_sgdet"
                ),
            },
            "reference_metrics": references,
            "reference_eval_images": 26446,
        }
        output = output_dir / f"pysgg_{model_key}_vg_live.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written.append(str(output))
        print("manifest=" + str(output))

    report = root / "artifacts/manifests/pysgg_live_registration.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "schema": "pysgg_live_registration_v1",
        "models": list(dict.fromkeys(args.models)),
        "manifests": written,
        "calibrator_parameters": CALIBRATOR_PARAMETERS,
    }, indent=2) + "\n")
    print("report=" + str(report))


if __name__ == "__main__":
    main()
