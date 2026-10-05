#!/usr/bin/env python3
"""Generate pinned task-specific PySGG configs for the formal VG benchmark."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path

import yaml


FAMILIES = {
    "motifs": {
        "base": "e2e_relation_X_101_32_8_FPN_1x.yaml",
        "predictor": "MotifPredictor",
        "max_iter": 50000,
    },
    "vctree": {
        "base": "e2e_relation_X_101_32_8_FPN_1x.yaml",
        "predictor": "VCTreePredictor",
        "max_iter": 50000,
    },
    "transformer": {
        "base": "e2e_relation_X_101_32_8_FPN_1x.yaml",
        "predictor": "TransformerPredictor",
        "max_iter": 16000,
    },
    "bgnn": {
        "base": "e2e_relBGNN_vg.yaml",
        "predictor": "BGNNPredictor",
        "max_iter": 70000,
    },
    "tde_motifs": {
        "base": "e2e_relation_X_101_32_8_FPN_1x.yaml",
        "predictor": "CausalAnalysisPredictor",
        "effect_type": "TDE",
        "max_iter": 50000,
    },
}
TASK_FLAGS = {
    "predcls": (True, True),
    "sgcls": (True, False),
    "sgdet": (False, False),
}
REFERENCE_TRAIN_BATCH = 12


def scale_steps(value, scale):
    """Scale either PySGG's tuple-string or a structured step sequence."""
    was_string = isinstance(value, str)
    parsed = ast.literal_eval(value) if was_string else value
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("SOLVER.STEPS must be a list, tuple, or tuple string")
    scaled = tuple(int(round(int(step) * scale)) for step in parsed)
    return str(scaled) if was_string else list(scaled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--detector_checkpoint")
    parser.add_argument("--glove_dir")
    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--test_batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    repo = root / "external/official_repos/PySGG"
    detector = Path(
        args.detector_checkpoint
        or root / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"
    ).expanduser().resolve()
    glove = Path(args.glove_dir or root / "data/derived/glove").expanduser().resolve()
    if not detector.is_file():
        raise FileNotFoundError(detector)
    if not (glove / "glove.6B.200d.txt").is_file():
        raise FileNotFoundError(glove / "glove.6B.200d.txt")
    if args.train_batch <= 0 or args.train_batch % 2:
        raise ValueError("Two-GPU training requires a positive even --train_batch")
    schedule_scale = REFERENCE_TRAIN_BATCH / float(args.train_batch)
    generated = root / "configs/pysgg_vg_tritask"
    generated.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, spec in FAMILIES.items():
        base_path = repo / "configs" / spec["base"]
        base = yaml.safe_load(base_path.read_text())
        for task, (use_gt_box, use_gt_label) in TASK_FLAGS.items():
            config = deepcopy(base)
            relation = config["MODEL"]["ROI_RELATION_HEAD"]
            relation["PREDICTOR"] = spec["predictor"]
            if "effect_type" in spec:
                relation["CAUSAL"]["EFFECT_TYPE"] = spec["effect_type"]
                relation["CAUSAL"]["EFFECT_ANALYSIS"] = True
                relation["CAUSAL"]["CONTEXT_LAYER"] = "motifs"
            relation["USE_GT_BOX"] = use_gt_box
            relation["USE_GT_OBJECT_LABEL"] = use_gt_label
            config["MODEL"]["PRETRAINED_DETECTOR_CKPT"] = str(detector)
            config["MODEL"]["WEIGHT"] = ""
            config["DTYPE"] = "float32"
            config["GLOVE_DIR"] = str(glove)
            config.setdefault("DATALOADER", {})["NUM_WORKERS"] = int(args.workers)
            config.setdefault("TEST", {})["IMS_PER_BATCH"] = int(args.test_batch)
            config["TEST"].setdefault("RELATION", {})["SYNC_GATHER"] = False
            solver = config.setdefault("SOLVER", {})
            solver["IMS_PER_BATCH"] = int(args.train_batch)
            solver["MAX_ITER"] = int(round(spec["max_iter"] * schedule_scale))
            solver["VAL_PERIOD"] = int(round(2000 * schedule_scale))
            solver["CHECKPOINT_PERIOD"] = int(round(2000 * schedule_scale))
            if family == "transformer":
                solver["BASE_LR"] = 0.001
                solver.setdefault("SCHEDULE", {})["TYPE"] = "WarmupMultiStepLR"
                solver["STEPS"] = [10000, 16000]
            solver["BASE_LR"] = float(solver["BASE_LR"]) * (
                args.train_batch / REFERENCE_TRAIN_BATCH
            )
            if "STEPS" in solver:
                solver["STEPS"] = scale_steps(solver["STEPS"], schedule_scale)
            output = root / "checkpoints/sgg/trained/pysgg" / family / task
            config["OUTPUT_DIR"] = str(output)
            path = generated / (family + "_" + task + ".yaml")
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            rows.append({
                "family": family, "task": task, "config": str(path),
                "output_dir": str(output), "max_iter": solver["MAX_ITER"],
                "train_batch": solver["IMS_PER_BATCH"],
            })
    report = {
        "schema": "pysgg_vg_tritask_training_plan_v1",
        "detector_checkpoint": str(detector),
        "glove_dir": str(glove),
        "reference_train_batch": REFERENCE_TRAIN_BATCH,
        "schedule_scale": schedule_scale,
        "runs": rows,
    }
    report_path = root / "artifacts/manifests/pysgg_vg_tritask_training_plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("configs=" + str(generated))
    print("report=" + str(report_path))


if __name__ == "__main__":
    main()
