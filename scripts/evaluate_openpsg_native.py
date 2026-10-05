#!/usr/bin/env python3
"""Reproduce released OpenPSG SGDet metrics with native panoptic matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import mmcv
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_openpsg_cache_manifest import MODEL_SPECS
from scripts.export_openpsg_predictions import load_model, sha256


def _atomic_dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    mmcv.dump(value, str(temporary))
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--annotation")
    parser.add_argument("--coco_root")
    parser.add_argument(
        "--config",
        help="Optional repository config; default is the checkpoint-embedded config.",
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--max_images", type=int, default=2177)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    spec = MODEL_SPECS[args.model]
    source = root / "external/official_repos/OpenPSG"
    checkpoint = root / "checkpoints/sgg/weights" / spec["checkpoint"]
    annotation = Path(args.annotation or root / "data/psg/psg.json").resolve()
    coco_root = Path(args.coco_root or root / "data/coco").resolve()
    output = Path(
        args.output_dir
        or root / "artifacts/native_reference" / ("openpsg_" + args.model + "_psg")
    ).resolve()
    marker = json.loads((source / ".official_source.json").read_text())
    checkpoint_digest = sha256(checkpoint)
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(config_path)
    if config_path is None:
        checkpoint_payload = torch.load(str(checkpoint), map_location="cpu")
        config_text = checkpoint_payload.get("meta", {}).get("config")
        if not config_text:
            raise RuntimeError("OpenPSG checkpoint has no embedded config")
        config_source = "checkpoint_embedded"
        config_digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        del checkpoint_payload
    else:
        config_source = str(config_path)
        config_digest = sha256(config_path)
    model_name = "openpsg_" + args.model + "_psg_official"
    state = {
        "schema": "openpsg_native_panoptic_reference_v1",
        "model": model_name,
        "source_commit": marker["commit"],
        "checkpoint_sha256": checkpoint_digest,
        "config_source": config_source,
        "config_sha256": config_digest,
        "annotation": str(annotation),
        "protocol": {
            "task": "sgdet",
            "detection_method": "pan_seg",
            "iou_threshold": 0.5,
            "graph_constraint": True,
        },
        "requested_images": int(args.max_images),
    }
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    if state_path.is_file() and json.loads(state_path.read_text()) != state:
        raise RuntimeError("Refusing to mix native-reference provenance")
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    model, dataset, loader = load_model(
        args.model, source, checkpoint, annotation, coco_root,
        native_panoptic=True, config_path=config_path,
    )
    total = min(len(dataset), int(args.max_images))
    if total != len(dataset):
        print("[smoke] partial native inference; no formal evaluation will be emitted")
    from mmcv.parallel import MMDataParallel
    model = MMDataParallel(model, device_ids=[0])
    prediction_dir = output / "predictions"
    started = time.monotonic()
    for index, data in enumerate(loader):
        if index >= total:
            break
        path = prediction_dir / ("%06d.pkl" % index)
        if args.resume and path.is_file():
            pass
        else:
            with torch.no_grad():
                values = model(return_loss=False, rescale=True, **data)
            if isinstance(values, (list, tuple)):
                if len(values) != 1:
                    raise ValueError("Native OpenPSG inference must return one Result")
                result = values[0]
            else:
                result = values
            if not hasattr(result, "rel_pair_idxes"):
                raise ValueError("Native OpenPSG output is not a scene-graph Result")
            _atomic_dump(result, path)
        if (index + 1) % args.log_every == 0 or index + 1 == total:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({
                "model": args.model, "completed": index + 1,
                "total": total, "images_per_second": (index + 1) / elapsed,
            }), flush=True)

    if total != len(dataset):
        return
    predictions = [
        mmcv.load(str(prediction_dir / ("%06d.pkl" % index)))
        for index in range(total)
    ]
    metrics = dataset.evaluate(
        predictions, metric="sgdet", classwise=True, multiple_preds=False,
        iou_thrs=0.5, detection_method="pan_seg",
    )
    observed = {
        "SGDet/R@50": float(metrics["sgdet_recall_R_50"]),
        "SGDet/mR@50": float(metrics["sgdet_mean_recall_mR_50"]),
    }
    references = {
        "SGDet/R@50": float(spec["r50"]),
        "SGDet/mR@50": float(spec["mr50"]),
    }
    tolerance = 0.02
    comparisons = {
        name: {
            "reference": references[name],
            "observed": observed[name],
            "absolute_delta": abs(observed[name] - references[name]),
            "within_tolerance": abs(observed[name] - references[name]) <= tolerance,
        }
        for name in references
    }
    report = {
        **state,
        "status": (
            "pass" if all(row["within_tolerance"] for row in comparisons.values())
            else "fail"
        ),
        "eval_images": total,
        "absolute_tolerance": tolerance,
        "comparisons": comparisons,
        "raw_metrics": {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("native_report=" + str(report_path))
    print("native_status=" + report["status"])
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
