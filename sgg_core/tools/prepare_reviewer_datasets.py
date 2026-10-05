#!/usr/bin/env python3
"""Validate dataset entry points and raw-image coverage before formal runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sgg_core.tools.download_openimages import (
    SPLIT_TO_VRD_FILE,
    download_annotations,
    download_images,
    resolve_annotation,
)


DEFAULT_DATASETS = ["vg", "oi", "gqa", "psg", "vrd"]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _entry(label: str, paths: list[Path]) -> dict:
    found = _first_existing(paths)
    print(f"  [{'ok' if found else 'miss'}] {label}: {found or ''}")
    if found is None:
        for path in paths:
            print(f"         expected: {path}")
    return {
        "label": label,
        "ok": found is not None,
        "resolved": str(found.resolve()) if found else None,
        "candidates": [str(path) for path in paths],
    }


def _image_ok(path: Path | None, verify_content: bool) -> bool:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return False
    if not verify_content:
        return True
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (ImportError, OSError, ValueError):
        return False


def _coverage_result(dataset: str, expected: int, present: int,
                     missing: list[str], invalid: list[str]) -> dict:
    result = {
        "dataset": dataset,
        "expected": expected,
        "present": present,
        "missing": len(missing),
        "invalid": len(invalid),
        "complete": not missing and not invalid,
        "missing_examples": missing[:20],
        "invalid_examples": invalid[:20],
    }
    print(f"  [{'ok' if result['complete'] else 'miss'}] {dataset} raw images: "
          f"{present:,}/{expected:,} missing={len(missing):,} "
          f"invalid={len(invalid):,}")
    return result


def _check_oi_images(oi_root: Path, num_samples: int,
                     verify_content: bool) -> dict:
    ann_dir = oi_root / "annotations"
    rel_csv = resolve_annotation(
        ann_dir, SPLIT_TO_VRD_FILE["validation"]
    )
    if rel_csv is None:
        return _coverage_result("oi", 0, 0, ["validation VRD CSV"], [])
    report = download_images(
        oi_root=oi_root,
        split="validation",
        rel_csv=rel_csv,
        max_images=num_samples,
        num_workers=1,
        verify_content=verify_content,
        verify_only=True,
    )
    return _coverage_result(
        "oi", report["expected"], report["present"],
        report["missing"], report["invalid"],
    )


def _check_vg_images(vg_root: Path, train_samples: int, val_samples: int,
                     test_samples: int, verify_content: bool) -> dict:
    from sgg_core.data.data_utils import build_vg_test_loader

    missing, invalid, present, expected = [], [], 0, 0
    split_coverage = {}
    split_requests = [
        ("train", 0, train_samples),
    ]
    if val_samples > 0:
        split_requests.append(("validation", 1, val_samples))
    split_requests.append(("test", 2, test_samples))
    for split_name, split_id, num_samples in split_requests:
        requested = 10 ** 9 if num_samples <= 0 else num_samples
        loader = build_vg_test_loader(
            str(vg_root), requested, split=split_id,
            include_proxy_features=False, include_raw_images=False,
        )
        split_missing, split_invalid, split_present = [], [], 0
        try:
            for index in range(len(loader.dataset)):
                item = loader.dataset[index]
                image_id = str(item.get("image_id", index))
                raw_path = item.get("image_path")
                path = Path(raw_path) if raw_path else None
                if path is None or not path.is_file():
                    split_missing.append(image_id)
                elif not _image_ok(path, verify_content):
                    split_invalid.append(image_id)
                else:
                    split_present += 1
        finally:
            annot_file = getattr(loader.dataset, "annot_file", None)
            if annot_file is not None:
                annot_file.close()
            feat_file = getattr(loader.dataset, "feat_file", None)
            if feat_file is not None:
                feat_file.close()
        split_expected = len(loader.dataset)
        split_coverage[split_name] = {
            "split_id": split_id,
            "expected": split_expected,
            "present": split_present,
            "missing": len(split_missing),
            "invalid": len(split_invalid),
            "missing_examples": split_missing[:20],
            "invalid_examples": split_invalid[:20],
        }
        expected += split_expected
        present += split_present
        missing.extend(f"{split_name}:{item}" for item in split_missing)
        invalid.extend(f"{split_name}:{item}" for item in split_invalid)
    result = _coverage_result("vg", expected, present, missing, invalid)
    result["split_coverage"] = split_coverage
    return result


def _check_gqa_images(scene_graph: Path, image_root: Path, num_samples: int,
                      verify_content: bool) -> dict:
    with open(scene_graph, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(payload.items())
    sample_limit = None if num_samples <= 0 else num_samples
    selected = []
    skipped_missing = []
    for image_id, graph in records:
        objects = graph.get("objects", {})
        object_ids = {str(value) for value in objects}
        has_relation = any(
            str(relation.get("object", "")) in object_ids
            for obj in objects.values()
            for relation in obj.get("relations", [])
        )
        if has_relation:
            image_id = str(image_id)
            candidates = [image_root / f"{image_id}.jpg", image_root / image_id]
            if not any(candidate.is_file() for candidate in candidates):
                skipped_missing.append(image_id)
                continue
            selected.append(image_id)
            if sample_limit is not None and len(selected) >= sample_limit:
                break
    missing, invalid, present = [], [], 0
    for image_id in selected:
        candidates = [image_root / f"{image_id}.jpg", image_root / image_id]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            missing.append(image_id)
        elif not _image_ok(path, verify_content):
            invalid.append(image_id)
        else:
            present += 1
    expected = len(selected) if sample_limit is None else sample_limit
    if len(selected) < expected:
        missing.append(
            f"insufficient_available_graphs:{len(selected)}/{expected};"
            f"skipped_missing_images={len(skipped_missing)}"
        )
    result = _coverage_result("gqa", expected, present, missing, invalid)
    result["selection_skipped_missing_images"] = len(skipped_missing)
    result["selection_skipped_examples"] = skipped_missing[:20]
    if skipped_missing:
        print(f"         deterministic selection skipped "
              f"{len(skipped_missing)} unavailable GQA images")
    return result


def _check_vrd_images(vrd_root: Path, num_samples: int,
                      verify_content: bool) -> dict:
    from sgg_core.data.vrd_data_utils import build_vrd_loader

    requested = 10 ** 9 if num_samples <= 0 else num_samples
    loader = build_vrd_loader(
        str(vrd_root), "test", requested,
        include_proxy_features=False, include_raw_images=False,
    )
    missing, invalid, present = [], [], 0
    for index in range(len(loader.dataset)):
        image_id = str(loader.dataset.items[index][0])
        path = loader.dataset._find_image(image_id)
        if path is None or not path.is_file():
            missing.append(image_id)
        elif not _image_ok(path, verify_content):
            invalid.append(image_id)
        else:
            present += 1
    return _coverage_result(
        "vrd", len(loader.dataset), present, missing, invalid
    )


def _check_psg_images(annotation: Path, image_root: Path, num_samples: int,
                      verify_content: bool) -> dict:
    with open(annotation, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else payload.get(
        "data", payload.get("annotations", [])
    )
    if isinstance(records, dict):
        records = list(records.values())
    sample_limit = None if num_samples <= 0 else num_samples
    selected = []
    skipped_missing = []
    for index, record in enumerate(records):
        if not record.get("segments_info", record.get("annotations", [])):
            continue
        if not record.get("relations", record.get("relation_annotations", [])):
            continue
        file_name = record.get("file_name")
        candidates = [] if not file_name else [
            image_root / file_name,
            image_root / Path(file_name).name,
        ]
        image_available = any(candidate.is_file() for candidate in candidates)
        if sample_limit is not None and not image_available:
            skipped_missing.append(str(file_name or record.get("image_id", index)))
            continue
        selected.append(record)
        if sample_limit is not None and len(selected) >= sample_limit:
            break
    missing, invalid, present = [], [], 0
    for index, record in enumerate(selected):
        image_id = str(record.get("image_id", index))
        file_name = record.get("file_name")
        candidates = [] if not file_name else [
            image_root / file_name,
            image_root / Path(file_name).name,
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        panoptic_name = record.get(
            "pan_seg_file_name", record.get("panoptic_file_name")
        )
        panoptic_candidates = [] if not panoptic_name else [
            image_root / panoptic_name,
            image_root / Path(panoptic_name).name,
        ]
        panoptic_path = next(
            (candidate for candidate in panoptic_candidates if candidate.is_file()),
            None,
        )
        if path is None:
            missing.append(image_id)
        elif panoptic_path is None:
            missing.append(f"panoptic:{image_id}")
        elif not _image_ok(path, verify_content):
            invalid.append(image_id)
        elif not _image_ok(panoptic_path, verify_content):
            invalid.append(f"panoptic:{image_id}")
        else:
            present += 1
    expected = len(selected) if sample_limit is None else sample_limit
    if len(selected) < expected:
        missing.append(
            f"insufficient_available_graphs:{len(selected)}/{expected};"
            f"skipped_missing_images={len(skipped_missing)}"
        )
    result = _coverage_result("psg", expected, present, missing, invalid)
    result["selection_skipped_missing_images"] = len(skipped_missing)
    result["selection_skipped_examples"] = skipped_missing[:20]
    if skipped_missing:
        print(f"         deterministic selection skipped "
              f"{len(skipped_missing)} unavailable PSG images")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check reviewer-facing SGG datasets before a formal run"
    )
    default_root = os.environ.get(
        "SGG_PROJECT_ROOT", Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--project_root", default=str(default_root))
    parser.add_argument(
        "--oi_root", default=None,
        help="Open Images root; defaults to SGG_OI_ROOT or the canonical project path",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        choices=["vg", "oi", "gqa", "psg", "vrd"],
    )
    parser.add_argument(
        "--download_oi", "--download_oi_annotations", action="store_true",
        help="Download only official Open Images train/validation metadata",
    )
    parser.add_argument(
        "--strict_images", action="store_true",
        help="Require complete raw-image coverage for the selected samples",
    )
    parser.add_argument("--main_samples", type=int, default=2000)
    parser.add_argument(
        "--vg_train_samples", type=int, default=None,
        help="VG train coverage; defaults to --main_samples",
    )
    parser.add_argument(
        "--vg_val_samples", type=int, default=0,
        help="Optional VG validation coverage for mitigation; 0 disables it.",
    )
    parser.add_argument(
        "--vg_test_samples", type=int, default=None,
        help="VG test coverage; defaults to --main_samples",
    )
    parser.add_argument("--external_samples", type=int, default=1000)
    parser.add_argument(
        "--psg_eval_ann",
        help="Explicit PSG evaluation annotation, including a derived official split.",
    )
    parser.add_argument("--oi_samples", type=int, default=None)
    parser.add_argument("--verify_image_content", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    data = root / "data"
    datasets = list(dict.fromkeys(args.datasets))
    oi_root = Path(
        args.oi_root
        or os.environ.get("SGG_OI_ROOT", "")
        or data / "openimages" / "open-images-v6"
    ).expanduser().resolve()

    if args.download_oi:
        download_annotations(oi_root / "annotations", ["train", "validation"])

    print("\nDataset entry-point check")
    print("=" * 72)
    entries = []
    if "vg" in datasets:
        entries.extend([
            _entry("VG-SGG h5", [
                data / "vg" / "v1.4" / "VG-SGG-with-attri.h5",
                data / "vg" / "v1.4" / "VG_SGG_with_attri.h5",
                data / "vg" / "v1.4" / "VG-SGG.h5",
            ]),
            _entry("VG-SGG dict", [
                data / "vg" / "v1.4" / "VG-SGG-dicts-with-attri.json",
                data / "vg" / "v1.4" / "VG_SGG_dicts_with_attri.json",
                data / "vg" / "v1.4" / "VG-SGG-dicts.json",
            ]),
        ])
    if "oi" in datasets:
        entries.extend([
            _entry("OpenImages boxable classes", [
                oi_root / "annotations" / "class-descriptions-boxable.csv",
                oi_root / "annotations" / "oidv7-class-descriptions-boxable.csv",
                oi_root / "annotations" / "oidv6-class-descriptions-boxable.csv",
            ]),
            _entry("OpenImages VRD validation", [
                oi_root / "annotations" / "oidv6-validation-annotations-vrd.csv",
                oi_root / "annotations" / "validation-annotations-vrd.csv",
                oi_root / "annotations" / "validation" / "vrd.csv",
            ]),
            _entry("OpenImages VRD train", [
                oi_root / "annotations" / "oidv6-train-annotations-vrd.csv",
                oi_root / "annotations" / "train-annotations-vrd.csv",
                oi_root / "annotations" / "train" / "vrd.csv",
            ]),
        ])
    gqa_train = _first_existing([
        data / "gqa" / "train_sceneGraphs.json",
        data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
    ])
    gqa_val = _first_existing([
        data / "gqa" / "val_sceneGraphs.json",
        data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
    ])
    if "gqa" in datasets:
        entries.extend([
            _entry("GQA train scene graphs", [
                data / "gqa" / "train_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
            ]),
            _entry("GQA val scene graphs", [
                data / "gqa" / "val_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
            ]),
        ])
    psg_eval = (
        Path(args.psg_eval_ann).expanduser().resolve()
        if args.psg_eval_ann else _first_existing([
            data / "psg" / "psg_val_test.json",
            data / "psg" / "psg.json",
        ])
    )
    if "psg" in datasets:
        entries.extend([
            _entry("PSG train annotations", [data / "psg" / "psg_train_val.json"]),
            _entry("PSG val/test annotations", [psg_eval]),
        ])
    if "vrd" in datasets:
        entries.extend([
            _entry("VRD train annotations", [
                data / "vrd" / "json_dataset" / "annotations_train.json"
            ]),
            _entry("VRD test annotations", [
                data / "vrd" / "json_dataset" / "annotations_test.json"
            ]),
        ])

    coverage = []
    if args.strict_images and all(entry["ok"] for entry in entries):
        print("\nExact raw-image coverage")
        print("=" * 72)
        if "vg" in datasets:
            vg_train_samples = (
                args.main_samples
                if args.vg_train_samples is None else args.vg_train_samples
            )
            vg_test_samples = (
                args.main_samples
                if args.vg_test_samples is None else args.vg_test_samples
            )
            coverage.append(_check_vg_images(
                data / "vg" / "v1.4", vg_train_samples,
                args.vg_val_samples, vg_test_samples,
                args.verify_image_content,
            ))
        if "oi" in datasets:
            oi_samples = args.main_samples if args.oi_samples is None else args.oi_samples
            coverage.append(_check_oi_images(
                oi_root, oi_samples, args.verify_image_content
            ))
        if "gqa" in datasets and gqa_val is not None:
            coverage.append(_check_gqa_images(
                gqa_val, data / "gqa" / "images", args.external_samples,
                args.verify_image_content,
            ))
        if "psg" in datasets and psg_eval is not None:
            coverage.append(_check_psg_images(
                psg_eval, data / "coco", args.external_samples,
                args.verify_image_content,
            ))
        if "vrd" in datasets:
            coverage.append(_check_vrd_images(
                data / "vrd", args.external_samples,
                args.verify_image_content,
            ))

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(root),
        "datasets": datasets,
        "entry_points": entries,
        "raw_image_coverage": coverage,
        "complete": (
            all(entry["ok"] for entry in entries)
            and (not args.strict_images or all(item["complete"] for item in coverage))
        ),
    }
    report = Path(args.report) if args.report else (
        root / "artifacts" / "manifests" / "data_readiness.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nreport={report}")
    if not payload["complete"]:
        print("[NOT READY] Fix missing entry points or raw-image coverage.")
        return 1
    print("[READY] Requested datasets satisfy the preflight contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
