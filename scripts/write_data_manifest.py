#!/usr/bin/env python3
"""
Write a reproducibility manifest for local dataset/checkpoint layout.

The manifest records whether each required file exists, its size, and a small
fingerprint for lightweight files. Large image archives are not hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(
    os.environ.get("SGG_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
HASH_LIMIT_BYTES = 64 * 1024 * 1024


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file() or path.stat().st_size > HASH_LIMIT_BYTES:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file(path: Path) -> dict:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "sha256": _sha256(path),
    }


def _first_existing(paths: list[Path]) -> dict:
    for path in paths:
        if path.exists():
            item = _file(path)
            item["candidates"] = [str(p) for p in paths]
            return item
    return {"path": None, "exists": False, "size_bytes": None, "sha256": None, "candidates": [str(p) for p in paths]}


def _image_directory(path: Path) -> dict:
    files = []
    if path.is_dir():
        files = [
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "image_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def _json_manifests(path: Path) -> list[dict]:
    records = []
    if not path.is_dir():
        return records
    for manifest in sorted(path.glob("openimages_*_vrd_*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            records.append({
                "path": str(manifest),
                "split": payload.get("split"),
                "selected_images": payload.get("selected_images"),
                "present_images": payload.get("present_images"),
                "complete": payload.get("complete"),
                "selected_ids_sha256": payload.get("selected_ids_sha256"),
            })
        except (OSError, ValueError, TypeError):
            records.append({"path": str(manifest), "invalid": True})
    return records


def build_manifest(project_root: Path) -> dict:
    data = project_root / "data"
    ckpt = project_root / "checkpoints"
    canonical_oi_root = data / "openimages" / "open-images-v6"
    configured_oi_value = os.environ.get("SGG_OI_ROOT")
    configured_oi_root = Path(configured_oi_value) if configured_oi_value else None
    oi_root = (
        configured_oi_root
        if configured_oi_root is not None
        and (configured_oi_root / "annotations").is_dir()
        else canonical_oi_root
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "datasets": {
            "vg_sgg_h5": _first_existing([
                data / "vg" / "v1.4" / "VG-SGG-with-attri.h5",
                data / "vg" / "v1.4" / "VG_SGG_with_attri.h5",
                data / "vg" / "v1.4" / "VG-SGG.h5",
                data / "vg" / "VG-SGG.h5",
                data / "vg" / "VG_SGG_with_attri.h5",
            ]),
            "vg_sgg_dict": _first_existing([
                data / "vg" / "v1.4" / "VG-SGG-dicts-with-attri.json",
                data / "vg" / "v1.4" / "VG_SGG_dicts_with_attri.json",
                data / "vg" / "v1.4" / "VG-SGG-dicts.json",
                data / "vg" / "VG-SGG-dicts.json",
                data / "vg" / "VG-SGG-dicts-with-attri.json",
            ]),
            "vg_relationships": _file(data / "vg" / "v1.4" / "relationships.json"),
            "openimages_vrd_validation": _first_existing([
                oi_root / "annotations" / "oidv6-validation-annotations-vrd.csv",
                oi_root / "annotations" / "validation-annotations-vrd.csv",
                oi_root / "annotations" / "validation" / "vrd.csv",
            ]),
            "openimages_vrd_train": _first_existing([
                oi_root / "annotations" / "oidv6-train-annotations-vrd.csv",
                oi_root / "annotations" / "train" / "vrd.csv",
            ]),
            "openimages_boxable_classes": _first_existing([
                oi_root / "annotations" / "class-descriptions-boxable.csv",
                oi_root / "annotations" / "oidv7-class-descriptions-boxable.csv",
                oi_root / "annotations" / "oidv6-class-descriptions-boxable.csv",
            ]),
            "openimages_all_class_descriptions": _first_existing([
                oi_root / "annotations" / "oidv6-class-descriptions.csv",
                oi_root / "annotations" / "oidv7-class-descriptions.csv",
                oi_root / "annotations" / "class-descriptions.csv",
            ]),
            "openimages_attribute_descriptions": _first_existing([
                oi_root / "annotations" / "oidv6-attributes-description.csv",
                oi_root / "annotations" / "attributes-description.csv",
            ]),
            "openimages_relationship_names": _first_existing([
                oi_root / "annotations" / "oidv6-relationships-description.csv",
                oi_root / "annotations" / "relationships_description.csv",
            ]),
            "openimages_images": _image_directory(oi_root / "images"),
            "openimages_selections": _json_manifests(oi_root / "manifests"),
            "gqa_train": _file(data / "gqa" / "train_sceneGraphs.json"),
            "gqa_train_scenegraphs_dir": _file(data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json"),
            "gqa_train_resolved": _first_existing([
                data / "gqa" / "train_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "train_sceneGraphs.json",
            ]),
            "gqa_val": _file(data / "gqa" / "val_sceneGraphs.json"),
            "gqa_val_scenegraphs_dir": _file(data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json"),
            "gqa_val_resolved": _first_existing([
                data / "gqa" / "val_sceneGraphs.json",
                data / "gqa" / "sceneGraphs" / "val_sceneGraphs.json",
            ]),
            "psg_train_val": _file(data / "psg" / "psg_train_val.json"),
            "psg_val_test": _file(data / "psg" / "psg_val_test.json"),
            "vrd_train": _file(data / "vrd" / "json_dataset" / "annotations_train.json"),
            "vrd_test": _file(data / "vrd" / "json_dataset" / "annotations_test.json"),
        },
        "checkpoints": {
            "sgg_dir": str(ckpt / "sgg"),
            "foundation_dir": str(ckpt / "foundation"),
            "sgg_files": sorted(str(p) for p in (ckpt / "sgg").glob("*.pth")) if (ckpt / "sgg").exists() else [],
            "foundation_files": sorted(str(p) for p in (ckpt / "foundation").glob("*")) if (ckpt / "foundation").exists() else [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Write SGG data/checkpoint manifest")
    parser.add_argument("--project_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out = Path(args.out) if args.out else project_root / "artifacts" / "manifests" / "data_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(project_root)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    stamped = out.with_name(f"data_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(stamped, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest: {out}")
    print(f"Wrote snapshot: {stamped}")


if __name__ == "__main__":
    main()
