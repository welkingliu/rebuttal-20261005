#!/usr/bin/env python3
"""Download and verify the exact Open Images VRD subset used by this project.

The official Open Images downloader accepts an explicit ``split/ImageID``
list.  This tool derives the same lexicographically ordered IDs as the local
loaders, downloads only those images, and writes a reproducibility manifest.

Recommended two-GPU preparation::

    python -m sgg_core.tools.download_openimages \
        --profile reduced_2gpu --strict --num_workers 16

The reduced profile downloads all validation images carrying VRD annotations
for the full standard benchmark, plus the first 1,500 train images used for
motif mining and focused diagnostics. Existing valid files are reused, so
rerunning resumes safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(
    os.environ.get("SGG_PROJECT_ROOT", Path(__file__).resolve().parents[3])
)
DEFAULT_OI_ROOT = PROJECT_ROOT / "data" / "openimages" / "open-images-v6"

# These links are the targets exposed by the official Open Images V6 page.
ANNOTATION_FILES = {
    "oidv6-class-descriptions.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-class-descriptions.csv"
    ),
    "oidv6-attributes-description.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-attributes-description.csv"
    ),
    "class-descriptions-boxable.csv": (
        "https://storage.googleapis.com/openimages/v5/"
        "class-descriptions-boxable.csv"
    ),
    "oidv6-relationships-description.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-relationships-description.csv"
    ),
    "oidv6-relationship-triplets.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-relationship-triplets.csv"
    ),
    "oidv6-train-annotations-vrd.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-train-annotations-vrd.csv"
    ),
    "oidv6-validation-annotations-vrd.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-validation-annotations-vrd.csv"
    ),
    "oidv6-test-annotations-vrd.csv": (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-test-annotations-vrd.csv"
    ),
}

SPLIT_TO_VRD_FILE = {
    "train": "oidv6-train-annotations-vrd.csv",
    "validation": "oidv6-validation-annotations-vrd.csv",
    "test": "oidv6-test-annotations-vrd.csv",
}

ANNOTATION_ALIASES = {
    "oidv6-attributes-description.csv": ["attributes-description.csv"],
    "oidv6-class-descriptions.csv": [
        "oidv7-class-descriptions.csv",
        "class-descriptions.csv",
    ],
    "class-descriptions-boxable.csv": [
        "oidv7-class-descriptions-boxable.csv",
        "oidv6-class-descriptions-boxable.csv",
    ],
    "oidv6-relationships-description.csv": ["relationships_description.csv"],
    "oidv6-validation-annotations-vrd.csv": [
        "validation-annotations-vrd.csv",
        "validation/vrd.csv",
    ],
    "oidv6-test-annotations-vrd.csv": [
        "test-annotations-vrd.csv",
        "test/vrd.csv",
    ],
    "oidv6-train-annotations-vrd.csv": [
        "train-annotations-vrd.csv",
        "train/vrd.csv",
    ],
}

OI_IMAGE_URL = (
    "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"
)

# 0 means every image ID in that relationship split.
PROFILES = {
    "smoke": {"train": 100, "validation": 100},
    "reduced_2gpu": {"train": 1500, "validation": 0},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _valid_annotation(path: Path, canonical_name: str) -> bool:
    if not _valid_nonempty_file(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    if canonical_name.endswith("annotations-vrd.csv"):
        return "ImageID" in head and "LabelName1" in head
    if "class-descriptions" in canonical_name:
        return "/m/" in head and "," in head
    if "relationships-description" in canonical_name:
        return "," in head and "<Error>" not in head
    if "attributes-description" in canonical_name:
        return "/m/" in head and "," in head and "<Error>" not in head
    if "relationship-triplets" in canonical_name:
        return "," in head and "<Error>" not in head
    return "<Error>" not in head


def _valid_image(path: Path, verify_content: bool = False) -> bool:
    if not _valid_nonempty_file(path):
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


def locate_image(
    images_dir: Path,
    image_id: str,
    fallback_dirs: Iterable[Path] = (),
) -> Path | None:
    """Resolve canonical split storage plus legacy flat/sharded layouts."""
    search_roots = tuple(dict.fromkeys((images_dir, *fallback_dirs)))
    for suffix in (".jpg", ".jpeg", ".png"):
        for root in search_roots:
            candidates = (
                root / f"{image_id}{suffix}",
                root / image_id[0].lower() / f"{image_id}{suffix}",
                root / image_id[0].upper() / f"{image_id}{suffix}",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    return None


def resolve_annotation(ann_dir: Path, canonical_name: str) -> Path | None:
    for name in (canonical_name, *ANNOTATION_ALIASES.get(canonical_name, [])):
        candidate = ann_dir / name
        if _valid_annotation(candidate, canonical_name):
            return candidate
    return None


def _download_atomic(url: str, dest: Path, retries: int = 4) -> bool:
    """Download to ``.part`` and atomically publish a complete file."""
    if _valid_nonempty_file(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part")

    for attempt in range(1, retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "sgg-experiment-data/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                with open(part, mode) as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            if not _valid_nonempty_file(part):
                raise OSError("downloaded file is empty")
            part.replace(dest)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                break
            if error.code == 416 and part.exists():
                part.unlink()
                continue
            if attempt == retries:
                break
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt == retries:
                break
        time.sleep(min(2 ** attempt, 16))
    return False


def required_annotation_names(splits: Iterable[str]) -> set[str]:
    """Return metadata actually consumed by the OI-VRD preparation protocol."""
    return {
        "oidv6-class-descriptions.csv",
        "class-descriptions-boxable.csv",
        "oidv6-relationships-description.csv",
        "oidv6-relationship-triplets.csv",
        *(SPLIT_TO_VRD_FILE[split] for split in splits),
    }


def download_annotations(ann_dir: Path, splits: Iterable[str]) -> dict[str, Path]:
    """Download canonical metadata and the requested relationship splits."""
    names = required_annotation_names(splits)
    resolved = {}
    print(f"\n[Annotations] {ann_dir}")
    for name in sorted(names):
        canonical_path = ann_dir / name
        existing = (
            canonical_path
            if _valid_annotation(canonical_path, name)
            else resolve_annotation(ann_dir, name)
        )
        # Pin the V6 VRD run to the official 600-class boxable table even when
        # a compatible V7 alias happens to be present locally.
        if name == "class-descriptions-boxable.csv" and existing != canonical_path:
            existing = None
        if existing is not None:
            print(f"  [ok]   {name} -> {existing.name}")
            resolved[name] = existing
            continue
        print(f"  [down] {name}")
        destination = canonical_path
        if destination.exists():
            destination.unlink()
        if not _download_atomic(ANNOTATION_FILES[name], destination):
            raise RuntimeError(f"Failed to download {name}")
        if not _valid_annotation(destination, name):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded annotation is invalid: {name}")
        resolved[name] = destination
    return resolved


def get_image_ids_from_rel_csv(rel_csv: Path) -> list[str]:
    """Return unique ImageIDs in the exact deterministic loader order."""
    image_ids = set()
    with open(rel_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "ImageID" not in (reader.fieldnames or []):
            raise ValueError(
                f"{rel_csv} is not an Open Images VRD CSV: missing ImageID"
            )
        for row in reader:
            image_id = row.get("ImageID", "").strip()
            if image_id:
                image_ids.add(image_id)
    return sorted(image_ids)


def select_image_ids(rel_csv: Path, max_images: int | None) -> list[str]:
    image_ids = get_image_ids_from_rel_csv(rel_csv)
    if max_images is not None and max_images > 0:
        return image_ids[:max_images]
    return image_ids


def image_coverage(
    images_dir: Path,
    image_ids: Iterable[str],
    verify_content: bool = False,
    fallback_dirs: Iterable[Path] = (),
) -> dict:
    present, missing, invalid = [], [], []
    for image_id in image_ids:
        path = locate_image(images_dir, image_id, fallback_dirs)
        if path is None:
            missing.append(image_id)
        elif not _valid_image(path, verify_content=verify_content):
            invalid.append(image_id)
        else:
            present.append(image_id)
    return {"present": present, "missing": missing, "invalid": invalid}


def _write_selection_manifest(
    oi_root: Path,
    split: str,
    rel_csv: Path,
    image_ids: list[str],
    coverage: dict,
    complete_split: bool,
) -> tuple[Path, Path]:
    manifest_dir = oi_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    count_label = "all" if complete_split else str(len(image_ids))
    stem = f"openimages_{split}_vrd_{count_label}"
    id_list_path = manifest_dir / f"{stem}_image_ids.txt"
    id_list_temporary = id_list_path.with_name(
        f"{id_list_path.name}.{os.getpid()}.tmp"
    )
    id_list_temporary.write_text(
        "".join(f"{split}/{image_id}\n" for image_id in image_ids),
        encoding="utf-8",
    )
    id_list_temporary.replace(id_list_path)
    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "dataset": "openimages-v6-vrd",
        "split": split,
        "selection_order": "lexicographic_unique_ImageID",
        "source_annotation": str(rel_csv.resolve()),
        "source_size_bytes": rel_csv.stat().st_size,
        "source_sha256": _sha256_file(rel_csv),
        "image_url_template": OI_IMAGE_URL,
        "selected_images": len(image_ids),
        "selected_ids_sha256": _sha256_lines(image_ids),
        "official_id_list": str(id_list_path.resolve()),
        "present_images": len(coverage["present"]),
        "missing_images": len(coverage["missing"]),
        "invalid_images": len(coverage["invalid"]),
        "complete": not coverage["missing"] and not coverage["invalid"],
        "missing_ids": coverage["missing"],
        "invalid_ids": coverage["invalid"],
    }
    manifest_path = manifest_dir / f"{stem}.json"
    temporary = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path, id_list_path


def download_images(
    oi_root: Path,
    split: str,
    rel_csv: Path,
    max_images: int | None,
    num_workers: int,
    verify_content: bool,
    verify_only: bool,
) -> dict:
    images_root = oi_root / "images"
    images_dir = images_root / split
    images_dir.mkdir(parents=True, exist_ok=True)
    image_ids = select_image_ids(rel_csv, max_images)
    # Existing releases of this project stored every split directly under
    # images/. Keep them readable while all new downloads use images/<split>/.
    legacy_dirs = (images_root,)
    coverage = image_coverage(
        images_dir, image_ids, verify_content, fallback_dirs=legacy_dirs
    )
    pending = sorted(set(coverage["missing"] + coverage["invalid"]))

    print(f"\n[Images:{split}] selected={len(image_ids):,} "
          f"present={len(coverage['present']):,} pending={len(pending):,}")
    if pending and not verify_only:
        def fetch(image_id: str) -> tuple[str, bool]:
            existing = locate_image(images_dir, image_id, legacy_dirs)
            destination = images_dir / f"{image_id}.jpg"
            if existing is not None and not _valid_image(existing, verify_content):
                existing.unlink(missing_ok=True)
            ok = _download_atomic(
                OI_IMAGE_URL.format(split=split, image_id=image_id), destination
            )
            if ok and not _valid_image(destination, verify_content):
                destination.unlink(missing_ok=True)
                ok = False
            return image_id, ok

        completed = failed = 0
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [executor.submit(fetch, image_id) for image_id in pending]
            for future in as_completed(futures):
                _, ok = future.result()
                completed += 1
                failed += int(not ok)
                if completed % 100 == 0 or completed == len(pending):
                    print(f"  progress={completed:,}/{len(pending):,} failed={failed:,}")
        coverage = image_coverage(
            images_dir, image_ids, verify_content, fallback_dirs=legacy_dirs
        )

    manifest_path, id_list_path = _write_selection_manifest(
        oi_root, split, rel_csv, image_ids, coverage,
        complete_split=max_images is None or max_images <= 0,
    )
    print(f"  manifest={manifest_path}")
    print(f"  official_image_list={id_list_path}")
    print(f"  canonical_image_dir={images_dir}")
    print(f"  legacy_flat_fallback={images_root}")
    print(f"  coverage={len(coverage['present']):,}/{len(image_ids):,} "
          f"missing={len(coverage['missing']):,} invalid={len(coverage['invalid']):,}")
    return {
        "split": split,
        "expected": len(image_ids),
        "present": len(coverage["present"]),
        "missing": coverage["missing"],
        "invalid": coverage["invalid"],
        "manifest": str(manifest_path),
    }


def build_plan(args: argparse.Namespace) -> dict[str, int]:
    if args.profile:
        plan = dict(PROFILES[args.profile])
        if args.max_images is not None:
            if args.split in (None, "all"):
                raise ValueError(
                    "--max_images with --profile requires one explicit --split"
                )
            plan = {args.split: args.max_images}
        return plan

    split = args.split or "validation"
    splits = list(SPLIT_TO_VRD_FILE) if split == "all" else [split]
    default_limit = 2000 if args.max_images is None else args.max_images
    return {name: default_limit for name in splits}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download exact Open Images V6 VRD annotations and images"
    )
    parser.add_argument(
        "--out_dir", "--oi_root", dest="oi_root", default=str(DEFAULT_OI_ROOT),
        help="Open Images root containing annotations/ and images/",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument(
        "--split", choices=["train", "validation", "test", "all"], default=None
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="First N lexicographic VRD image IDs; 0 means the complete split",
    )
    parser.add_argument("--num_workers", "--num_processes", type=int, default=8)
    parser.add_argument(
        "--include_train", action="store_true",
        help="Compatibility option: also prepare the train split",
    )
    parser.add_argument("--annotations_only", action="store_true")
    parser.add_argument("--images_only", action="store_true")
    parser.add_argument("--verify", "--verify_only", action="store_true")
    parser.add_argument(
        "--verify_content", action="store_true",
        help="Decode every selected image with Pillow during verification",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero unless every selected image is present and valid",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.annotations_only and args.images_only:
        raise ValueError("--annotations_only and --images_only are mutually exclusive")
    plan = build_plan(args)
    oi_root = Path(args.oi_root).expanduser().resolve()
    ann_dir = oi_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Open Images V6 VRD data preparation")
    print(f"root={oi_root}")
    print(f"plan={plan}")

    annotation_splits = set(plan)
    if args.include_train:
        annotation_splits.add("train")
    required_names = required_annotation_names(annotation_splits)
    if args.verify:
        resolved = {
            name: resolve_annotation(ann_dir, name) for name in required_names
        }
        missing_annotations = [
            name for name, path in resolved.items() if path is None
        ]
        if missing_annotations:
            print(f"[ERROR] missing annotations: {missing_annotations}")
            return 1
    elif args.images_only:
        resolved = {
            name: resolve_annotation(ann_dir, name) for name in required_names
        }
        missing_annotations = [
            name for name, path in resolved.items() if path is None
        ]
        if missing_annotations:
            print(f"[ERROR] missing annotations: {missing_annotations}")
            return 1
    else:
        resolved = download_annotations(ann_dir, annotation_splits)

    if args.annotations_only:
        print("\nAnnotation preparation complete.")
        return 0

    reports = []
    for split, max_images in plan.items():
        canonical = SPLIT_TO_VRD_FILE[split]
        rel_csv = resolved.get(canonical) or resolve_annotation(ann_dir, canonical)
        if rel_csv is None:
            print(f"[ERROR] missing relationship annotation for {split}")
            return 1
        reports.append(download_images(
            oi_root=oi_root,
            split=split,
            rel_csv=rel_csv,
            max_images=max_images,
            num_workers=args.num_workers,
            verify_content=args.verify_content,
            verify_only=args.verify,
        ))

    incomplete = [
        report for report in reports
        if report["missing"] or report["invalid"]
    ]
    if incomplete:
        print("\n[INCOMPLETE] Some selected images are unavailable or invalid.")
        for report in incomplete:
            print(f"  {report['split']}: missing={len(report['missing'])} "
                  f"invalid={len(report['invalid'])}")
        if args.strict:
            return 2
    else:
        print("\n[READY] All selected Open Images files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
