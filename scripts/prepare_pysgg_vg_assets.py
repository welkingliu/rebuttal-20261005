#!/usr/bin/env python3
"""Create PySGG's native VG-150 layout without copying the raw dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def replace_symlink(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError("Refusing to replace non-symlink asset: " + str(target))
    target.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--glove_dir")
    parser.add_argument("--expected_images", type=int, default=108073)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    repo = root / "external/official_repos/PySGG"
    data = root / "data/vg"
    native = repo / "datasets/vg"
    required = {
        "VG-SGG-with-attri.h5": data / "VG-SGG-with-attri.h5",
        "VG-SGG-dicts-with-attri.json": data / "VG-SGG-dicts-with-attri.json",
        "image_data.json": data / "image_data.json",
    }
    for name, source in required.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        replace_symlink(source, native / name)

    image_dir = native / "stanford_spilt/VG_100k_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_metadata = json.loads(required["image_data.json"].read_text())
    expected_names = {
        Path(urlsplit(str(row.get("url", ""))).path).name
        or (str(int(row["image_id"])) + ".jpg")
        for row in image_metadata
    }
    if len(image_metadata) != int(args.expected_images) or len(expected_names) != int(args.expected_images):
        raise RuntimeError(
            "VG image_data coverage=%d unique_names=%d expected=%d" % (
                len(image_metadata), len(expected_names), args.expected_images,
            )
        )

    discovered = {}
    for candidate_root in (data / "VG_100K", data / "VG_100K_2"):
        if not candidate_root.is_dir():
            continue
        for path in candidate_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                previous = discovered.setdefault(path.name, path.resolve())
                if previous != path.resolve() and previous.stat().st_size != path.stat().st_size:
                    raise RuntimeError("Conflicting VG image basename: " + path.name)
    missing = sorted(expected_names - set(discovered))
    extras = sorted(set(discovered) - expected_names)
    if missing:
        raise RuntimeError(
            "VG official image coverage=%d/%d missing_examples=%s" % (
                len(expected_names) - len(missing), args.expected_images, missing[:10],
            )
        )
    for name in sorted(expected_names):
        source = discovered[name]
        target = image_dir / name
        if not target.exists():
            target.symlink_to(source)

    glove = Path(
        args.glove_dir or root / "data/derived/glove"
    ).expanduser().resolve()
    glove_file = glove / "glove.6B.200d.txt"
    report = {
        "status": "ready" if glove_file.is_file() else "missing_glove",
        "repository": str(repo),
        "native_data_root": str(native),
        "images": len(expected_names),
        "discovered_files": len(discovered),
        "extra_files_ignored": len(extras),
        "extra_file_examples": extras[:20],
        "glove_dir": str(glove),
        "glove_file": str(glove_file),
        "glove_download_url": "https://nlp.stanford.edu/data/glove.6B.zip",
    }
    report_path = root / "artifacts/manifests/pysgg_vg_assets.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("report=" + str(report_path))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
