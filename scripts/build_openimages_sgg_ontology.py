#!/usr/bin/env python3
"""Build the checkpoint-aligned Open Images V6 SGG ontology."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--categories")
    parser.add_argument("--full_classes")
    parser.add_argument("--boxable_classes")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    ann = root / "data/openimages/open-images-v6/annotations"
    categories_path = Path(
        args.categories
        or root / "external/official_repos/Pix2Grp_CVPR2024/"
        "all_categories_dict/openimages/open-imagev6/categories_dict.json"
    ).expanduser().resolve()
    full_path = Path(
        args.full_classes or ann / "oidv6-class-descriptions.csv"
    ).expanduser().resolve()
    boxable_path = Path(
        args.boxable_classes or ann / "class-descriptions-boxable.csv"
    ).expanduser().resolve()
    output = Path(
        args.output or ann / "oi_sgg_ontology.json"
    ).expanduser().resolve()
    for path in (categories_path, full_path, boxable_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    object_names = list(categories.get("obj", []))
    predicate_names = list(categories.get("rel", []))
    if len(object_names) != 601 or len(predicate_names) != 30:
        raise ValueError("Expected the official 601-object/30-predicate OIv6 ontology")

    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with full_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                by_name[normalize(row[1])].append((row[0].strip(), row[1].strip()))
    with boxable_path.open(newline="", encoding="utf-8") as handle:
        boxable_mids = {row[0].strip() for row in csv.reader(handle) if row}

    object_categories = []
    for index, name in enumerate(object_names):
        candidates = by_name.get(normalize(name), [])
        preferred = [value for value in candidates if value[0] in boxable_mids]
        resolved = preferred if len(preferred) == 1 else candidates
        if len(resolved) != 1:
            raise ValueError(
                f"Cannot uniquely resolve OI category {index}:{name!r}: {candidates}"
            )
        mid, official_name = resolved[0]
        object_categories.append({"index": index, "mid": mid, "name": official_name})

    if len({item["mid"] for item in object_categories}) != len(object_categories):
        raise ValueError("Resolved OI object MIDs are not unique")
    canonical = {
        "schema": "openimages_v6_sgg_ontology_v1",
        "object_categories": object_categories,
        "predicate_categories": predicate_names,
        "source_categories": str(categories_path),
        "source_full_classes": str(full_path),
        "source_boxable_classes": str(boxable_path),
    }
    digest_payload = json.dumps(
        {"objects": object_categories, "predicates": predicate_names},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    canonical["sha256"] = hashlib.sha256(digest_payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    print(f"ontology={output}")
    print("objects=601 predicates=30 background_is_implicit=true")
    print(f"sha256={canonical['sha256']}")


if __name__ == "__main__":
    main()
