#!/usr/bin/env python3
"""Materialize the exact released OpenPSG test split as a derived annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(source: Path, expected_nonempty: int) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("PSG source must be a dictionary with a data list")
    test_ids = {str(value) for value in payload.get("test_image_ids", [])}
    if not test_ids:
        raise ValueError("PSG source has no test_image_ids")
    selected = [
        record for record in payload["data"]
        if str(record.get("image_id")) in test_ids
    ]
    selected_ids = {str(record.get("image_id")) for record in selected}
    missing_ids = sorted(test_ids - selected_ids)
    if missing_ids:
        raise ValueError("PSG test IDs missing records: " + str(missing_ids[:10]))
    nonempty = [record for record in selected if record.get("relations")]
    if len(nonempty) != expected_nonempty:
        raise ValueError(
            "Unexpected nonempty PSG test graph count: "
            f"{len(nonempty)} != {expected_nonempty}"
        )
    derived = {key: value for key, value in payload.items() if key != "data"}
    derived["data"] = selected
    derived["_sgg_derivation"] = {
        "schema": "psg_official_test_split_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "test_ids": len(test_ids),
        "records": len(selected),
        "nonempty_relation_graphs": len(nonempty),
    }
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--expected_nonempty", type=int, default=2177)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    source = Path(args.source or root / "data/psg/psg.json").expanduser().resolve()
    output = Path(
        args.output or root / "data/derived/psg/psg_official_test.json"
    ).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    derived = build_split(source, args.expected_nonempty)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(derived, ensure_ascii=True) + "\n", encoding="utf-8")
    metadata = derived["_sgg_derivation"]
    print("output=" + str(output))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
