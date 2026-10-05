#!/usr/bin/env python3
"""Extract a detector-only initialization from an official PySGG checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload = torch.load(str(source), map_location="cpu")
    state = payload.get("model", payload)
    detector = {
        key: value for key, value in state.items()
        if "roi_heads.relation." not in key
    }
    if not detector or len(detector) == len(state):
        raise RuntimeError("Checkpoint did not expose separate detector/relation keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": detector}, str(output))
    report = {
        "schema": "pysgg_detector_extraction_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "source_tensors": len(state),
        "detector_tensors": len(detector),
        "excluded_relation_tensors": len(state) - len(detector),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
