#!/usr/bin/env python3
"""Validate exact object/predicate ID alignment for legacy VG checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgg_core.vg_ontology import assert_vg150_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_dict", required=True)
    parser.add_argument("--candidate", action="append", default=[], required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    results = [
        assert_vg150_alignment(args.canonical_dict, candidate)
        for candidate in args.candidate
    ]
    payload = {
        "schema": "vg150_ontology_alignment_v1",
        "status": "aligned",
        "canonical": str(Path(args.canonical_dict).expanduser().resolve()),
        "candidates": results,
    }
    if args.report:
        output = Path(args.report).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"report={output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
