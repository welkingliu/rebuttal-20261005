#!/usr/bin/env python3
"""Build the exact 11-run native SGDet manifest set reported in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


MANIFESTS = (
    "egtr_vg.json",
    "reltr_vg.json",
    "sgtr_vg.json",
    "kern_official_vg.json",
    "pysgg_bgnn_vg_sgdet.json",
    "egtr_oi.json",
    "sgtr_oi.json",
    "openpsg_motifs_psg.json",
    "openpsg_vctree_psg.json",
    "openpsg_psgformer_psg.json",
    "openpsg_psgtr_psg.json",
)
EXPECTED = {"vg": 5, "oi": 2, "psg": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    source = root / "checkpoints/sgg/manifests"
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "artifacts/manifests/paper_experiment4_native_sgdet"
    )
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("*.json"):
        path.unlink()

    counts = {dataset: 0 for dataset in EXPECTED}
    rows = []
    for name in MANIFESTS:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing paper Experiment IV manifest: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        datasets = {
            str(value).lower()
            for value in payload.get("supported_datasets", [])
        }
        tasks = {
            str(value).lower() for value in payload.get("supported_tasks", [])
        }
        native = sorted(datasets & set(EXPECTED))
        if len(native) != 1 or "sgdet" not in tasks:
            raise RuntimeError(
                f"Manifest is not one native SGDet run: {path}"
            )
        dataset = native[0]
        counts[dataset] += 1
        rows.append({
            "manifest": name,
            "dataset": dataset,
            "model": payload.get("name"),
            "family": payload.get("architecture_family"),
        })
        shutil.copy2(path, output / name)
    if counts != EXPECTED:
        raise RuntimeError(
            f"Experiment IV set mismatch: observed={counts} expected={EXPECTED}"
        )
    report = {
        "status": "ready",
        "protocol": "paper_experiment4_native_sgdet_v1",
        "counts": counts,
        "runs": rows,
        "manifest_dir": str(output),
    }
    destination = output.parent / "paper_experiment4_native_sgdet.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[READY] Experiment IV manifests: {destination}")


if __name__ == "__main__":
    main()
