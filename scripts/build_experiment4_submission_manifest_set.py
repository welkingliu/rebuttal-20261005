#!/usr/bin/env python3
"""Build the pinned Experiment-IV manifest set, excluding known failed releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


MANIFESTS = (
    "egtr_oi.json",
    "sgtr_oi.json",
    "openpsg_motifs_psg.json",
    "openpsg_vctree_psg.json",
    "pysgg_motifs_vg_tritask.json",
    "pysgg_vctree_vg_tritask.json",
    "pysgg_transformer_vg_tritask.json",
    "pysgg_bgnn_vg_tritask.json",
    "pysgg_tde_motifs_vg_tritask.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    source = root / "checkpoints/sgg/manifests"
    output = Path(
        args.output_dir or root / "artifacts/manifests/experiment4_submission_set"
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stale = {path.name for path in output.glob("*.json")} - set(MANIFESTS)
    for name in stale:
        (output / name).unlink()
    families = set()
    models = []
    for name in MANIFESTS:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        family = str(payload.get("architecture_family", "")).strip()
        if not family:
            raise RuntimeError("Manifest has no architecture family: " + str(path))
        shutil.copy2(path, output / name)
        families.add(family)
        models.append(payload["name"])
    core_vg = {
        json.loads((output / name).read_text())["architecture_family"]
        for name in MANIFESTS if name.startswith("pysgg_")
    }
    expected_core_vg = {
        "Neural Motifs", "VCTree", "TDE-Motifs", "BGNN", "SGG Transformer",
    }
    if core_vg != expected_core_vg:
        raise RuntimeError(
            "VG core family mismatch: "
            f"observed={sorted(core_vg)} expected={sorted(expected_core_vg)}"
        )
    report = {
        "status": "ready",
        "manifests": list(MANIFESTS),
        "models": models,
        "architecture_families": sorted(families),
        "core_vg_families": sorted(core_vg),
        "external_sgdet_contract": {"oi": 2, "psg": 2},
        "removed_stale_manifests": sorted(stale),
        "known_exclusion": {
            "manifests": [
                "openpsg_psgtr_psg.json", "egtr_vg.json", "reltr_vg.json",
                "sgtr_vg.json", "pysgg_bgnn_vg_sgdet.json",
            ],
            "reason": (
                "outside the converged five-family VG tri-task table or failed "
                "the pinned native reference protocol"
            ),
        },
    }
    report_path = output.parent / (output.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("manifest_dir=" + str(output))
    print("report=" + str(report_path))


if __name__ == "__main__":
    main()
