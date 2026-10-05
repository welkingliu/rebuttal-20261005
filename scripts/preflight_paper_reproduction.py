#!/usr/bin/env python3
"""Check code, data, checkpoint, and runtime assets before paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


CAUSAL_MOTIFS_HASHES = {
    "checkpoints/sgg/weights/causal_motifs_sum/vg/predcls/model_0030000.pth":
        "57663ccf4c57ed8740e830afbbde8e5c4334577cc9883aebef9b1b73c9113ec0",
    "checkpoints/sgg/weights/causal_motifs_sum/vg/sgcls/model_final.pth":
        "467da372633dbd77720cd3e7e5cc056552b86d957dd0ef4d571757e0786fc674",
    "checkpoints/sgg/weights/causal_motifs_sum/vg/sgdet/model_0028000.pth":
        "f57891f578a04320d0078a7bda1ca3dc192eb847744e6082e1e085b39f530f20",
}
DATA_PATHS = {
    "vg_h5": (
        "data/vg/v1.4/VG-SGG.h5",
        "data/vg/v1.4/VG-SGG-with-attri.h5",
    ),
    "vg_dictionary": (
        "data/vg/v1.4/VG-SGG-dicts.json",
        "data/vg/v1.4/VG-SGG-dicts-with-attri.json",
    ),
    "psg_annotation": (
        "data/psg/psg_train_val.json",
    ),
    "gqa_validation": (
        "data/gqa/val_sceneGraphs.json",
        "data/gqa/sceneGraphs/val_sceneGraphs.json",
    ),
    "vrd_test": (
        "data/vrd/json_dataset/annotations_test.json",
    ),
    "oi_validation": (
        "data/openimages/open-images-v6/annotations/"
        "oidv6-validation-annotations-vrd.csv",
    ),
}
CODE_PATHS = (
    "sgg_core/experiments/experiment_1a.py",
    "sgg_core/experiments/experiment_1b.py",
    "sgg_core/experiments/experiment_3.py",
    "sgg_core/experiments/experiment_4.py",
    "sgg_core/experiments/experiment_5.py",
    "scripts/run_paper_experiment3_2gpu.sh",
    "scripts/run_paper_experiment4_native_2gpu.sh",
    "scripts/run_paper_experiment5_tde_motifs.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(root: Path, choices: tuple[str, ...]) -> Path:
    paths = [root / value for value in choices]
    return next((path for path in paths if path.is_file()), paths[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--verify_large_hashes",
        action="store_true",
        help="Read all three multi-GB Causal Motifs checkpoints.",
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else root / "artifacts/manifests/paper_reproduction_preflight.json"
    )
    checks: dict[str, dict] = {}
    for relative in CODE_PATHS:
        path = root / relative
        checks[f"code:{relative}"] = {"ok": path.is_file(), "path": str(path)}
    for name, choices in DATA_PATHS.items():
        path = _first(root, choices)
        checks[f"data:{name}"] = {"ok": path.is_file(), "path": str(path)}
    for module in ("numpy", "torch", "yaml", "PIL"):
        checks[f"python:{module}"] = {
            "ok": importlib.util.find_spec(module) is not None
        }
    for relative, expected in CAUSAL_MOTIFS_HASHES.items():
        path = root / relative
        observed = (
            _sha256(path) if path.is_file() and args.verify_large_hashes else None
        )
        checks[f"checkpoint:{relative}"] = {
            "ok": path.is_file() and (
                observed is None or observed == expected
            ),
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
        }
    source_markers = (
        "external/official_repos/PySGG/.official_source.json",
        "external/official_repos/OpenPSG/.official_source.json",
        "external/official_repos/Scene-Graph-Benchmark.pytorch/.official_source.json",
    )
    for relative in source_markers:
        path = root / relative
        checks[f"source:{relative}"] = {"ok": path.is_file(), "path": str(path)}

    failures = sorted(key for key, value in checks.items() if not value["ok"])
    report = {
        "schema": "grounded_sgg_paper_preflight_v1",
        "status": "ready" if not failures else "not_ready",
        "python": sys.executable,
        "project_root": str(root),
        "large_hashes_verified": args.verify_large_hashes,
        "checks": checks,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{report['status'].upper()}] Paper preflight: {report_path}")
    if failures:
        for failure in failures:
            print(f"  missing-or-invalid: {failure}")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
