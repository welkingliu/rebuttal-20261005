#!/usr/bin/env python3
"""Report readiness of Causal Motifs-SUM and KERN VG assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _first(paths):
    return next((path for path in paths if path.is_file()), paths[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--report")
    parser.add_argument("--allow_missing_causal_sgcls", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    vg_dict = _first([
        root / "data/vg/v1.4/VG-SGG-dicts.json",
        root / "data/vg/VG-SGG-dicts.json",
        root / "data/vg/VG-SGG-dicts-with-attri.json",
    ])
    required = {
        "canonical_vg_dict": vg_dict,
        "kaihua_source": root / "external/official_repos/Scene-Graph-Benchmark.pytorch/.official_source.json",
        "kern_source": root / "external/official_repos/KERN/.official_source.json",
        "causal_predcls": root / "checkpoints/sgg/weights/causal_motifs_sum/vg/predcls/model_0030000.pth",
        "causal_sgcls": root / "checkpoints/sgg/weights/causal_motifs_sum/vg/sgcls/model_final.pth",
        "causal_sgdet": root / "checkpoints/sgg/weights/causal_motifs_sum/vg/sgdet/model_0028000.pth",
        "kern_sgcls_predcls": root / "checkpoints/sgg/weights/kern/vg/kern_sgcls_predcls.tar",
        "kern_sgdet": root / "checkpoints/sgg/weights/kern/vg/kern_sgdet.tar",
        "kern_sgcls_cache": root / "checkpoints/sgg/native_predictions/kern/vg/kern_sgcls.pkl",
        "kern_sgdet_cache": root / "checkpoints/sgg/native_predictions/kern/vg/kern_sgdet.pkl",
    }
    assets = {}
    failures = []
    for name, path in required.items():
        present = path.is_file()
        assets[name] = {
            "path": str(path),
            "present": present,
            "bytes": path.stat().st_size if present else 0,
        }
        optional = name == "causal_sgcls" and args.allow_missing_causal_sgcls
        if not present and not optional:
            failures.append(name)
    payload = {
        "schema": "legacy_vg_asset_readiness_v1",
        "status": "ready" if not failures else "not_ready",
        "assets": assets,
        "failures": failures,
        "causal_sgcls_url": "https://1drv.ms/u/s!AmRLLNf6bzcir9xyuLO_I8TSZ6kfyQ?e=Y5686s",
        "kern_cache_download_script": str(root / "scripts/download_kern_official_caches.sh"),
    }
    report = (
        Path(args.report).expanduser().resolve()
        if args.report else root / "artifacts/manifests/legacy_vg_assets.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"report={report}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
