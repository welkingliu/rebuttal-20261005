#!/usr/bin/env python3
"""Instantiate each required live adapter and run one real VG request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.audits.pair_audit import VisualPerturbation
from sgg_core.data.data_utils import build_vg_test_loader
from sgg_core.models.official_adapter import OfficialSGGAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--models", nargs="+", default=["motifs", "bgnn", "transformer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    vg = root / "data/vg/v1.4"
    if not vg.is_dir():
        vg = root / "data/vg"
    loader = build_vg_test_loader(
        str(vg), num_samples=1, split=2, include_proxy_features=True,
        require_relations=True, include_raw_images=True,
    )
    batch = next(iter(loader))
    moved = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    rows = []
    for model_key in tuple(dict.fromkeys(args.models)):
        manifest = root / "checkpoints/sgg/manifests" / f"pysgg_{model_key}_vg_live.json"
        adapter = OfficialSGGAdapter(str(manifest), device=args.device)
        try:
            clean_fingerprint = adapter.diagnostic_input_fingerprint(moved)
            perturbed = VisualPerturbation().inject_visual_noise(
                moved, strength=0.1, seed=17
            )
            perturbed_fingerprint = adapter.diagnostic_input_fingerprint(perturbed)
            if clean_fingerprint == perturbed_fingerprint:
                raise RuntimeError("raw-image perturbation did not change fingerprint")
            live = adapter.predict(moved)
            sgcls = adapter.predict_scene_graph(moved, task="sgcls")
            sgdet = adapter.predict_scene_graph(moved, task="sgdet")
            if live["pred_rel_scores"].shape[0] != moved["rel_pairs"].shape[0]:
                raise RuntimeError("live GT-pair rows are not aligned")
            rows.append({
                "model": adapter.name,
                "family": adapter.architecture_family,
                "live_relations": int(live["pred_rel_scores"].shape[0]),
                "live_entities": int(live["pred_entity_scores"].shape[0]),
                "sgcls_relations": int(sgcls["pred_rel_scores"].shape[0]),
                "sgdet_relations": int(sgdet["pred_rel_scores"].shape[0]),
                "parameter_count": adapter.checkpoint_status["parameter_count"],
                "status": "ok",
            })
        finally:
            adapter.model.close()
    report = {
        "schema": "pysgg_live_adapter_smoke_v1",
        "status": "ready",
        "image_id": batch.get("image_id"),
        "models": rows,
    }
    report_path = Path(
        args.report or root / "artifacts/manifests/pysgg_live_adapter_smoke.json"
    ).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("report=" + str(report_path))


if __name__ == "__main__":
    main()
