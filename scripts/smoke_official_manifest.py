#!/usr/bin/env python3
"""Load one official manifest and run one standard SGG image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sgg_core.models.official_adapter import OfficialSGGAdapter
from sgg_core.protocol import build_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vg_root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = OfficialSGGAdapter(args.manifest, device=args.device)
    _, loader = build_loaders(
        dataset="vg",
        data_root=args.vg_root,
        train_samples=1,
        eval_samples=1,
    )
    batch = next(iter(loader))
    device_batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    prediction = model.predict_scene_graph(device_batch, task="sgdet")
    summary = {
        "model": model.name,
        "family": model.architecture_family,
        "image_id": batch.get("image_id"),
        "ontology_id": batch.get("ontology_id"),
        "pred_boxes": list(prediction["pred_boxes"].shape),
        "pred_entity_scores": list(prediction["pred_entity_scores"].shape),
        "pred_rel_pairs": list(prediction["pred_rel_pairs"].shape),
        "pred_rel_scores": list(prediction["pred_rel_scores"].shape),
        "finite": all(
            bool(torch.isfinite(value).all())
            for value in prediction.values() if isinstance(value, torch.Tensor)
        ),
        "input_fingerprint": model.diagnostic_input_fingerprint(device_batch),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
