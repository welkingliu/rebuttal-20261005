#!/usr/bin/env python3
"""Evaluate an Experiment V mitigation checkpoint without retraining it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sgg_core.audits.standard_sgg_eval import StandardSGGAudit
from sgg_core.data.data_utils import build_vg_test_loader
from sgg_core.mitigation.run_mitigation import _json_default, _training_metadata
from sgg_core.models.official_adapter import OfficialSGGAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Mitigation checkpoint. Omit it to evaluate the identity baseline.",
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=26446)
    parser.add_argument(
        "--tasks", nargs="+", choices=("predcls", "sgcls", "sgdet"),
        default=("predcls", "sgcls", "sgdet"),
    )
    parser.add_argument(
        "--recall_ks", nargs="+", type=int,
        default=(1, 5, 10, 20, 50, 100),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache_only", action="store_true",
        help="Skip the native live worker; valid for prediction-cache tests only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "checkpoint_test_start",
        "tasks": args.tasks,
        "test_samples": args.test_samples,
    }, sort_keys=True), flush=True)

    if args.cache_only:
        os.environ["SGG_PYSGG_CACHE_ONLY"] = "1"
    model = OfficialSGGAdapter(args.manifest, device=args.device)
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint else None
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=args.device,
            weights_only=False,
        )
        model.load_grounding_state_dict(checkpoint["grounding_state_dict"])
    train_loader = build_vg_test_loader(
        args.data_root, args.train_samples, split=0
    )
    test_loader = build_vg_test_loader(
        args.data_root, args.test_samples, split=2
    )
    seen_triplets, _ = _training_metadata(train_loader)
    result = StandardSGGAudit(
        ks=args.recall_ks,
        tasks=args.tasks,
        device=args.device,
        seen_triplets=seen_triplets,
    ).run({model.name: model}, test_loader)[model.name]
    payload = {
        "schema": "experiment5_checkpoint_test_v1",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "tasks": list(args.tasks),
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "evaluation": result,
    }
    output_path.write_text(
        json.dumps(
            payload, indent=2, default=_json_default, allow_nan=True
        ) + "\n",
        encoding="utf-8",
    )
    statuses = {
        task: result.get("tasks", {}).get(task, {}).get("status")
        for task in args.tasks
    }
    print(json.dumps({
        "event": "checkpoint_test_complete",
        "output": str(output_path),
        "status": result.get("status"),
        "task_statuses": statuses,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
