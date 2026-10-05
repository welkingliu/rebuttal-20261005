#!/usr/bin/env python3
"""Run one request through the cross-environment PySGG worker transport."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.data.data_utils import build_vg_test_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--worker_python", default=os.environ.get(
        "PYSGG_PYTHON", "python3"
    ))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    source = root / "external/official_repos/PySGG"
    config = Path(
        args.config or root / "configs/pysgg_vg_tritask/bgnn_sgdet.yaml"
    ).expanduser().resolve()
    checkpoint = Path(
        args.checkpoint or root / "checkpoints/sgg/weights/pysgg/vg/bgnn_vg.pth"
    ).expanduser().resolve()
    worker_python = Path(args.worker_python).expanduser().resolve()
    worker_script = root / "scripts/pysgg_live_worker.py"
    for path in (source, config, checkpoint, worker_python, worker_script):
        if not path.exists():
            raise FileNotFoundError(path)

    queue_parent = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    queue = Path(tempfile.mkdtemp(prefix="sgg_worker_smoke_", dir=queue_parent))
    log_path = queue / "worker.log"
    log = log_path.open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((
        str(root / "legacy_runtime"), str(source),
    )) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.Popen([
        str(worker_python), str(worker_script),
        "--source_root", str(source), "--config", str(config),
        "--checkpoint", str(checkpoint), "--queue_dir", str(queue),
        "--device", args.device,
    ], cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + args.timeout
        ready = queue / "READY.json"
        while not ready.is_file():
            if process.poll() is not None:
                log.flush()
                raise RuntimeError(log_path.read_text(errors="replace")[-4000:])
            if time.monotonic() >= deadline:
                raise TimeoutError("worker startup timed out")
            time.sleep(0.1)

        vg = root / "data/vg/v1.4"
        if not vg.is_dir():
            vg = root / "data/vg"
        batch = next(iter(build_vg_test_loader(
            str(vg), num_samples=1, split=2, include_proxy_features=False,
            require_relations=True, include_raw_images=True,
        )))
        request_id = "smoke"
        np.savez(
            queue / f"{request_id}.input.npz",
            image=batch["image"].float().cpu().numpy(),
            boxes=batch["boxes"].float().cpu().numpy(),
            entity_labels=batch["entity_labels"].long().cpu().numpy(),
            rel_pairs=batch["rel_pairs"].long().cpu().numpy(),
            rel_labels=batch["rel_labels"].long().cpu().numpy(),
        )
        (queue / f"{request_id}.request.json").write_text(
            json.dumps({"request_id": request_id}) + "\n"
        )
        done = queue / f"{request_id}.done.json"
        error = queue / f"{request_id}.error.json"
        while not done.is_file():
            if error.is_file():
                raise RuntimeError(error.read_text(encoding="utf-8"))
            if process.poll() is not None:
                raise RuntimeError("worker exited during request")
            if time.monotonic() >= deadline:
                raise TimeoutError("worker request timed out")
            time.sleep(0.02)
        with np.load(queue / f"{request_id}.output.npz", allow_pickle=False) as payload:
            relation_rows = int(payload["pred_rel_scores"].shape[0])
            entity_rows = int(payload["pred_entity_scores"].shape[0])
        if relation_rows != int(batch["rel_pairs"].shape[0]):
            raise RuntimeError("GT-pair output alignment failed")
        report = {
            "schema": "pysgg_worker_transport_smoke_v1",
            "status": "ready",
            "image_id": batch["image_id"],
            "relation_rows": relation_rows,
            "entity_rows": entity_rows,
            "worker": json.loads(ready.read_text(encoding="utf-8")),
            "config": str(config),
            "checkpoint": str(checkpoint),
        }
        report_path = Path(
            args.report or root / "artifacts/manifests/pysgg_worker_transport_smoke.json"
        ).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print("report=" + str(report_path))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log.close()
        shutil.rmtree(queue, ignore_errors=True)


if __name__ == "__main__":
    main()
