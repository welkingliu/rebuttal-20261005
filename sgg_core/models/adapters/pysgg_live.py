"""PySGG bridge for GT-pair interventions and lightweight mitigation.

PySGG's pinned CUDA extension lives in a Python-3.8 environment, while the
benchmark runs in a modern Python environment. A persistent native worker owns
the frozen SGCls model; requests travel through ``/dev/shm``. The benchmark
process owns only two trainable, identity-initialised calibration heads.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn

from sgg_core.models.official_adapter import fingerprint_tensors


NUM_OBJECTS = 151
NUM_PREDICATES = 51


def _identity_linear(size: int) -> nn.Linear:
    layer = nn.Linear(size, size)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(size))
        layer.bias.zero_()
    return layer


class PySGGLiveAdapter(nn.Module):
    """One native SGCls worker plus cache-backed three-task inference."""

    def __init__(self, checkpoints: dict[str, str], device: torch.device,
                 config: dict[str, Any], diagnostic_task: str):
        super().__init__()
        self.device = torch.device(device)
        if str(diagnostic_task).lower() != "sgcls":
            raise ValueError("PySGG diagnostic_task must be sgcls")
        if set(checkpoints) != {"predcls", "sgcls", "sgdet"}:
            raise ValueError("PySGG live manifests require all three VG tasks")
        self.cache_root = Path(
            str(config["prediction_cache_root"])
        ).expanduser().resolve()
        self.test_cache_root = Path(
            str(config.get("test_prediction_cache_root", self.cache_root))
        ).expanduser().resolve()
        self._checkpoints = {
            task: Path(checkpoint).expanduser().resolve()
            for task, checkpoint in checkpoints.items()
        }
        self._validated_test_cache_tasks = set()
        self.metadata = json.loads(
            (self.cache_root / "metadata.json").read_text(encoding="utf-8")
        )
        for task, checkpoint in checkpoints.items():
            state = json.loads(
                (self.cache_root / f"state_{task}.json").read_text(encoding="utf-8")
            )
            if Path(state["checkpoint"]).resolve() != Path(checkpoint).resolve():
                raise RuntimeError(f"PySGG cache/checkpoint mismatch for task={task}")

        self.relation_calibrator = _identity_linear(NUM_PREDICATES).to(self.device)
        self.entity_calibrator = _identity_linear(NUM_OBJECTS).to(self.device)
        self.reported_parameter_count = int(config["official_parameter_count"]) + sum(
            parameter.numel() for parameter in self.grounding_parameters()
        )
        self._worker_timeout = float(config.get("worker_timeout_seconds", 1800))
        self._worker = None
        self._worker_log = None
        self._queue_dir = None
        self._cache_only = os.environ.get("SGG_PYSGG_CACHE_ONLY") == "1"
        if not self._cache_only:
            self._start_worker(
                python=Path(str(config["worker_python"])).expanduser().resolve(),
                script=Path(str(config["worker_script"])).expanduser().resolve(),
                source=Path(str(config["source_root"])).expanduser().resolve(),
                model_config=Path(
                    str(config["diagnostic_config"])
                ).expanduser().resolve(),
                checkpoint=Path(checkpoints["sgcls"]).expanduser().resolve(),
            )
        atexit.register(self.close)

    def _start_worker(self, *, python: Path, script: Path, source: Path,
                      model_config: Path, checkpoint: Path) -> None:
        for path in (python, script, source, model_config, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        queue_parent = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
        self._queue_dir = Path(tempfile.mkdtemp(
            prefix=f"sgg_pysgg_{os.getpid()}_", dir=queue_parent
        ))
        log_path = self._queue_dir / "worker.log"
        self._worker_log = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        legacy_runtime = script.resolve().parents[1] / "legacy_runtime"
        python_paths = [str(legacy_runtime), str(source)]
        environment["PYTHONPATH"] = os.pathsep.join(python_paths) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH") else ""
        )
        command = [
            str(python), str(script), "--source_root", str(source),
            "--config", str(model_config), "--checkpoint", str(checkpoint),
            "--queue_dir", str(self._queue_dir), "--device", str(self.device),
        ]
        self._worker = subprocess.Popen(
            command, cwd=source, env=environment,
            stdout=self._worker_log, stderr=subprocess.STDOUT,
        )
        ready = self._queue_dir / "READY.json"
        deadline = time.monotonic() + self._worker_timeout
        while not ready.is_file():
            returncode = self._worker.poll()
            if returncode is not None:
                self._worker_log.flush()
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(
                    f"PySGG worker exited with code {returncode}: {detail}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for PySGG worker: {log_path}")
            time.sleep(0.1)
        payload = json.loads(ready.read_text(encoding="utf-8"))
        if payload.get("object_score_semantics") != "full_refined_logits":
            raise RuntimeError(
                "PySGG worker must expose complete refined object logits"
            )
        if int(payload["parameter_count"]) + sum(
            parameter.numel() for parameter in self.grounding_parameters()
        ) != self.reported_parameter_count:
            raise RuntimeError("PySGG worker parameter-count mismatch")

    def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._queue_dir is not None:
            shutil.rmtree(self._queue_dir, ignore_errors=True)
            self._queue_dir = None

    @staticmethod
    def _image_id(batch: dict) -> str:
        value = batch.get("image_id", batch.get("img_id"))
        if isinstance(value, torch.Tensor):
            value = value.item()
        if value is None:
            raise KeyError("PySGG cache inference requires image_id")
        return str(value).replace("/", "_").replace("\\", "_")

    def _load_cache(self, batch: dict, task: str) -> dict[str, torch.Tensor]:
        task = str(task).lower()
        if task not in self._validated_test_cache_tasks:
            state_path = self.test_cache_root / f"state_{task}.json"
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"Formal test cache is incomplete for task={task}: {state_path}"
                )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_checkpoint = self._checkpoints.get(task)
            if expected_checkpoint is None:
                raise ValueError(f"Unknown PySGG test task: {task}")
            if Path(state["checkpoint"]).expanduser().resolve() != expected_checkpoint:
                raise RuntimeError(
                    f"Formal test cache/checkpoint mismatch for task={task}"
                )
            self._validated_test_cache_tasks.add(task)
        path = (
            self.test_cache_root / "predictions" / task
            / f"{self._image_id(batch)}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            output = {
                key: torch.from_numpy(np.asarray(payload[key])).to(self.device)
                for key in payload.files if key != "image_id"
            }
        output["pred_rel_pairs"] = output["pred_rel_pairs"].long()
        for key in (
            "pred_boxes", "pred_entity_scores", "pred_box_scores",
            "pred_rel_scores", "pred_masks",
        ):
            if key in output:
                output[key] = output[key].float()
        return output

    @staticmethod
    def _log_probabilities(scores: torch.Tensor) -> torch.Tensor:
        values = scores.float()
        if values.numel() == 0:
            return values
        if bool((values >= 0).all()) and bool((values <= 1.0 + 1e-5).all()):
            return values.clamp_min(1e-8).log()
        return values

    def _calibrate(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calibrated = dict(output)
        if "pred_rel_scores" in calibrated:
            calibrated["pred_rel_scores"] = self.relation_calibrator(
                self._log_probabilities(calibrated["pred_rel_scores"])
            )
        if "pred_entity_scores" in calibrated:
            calibrated["pred_entity_scores"] = self.entity_calibrator(
                self._log_probabilities(calibrated["pred_entity_scores"])
            )
        return calibrated

    def _live_raw(
        self, batch: dict, *, require_gt_pairs: bool = False
    ) -> dict[str, torch.Tensor]:
        if self._cache_only:
            raise RuntimeError(
                "Live PySGG inference is disabled in cache-only test mode"
            )
        if self._queue_dir is None or self._worker is None:
            raise RuntimeError("PySGG worker is closed")
        image = batch.get("image")
        required = ("boxes", "entity_labels", "rel_pairs", "rel_labels")
        if not isinstance(image, torch.Tensor) or any(
            not isinstance(batch.get(key), torch.Tensor) for key in required
        ):
            raise KeyError("PySGG live inference requires image and VG annotations")
        if image.ndim == 4:
            if image.size(0) != 1:
                raise ValueError("PySGG live inference requires image batch_size=1")
            image = image[0]
        request_id = uuid.uuid4().hex
        input_path = self._queue_dir / f"{request_id}.input.npz"
        request_path = self._queue_dir / f"{request_id}.request.json"
        output_path = self._queue_dir / f"{request_id}.output.npz"
        done_path = self._queue_dir / f"{request_id}.done.json"
        error_path = self._queue_dir / f"{request_id}.error.json"
        np.savez(
            input_path,
            image=image.detach().float().cpu().numpy(),
            boxes=batch["boxes"].detach().float().cpu().numpy(),
            entity_labels=batch["entity_labels"].detach().long().cpu().numpy(),
            rel_pairs=batch["rel_pairs"].detach().long().cpu().numpy(),
            rel_labels=batch["rel_labels"].detach().long().cpu().numpy(),
            require_gt_pairs=np.asarray(
                int(require_gt_pairs), dtype=np.int8
            ),
        )
        temporary = request_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"request_id": request_id}) + "\n")
        os.replace(temporary, request_path)
        deadline = time.monotonic() + self._worker_timeout
        try:
            while not done_path.is_file():
                if error_path.is_file():
                    detail = json.loads(error_path.read_text(encoding="utf-8"))
                    raise RuntimeError(f"PySGG worker request failed: {detail}")
                returncode = self._worker.poll()
                if returncode is not None:
                    raise RuntimeError(f"PySGG worker exited with code {returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"PySGG request timed out: {request_id}")
                time.sleep(0.02)
            with np.load(output_path, allow_pickle=False) as payload:
                return {
                    "pred_rel_scores": torch.from_numpy(
                        np.asarray(payload["pred_rel_scores"])
                    ).to(self.device).float(),
                    "pred_entity_scores": torch.from_numpy(
                        np.asarray(payload["pred_entity_scores"])
                    ).to(self.device).float(),
                    "pred_rel_pairs": torch.from_numpy(
                        np.asarray(payload["pred_rel_pairs"])
                    ).to(self.device).long(),
                }
        finally:
            for path in (input_path, request_path, output_path, done_path, error_path):
                path.unlink(missing_ok=True)

    def predict(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._calibrate(
            self._live_raw(batch, require_gt_pairs=True)
        )

    def predict_scene_graph(self, batch: dict, task: str) -> dict[str, torch.Tensor]:
        return self._calibrate(self._load_cache(batch, str(task).lower()))

    def predict_scene_graph_tasks(self, batch: dict, tasks) -> dict:
        return {task: self.predict_scene_graph(batch, task) for task in tasks}

    def extract_node_features(self, batch: dict) -> torch.Tensor:
        return self._log_probabilities(
            self._live_raw(batch, require_gt_pairs=True)["pred_entity_scores"]
        )

    def diagnostic_input_fingerprint(self, batch: dict) -> str:
        return fingerprint_tensors(
            image=batch.get("image"), boxes=batch.get("boxes"),
            labels=batch.get("entity_labels"),
        )

    def grounding_parameters(self):
        return list(self.relation_calibrator.parameters()) + list(
            self.entity_calibrator.parameters()
        )

    def grounding_parameter_groups(self):
        return {
            "relation": list(self.relation_calibrator.parameters()),
            "object": list(self.entity_calibrator.parameters()),
        }

    def grounding_state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for prefix, module in (
            ("relation_calibrator", self.relation_calibrator),
            ("entity_calibrator", self.entity_calibrator),
        ):
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value
        return state

    def load_grounding_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for prefix, module in (
            ("relation_calibrator", self.relation_calibrator),
            ("entity_calibrator", self.entity_calibrator),
        ):
            values = {
                key[len(prefix) + 1:]: value for key, value in state.items()
                if key.startswith(prefix + ".")
            }
            module.load_state_dict(values, strict=True)

    def forward_grounding(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._calibrate(
            self._live_raw(batch, require_gt_pairs=True)
        )

    def train(self, mode: bool = True):
        self.relation_calibrator.train(mode)
        self.entity_calibrator.train(mode)
        return self


def create_adapter(*, checkpoint: str, checkpoints: dict[str, str], device,
                   config: dict[str, Any], diagnostic_task: str):
    del checkpoint
    return PySGGLiveAdapter(
        checkpoints=checkpoints, device=torch.device(device), config=config,
        diagnostic_task=diagnostic_task,
    )
