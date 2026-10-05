"""Strict offline prediction cache for legacy official SGG environments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


CACHE_SCHEMA = "sgg_official_prediction_cache_v1"
REQUIRED_ARRAYS = ("pred_rel_pairs", "pred_rel_scores")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialPredictionCacheModel:
    """Read official predictions exported in each model's native environment."""

    def __init__(self, manifest: dict, device="cpu"):
        self.manifest = manifest
        self.device = torch.device(device)
        config = manifest.get("config", {})
        self.relation_score_mode = str(
            config.get("relation_score_mode", "categorical")
        ).lower()
        if self.relation_score_mode not in {
            "categorical", "independent_probabilities",
        }:
            raise ValueError(
                f"Unsupported prediction-cache relation_score_mode: "
                f"{self.relation_score_mode}"
            )
        self.cache_root = Path(str(config.get("prediction_cache_root", ""))).expanduser().resolve()
        metadata_path = self.cache_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Prediction-cache metadata missing: {metadata_path}")
        expected_metadata_sha = str(config.get("prediction_cache_metadata_sha256", ""))
        actual_metadata_sha = sha256_file(metadata_path)
        if not expected_metadata_sha or actual_metadata_sha != expected_metadata_sha:
            raise RuntimeError(
                "Prediction-cache metadata SHA256 mismatch: "
                f"expected={expected_metadata_sha} actual={actual_metadata_sha}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata()

    def _validate_metadata(self):
        metadata = self.metadata
        if metadata.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"Unsupported prediction-cache schema: {metadata.get('schema')}")
        checks = {
            "model_name": self.manifest["name"],
            "architecture_family": self.manifest["architecture_family"],
            "source_commit": self.manifest["source_commit"],
        }
        for key, expected in checks.items():
            if str(metadata.get(key)) != str(expected):
                raise ValueError(
                    f"Prediction-cache {key} mismatch: {metadata.get(key)!r} != {expected!r}"
                )
        cached_tasks = set(metadata.get("tasks", []))
        if cached_tasks != set(self.manifest["supported_tasks"]):
            raise ValueError(
                f"Prediction-cache task mismatch: {sorted(cached_tasks)} != "
                f"{sorted(self.manifest['supported_tasks'])}"
            )
        checkpoint_sha = metadata.get("checkpoint_sha256_by_task", {})
        for task, spec in self.manifest["checkpoints"].items():
            if str(checkpoint_sha.get(task, "")).lower() != str(spec["sha256"]).lower():
                raise ValueError(f"Prediction cache was not exported from {task} checkpoint SHA")
        self.reported_parameter_count = int(metadata.get("parameter_count", -1))
        if self.reported_parameter_count != int(self.manifest["parameter_count"]):
            raise ValueError("Prediction-cache parameter count does not match manifest")
        cached_by_task = metadata.get("parameter_count_by_task", {})
        declared_by_task = self.manifest.get("parameter_count_by_task", {})
        if declared_by_task and {
            str(key): int(value) for key, value in cached_by_task.items()
        } != {
            str(key): int(value) for key, value in declared_by_task.items()
        }:
            raise ValueError(
                "Prediction-cache task parameter counts do not match manifest"
            )

    @staticmethod
    def _image_id(batch: dict) -> str:
        value = batch.get("image_id", batch.get("img_id"))
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Prediction cache requires one image per batch")
            value = value.item()
        if value is None:
            raise KeyError("Prediction cache requires batch['image_id']")
        return str(value).replace("/", "_").replace("\\", "_")

    def _path(self, task: str, image_id: str) -> Path:
        return self.cache_root / "predictions" / task / f"{image_id}.npz"

    def _load(self, batch: dict, task: str) -> dict:
        image_id = self._image_id(batch)
        path = self._path(task, image_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"Official prediction missing for task={task} image_id={image_id}: {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            missing = [name for name in REQUIRED_ARRAYS if name not in payload]
            if missing:
                raise KeyError(f"Prediction cache {path} lacks arrays: {missing}")
            output = {
                key: torch.from_numpy(np.asarray(payload[key])).to(self.device)
                for key in payload.files
                if key not in {"image_id"}
            }
        output["pred_rel_pairs"] = output["pred_rel_pairs"].long()
        output["pred_rel_scores"] = output["pred_rel_scores"].float()
        output["pred_rel_score_mode"] = self.relation_score_mode
        for key in ("pred_boxes", "pred_entity_scores", "pred_box_scores", "pred_masks"):
            if key in output:
                output[key] = output[key].float()
        return output

    def predict_scene_graph(self, batch: dict, task: str) -> dict:
        return self._load(batch, str(task).lower())

    def predict_scene_graph_tasks(self, batch: dict, tasks) -> dict:
        return {task: self._load(batch, str(task).lower()) for task in tasks}

    def predict(self, batch: dict) -> dict:
        return self._load(batch, self.manifest.get("diagnostic_task", "sgdet"))

    def extract_node_features(self, batch: dict) -> torch.Tensor:
        output = self._load(batch, self.manifest.get("diagnostic_task", "sgdet"))
        features = output.get("node_features")
        if features is None:
            raise NotImplementedError(
                "This prediction cache has no node_features; do not report feature diagnostics"
            )
        return features.float()

    def diagnostic_input_fingerprint(self, batch: dict) -> str:
        image_id = self._image_id(batch)
        return hashlib.sha256(
            f"prediction-cache:{self.cache_root}:{image_id}".encode("utf-8")
        ).hexdigest()

    def parameters(self):
        return iter(())

    def named_parameters(self):
        return iter(())

    def eval(self):
        return self

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Offline prediction caches cannot be trained")
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self


def create_adapter(*, checkpoint, checkpoints, device, config,
                   diagnostic_task, manifest=None):
    del checkpoint, checkpoints, config, diagnostic_task
    if manifest is None:
        raise ValueError("Prediction-cache factory requires the complete manifest")
    return OfficialPredictionCacheModel(manifest, device=device)
