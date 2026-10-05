"""Small NumPy-only SDK for exporting official SGG predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_SCHEMA = "sgg_official_prediction_cache_v1"


class OfficialPredictionCacheWriter:
    """Write one versioned cache from an official repository environment."""

    def __init__(self, root, *, model_name, architecture_family,
                 source_commit, parameter_count, checkpoint_sha256_by_task,
                 dataset, ontology_id, split, tasks):
        self.root = Path(root).expanduser().resolve()
        self.tasks = tuple(str(task).lower() for task in tasks)
        self.metadata = {
            "schema": CACHE_SCHEMA,
            "model_name": str(model_name),
            "architecture_family": str(architecture_family),
            "source_commit": str(source_commit),
            "parameter_count": int(parameter_count),
            "checkpoint_sha256_by_task": dict(checkpoint_sha256_by_task),
            "dataset": str(dataset).lower(),
            "ontology_id": str(ontology_id),
            "split": str(split),
            "tasks": list(self.tasks),
            "image_ids": [],
            "images_by_task": {task: 0 for task in self.tasks},
        }
        for task in self.tasks:
            (self.root / "predictions" / task).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_image_id(value) -> str:
        return str(value).replace("/", "_").replace("\\", "_")

    def add(self, task, image_id, *, pred_rel_pairs, pred_rel_scores,
            pred_boxes=None, pred_entity_scores=None, pred_box_scores=None,
            pred_masks=None, node_features=None):
        task = str(task).lower()
        if task not in self.tasks:
            raise ValueError(f"Task {task} was not declared for this cache")
        arrays = {
            "pred_rel_pairs": np.asarray(pred_rel_pairs, dtype=np.int64),
            "pred_rel_scores": np.asarray(pred_rel_scores, dtype=np.float32),
        }
        optional = {
            "pred_boxes": pred_boxes,
            "pred_entity_scores": pred_entity_scores,
            "pred_box_scores": pred_box_scores,
            "pred_masks": pred_masks,
            "node_features": node_features,
        }
        for key, value in optional.items():
            if value is not None:
                arrays[key] = np.asarray(value)
        if arrays["pred_rel_pairs"].ndim != 2 or arrays["pred_rel_pairs"].shape[1] != 2:
            raise ValueError("pred_rel_pairs must have shape [M,2]")
        if (
            arrays["pred_rel_scores"].ndim != 2
            or arrays["pred_rel_scores"].shape[0] != arrays["pred_rel_pairs"].shape[0]
        ):
            raise ValueError("pred_rel_scores must have shape [M,C_rel]")
        safe_id = self._safe_image_id(image_id)
        np.savez_compressed(
            self.root / "predictions" / task / f"{safe_id}.npz", **arrays
        )
        image_id = str(image_id)
        if image_id not in self.metadata["image_ids"]:
            self.metadata["image_ids"].append(image_id)
        self.metadata["images_by_task"][task] += 1

    def finalize(self):
        self.metadata["image_ids"] = sorted(self.metadata["image_ids"])
        path = self.root / "metadata.json"
        path.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        return path
