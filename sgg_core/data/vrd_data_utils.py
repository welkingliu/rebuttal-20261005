"""
VRD loader for Experiment IV.

Converts Stanford Visual Relationship Detection annotations into the diagnostic
batch schema shared by VG, OpenImages, GQA, and PSG.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sgg_core.data.data_utils import (
    ROI_FEAT_DIM, _collate_fn, build_proxy_features, load_image_tensor,
)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_to_xyxy(bbox) -> np.ndarray:
    """Convert Stanford VRD [ymin, ymax, xmin, xmax] boxes to xyxy."""
    ymin, ymax, xmin, xmax = [float(v) for v in bbox[:4]]
    return np.asarray([xmin, ymin, xmax, ymax], dtype=np.float32)


def _bbox_to_xyxy_norm(bbox, image_width: float, image_height: float) -> np.ndarray:
    box = _bbox_to_xyxy(bbox)
    scale = np.asarray([
        max(float(image_width), 1.0), max(float(image_height), 1.0),
        max(float(image_width), 1.0), max(float(image_height), 1.0),
    ], dtype=np.float32)
    return np.clip(box / scale, 0.0, 1.0)


def _dedupe_object(
    obj: dict,
    boxes: List[np.ndarray],
    labels: List[int],
    key_to_idx: Dict[Tuple[int, Tuple[float, ...]], int],
    image_width: float,
    image_height: float,
) -> int:
    label = max(1, int(obj.get("category", 0)) + 1)
    box = _bbox_to_xyxy_norm(
        obj.get("bbox", [0, 1, 0, 1]), image_width, image_height,
    )
    key = (label, tuple(np.round(box, 3).tolist()))
    if key not in key_to_idx:
        key_to_idx[key] = len(boxes)
        boxes.append(box)
        labels.append(label)
    return key_to_idx[key]


def _annotation_extent(relations: list) -> Tuple[float, float]:
    raw_boxes = [
        _bbox_to_xyxy(entity.get("bbox", [0, 1, 0, 1]))
        for rel in relations
        for entity in (rel.get("subject", {}), rel.get("object", {}))
    ]
    if not raw_boxes:
        return 1.0, 1.0
    stacked = np.stack(raw_boxes)
    return max(float(stacked[:, 2].max()), 1.0), max(float(stacked[:, 3].max()), 1.0)


def _scene_to_batch(image_id: str, relations: list, image_path: Path | None,
                    ontology_id: str, num_objects: int, num_predicates: int,
                    include_proxy_features: bool = True,
                    include_raw_images: bool = True) -> dict:
    image_width = image_height = None
    if image_path is not None:
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, ImportError):
            image_path = None
    if image_width is None or image_height is None:
        image_width, image_height = _annotation_extent(relations)
        geometry_source = "annotation_extent"
    else:
        geometry_source = "image_dimensions"

    boxes: List[np.ndarray] = []
    labels: List[int] = []
    key_to_idx: Dict[Tuple[int, Tuple[float, ...]], int] = {}
    rel_pairs, rel_labels = [], []

    for rel in relations:
        s_idx = _dedupe_object(
            rel.get("subject", {}), boxes, labels, key_to_idx,
            image_width, image_height,
        )
        o_idx = _dedupe_object(
            rel.get("object", {}), boxes, labels, key_to_idx,
            image_width, image_height,
        )
        pred = max(1, int(rel.get("predicate", 0)) + 1)
        if s_idx != o_idx:
            rel_pairs.append([s_idx, o_idx])
            rel_labels.append(pred)

    while len(boxes) < 2:
        boxes.append(np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32))
        labels.append(len(labels) + 1)
    if not rel_pairs:
        rel_pairs = [[0, 1]]
        rel_labels = [1]

    boxes_arr = np.stack(boxes).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    rel_pairs_arr = np.asarray(rel_pairs, dtype=np.int64)
    rel_labels_arr = np.asarray(rel_labels, dtype=np.int64)

    adj = np.zeros((len(labels_arr), len(labels_arr)), dtype=np.float32)
    for s, o in rel_pairs_arr:
        adj[int(s), int(o)] = 1.0
        adj[int(o), int(s)] = 1.0

    result = {
        "boxes": torch.from_numpy(boxes_arr),
        "entity_labels": torch.from_numpy(labels_arr),
        "rel_pairs": torch.from_numpy(rel_pairs_arr),
        "rel_labels": torch.from_numpy(rel_labels_arr),
        "graph_adj": torch.from_numpy(adj),
        "num_nodes": len(labels_arr),
        "image_id": image_id,
        "feature_source": "annotation_only",
        "dataset": "vrd",
        "ontology_id": ontology_id,
        "num_entity_classes": num_objects + 1,
        "num_predicate_classes": num_predicates + 1,
        "geometry_source": geometry_source,
    }
    if include_proxy_features:
        vis, uni, feat_src = build_proxy_features(
            boxes_arr,
            labels_arr,
            rel_pairs_arr,
            out_dim=ROI_FEAT_DIM,
            img_path=image_path,
            img_w=image_width,
            img_h=image_height,
        )
        result["visual_features"] = torch.from_numpy(vis)
        result["union_features"] = torch.from_numpy(uni)
        result["feature_source"] = feat_src
    if image_path is not None:
        result["image_path"] = str(image_path)
    if include_raw_images:
        image_tensor = load_image_tensor(image_path)
        if image_tensor is not None:
            result["image"] = image_tensor
    return result


class VRDDataset(Dataset):
    def __init__(self, vrd_root: str, split: str = "test", num_samples: int = 500,
                 include_proxy_features: bool = True,
                 include_raw_images: bool = True):
        self.vrd_root = Path(vrd_root)
        self.split = split
        self.include_proxy_features = bool(include_proxy_features)
        self.include_raw_images = bool(include_raw_images)
        ann_path = self.vrd_root / "json_dataset" / f"annotations_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"VRD annotation not found: {ann_path}")

        raw = _load_json(ann_path)
        objects = _load_json(self.vrd_root / "json_dataset" / "objects.json")
        predicates = _load_json(self.vrd_root / "json_dataset" / "predicates.json")
        self.objects = list(objects)
        self.predicates = list(predicates)
        canonical = json.dumps(
            {"objects": self.objects, "predicates": self.predicates},
            ensure_ascii=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        self.ontology_id = f"vrd:{digest}"
        self.num_entity_classes = len(self.objects) + 1
        self.num_predicate_classes = len(self.predicates) + 1
        self.sgg_dict = {
            "idx_to_label": {str(i + 1): name for i, name in enumerate(self.objects)},
            "idx_to_predicate": {str(i + 1): name for i, name in enumerate(self.predicates)},
        }
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        parsed = [(str(image_id), rels) for image_id, rels in items if rels]
        self.items = (
            parsed if num_samples is None or int(num_samples) <= 0
            else parsed[:int(num_samples)]
        )
        if not self.items:
            raise ValueError(f"No VRD relations parsed from {ann_path}")
        print(f"  [VRD] split={split} graphs={len(self.items)} root={self.vrd_root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_id, relations = self.items[idx]
        image_path = self._find_image(image_id)
        return _scene_to_batch(
            image_id, relations, image_path, self.ontology_id,
            len(self.objects), len(self.predicates), self.include_proxy_features,
            self.include_raw_images,
        )

    def _find_image(self, image_id: str):
        directories = [
            self.vrd_root / "sg_dataset",
            self.vrd_root / "sg_dataset" / f"sg_{self.split}_images",
            self.vrd_root / "images",
            self.vrd_root,
        ]
        exact = [directory / image_id for directory in directories]
        resolved = next((path for path in exact if path.is_file()), None)
        if resolved is not None:
            return resolved
        stem = Path(image_id).stem
        for directory in directories:
            for extension in (".jpg", ".jpeg", ".png", ".gif"):
                candidate = directory / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate
        return None


def build_vrd_loader(vrd_root: str, split: str = "test", num_samples: int = 500,
                     batch_size: int = 1,
                     include_proxy_features: bool = True,
                     include_raw_images: bool = True) -> DataLoader:
    dataset = VRDDataset(
        vrd_root=vrd_root, split=split, num_samples=num_samples,
        include_proxy_features=include_proxy_features,
        include_raw_images=include_raw_images,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)


def check_vrd_files(vrd_root: str, split: str = "test") -> dict:
    root = Path(vrd_root) if vrd_root else Path("")
    required = [
        root / "json_dataset" / f"annotations_{split}.json",
        root / "json_dataset" / "objects.json",
        root / "json_dataset" / "predicates.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {"ok": not missing, "missing": missing}
