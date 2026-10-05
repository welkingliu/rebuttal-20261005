"""
GQA and PSG loaders for Experiment IV.

Both datasets are converted into the same diagnostic batch schema used by VG
and OpenImages. Labels retain the complete dataset-local ontology. A model must
therefore provide an explicit dataset adapter instead of silently clipping
classes to the VG-150 head dimensions.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sgg_core.data.data_utils import (
    ROI_FEAT_DIM, build_proxy_features, load_image_tensor, _collate_fn,
)

def _norm(value) -> str:
    return str(value).strip().lower().replace("_", " ")


def _box_xywh(x, y, w, h, img_w=1, img_h=1) -> np.ndarray:
    img_w = max(float(img_w), 1.0)
    img_h = max(float(img_h), 1.0)
    arr = np.array([
        float(x) / img_w,
        float(y) / img_h,
        (float(x) + float(w)) / img_w,
        (float(y) + float(h)) / img_h,
    ], dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _box_from_annotation(annotation: dict, img_w=1, img_h=1) -> np.ndarray:
    bbox = annotation.get("bbox", [0, 0, 1, 1])[:4]
    mode = int(annotation.get("bbox_mode", 1))
    if mode == 0:  # Detectron2 BoxMode.XYXY_ABS
        x1, y1, x2, y2 = map(float, bbox)
        return np.clip(np.asarray([
            x1 / max(float(img_w), 1.0), y1 / max(float(img_h), 1.0),
            x2 / max(float(img_w), 1.0), y2 / max(float(img_h), 1.0),
        ], dtype=np.float32), 0.0, 1.0)
    return _box_xywh(*bbox, img_w, img_h)


def _build_vocab(names: Iterable[str]) -> Dict[str, int]:
    counts = Counter(_norm(n) for n in names if _norm(n))
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    return {name: i + 1 for i, name in enumerate(ordered)}


def _id(vocab: Dict[str, int], name: str) -> int:
    key = _norm(name)
    if key not in vocab:
        raise KeyError(f"Label '{key}' is missing from the declared ontology")
    return vocab[key]


def _ontology_id(dataset_name: str,
                 obj_vocab: Dict[str, int],
                 pred_vocab: Dict[str, int]) -> str:
    canonical = json.dumps(
        {"objects": obj_vocab, "predicates": pred_vocab},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_name}:{digest}"


def _scene_to_batch(
    image_id,
    node_labels: List[str],
    boxes: List[np.ndarray],
    rels: List[Tuple[int, int, str]],
    obj_vocab: Dict[str, int],
    pred_vocab: Dict[str, int],
    dataset_name: str,
    ontology_id: str,
    include_proxy_features: bool = True,
) -> dict:
    if len(node_labels) < 2:
        node_labels = node_labels + ["object"] * (2 - len(node_labels))
        boxes = boxes + [np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)] * (2 - len(boxes))

    entity_labels = np.array([_id(obj_vocab, n) for n in node_labels], dtype=np.int64)
    boxes_arr = np.stack(boxes).astype(np.float32)

    rel_pairs, rel_labels = [], []
    for s, o, pred in rels:
        if 0 <= s < len(node_labels) and 0 <= o < len(node_labels) and s != o:
            rel_pairs.append([s, o])
            rel_labels.append(_id(pred_vocab, pred))

    if rel_pairs:
        rel_pairs_arr = np.asarray(rel_pairs, dtype=np.int64)
        rel_labels_arr = np.asarray(rel_labels, dtype=np.int64)
    else:
        rel_pairs_arr = np.zeros((0, 2), dtype=np.int64)
        rel_labels_arr = np.zeros(0, dtype=np.int64)

    adj = np.zeros((len(node_labels), len(node_labels)), dtype=np.float32)
    for s, o in rel_pairs_arr:
        adj[int(s), int(o)] = 1.0
        adj[int(o), int(s)] = 1.0

    result = {
        "boxes": torch.from_numpy(boxes_arr),
        "entity_labels": torch.from_numpy(entity_labels),
        "rel_pairs": torch.from_numpy(rel_pairs_arr),
        "rel_labels": torch.from_numpy(rel_labels_arr),
        "graph_adj": torch.from_numpy(adj),
        "num_nodes": len(node_labels),
        "image_id": image_id,
        "feature_source": "annotation_only",
        "dataset": dataset_name,
        "ontology_id": ontology_id,
        "num_entity_classes": len(obj_vocab) + 1,
        "num_predicate_classes": len(pred_vocab) + 1,
    }
    if include_proxy_features:
        vis, uni, feat_src = build_proxy_features(
            boxes_arr,
            entity_labels,
            rel_pairs_arr,
            out_dim=ROI_FEAT_DIM,
        )
        result["visual_features"] = torch.from_numpy(vis)
        result["union_features"] = torch.from_numpy(uni)
        result["feature_source"] = feat_src
    return result


class GQASceneGraphDataset(Dataset):
    def __init__(self, scene_graph_path: str, num_samples: int = 500,
                 vocabulary_path: str | None = None,
                 image_root: str | None = None,
                 include_proxy_features: bool = True,
                 include_raw_images: bool = True):
        self.scene_graph_path = Path(scene_graph_path)
        self.image_root = Path(image_root) if image_root else None
        self.include_proxy_features = bool(include_proxy_features)
        self.include_raw_images = bool(include_raw_images)
        with open(self.scene_graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = list(data.items())
        sample_limit = None if num_samples is None or num_samples <= 0 else int(num_samples)
        obj_names, pred_names = [], []
        parsed = []
        skipped_missing_images = 0
        for image_id, graph in records:
            objects = graph.get("objects", {})
            id_to_idx = {str(oid): i for i, oid in enumerate(objects.keys())}
            width = graph.get("width", graph.get("image_width", 1))
            height = graph.get("height", graph.get("image_height", 1))
            nodes, boxes, rels = [], [], []
            for oid, obj in objects.items():
                name = _norm(obj.get("name", "object"))
                nodes.append(name)
                boxes.append(_box_xywh(obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1), width, height))
                obj_names.append(name)
                s_idx = id_to_idx[str(oid)]
                for rel in obj.get("relations", []):
                    o_id = str(rel.get("object", ""))
                    if o_id not in id_to_idx:
                        continue
                    pred = _norm(rel.get("name", "related to"))
                    rels.append((s_idx, id_to_idx[o_id], pred))
                    pred_names.append(pred)
            if rels:
                if self.image_root is not None:
                    image_candidates = [
                        self.image_root / f"{image_id}.jpg",
                        self.image_root / str(image_id),
                    ]
                    if not any(path.is_file() for path in image_candidates):
                        skipped_missing_images += 1
                        continue
                parsed.append((image_id, nodes, boxes, rels))
                if sample_limit is not None and len(parsed) >= sample_limit:
                    break

        if not parsed:
            raise ValueError(f"No GQA scene graphs parsed from {scene_graph_path}")

        vocabulary_data = data
        if vocabulary_path and Path(vocabulary_path).resolve() != self.scene_graph_path.resolve():
            with open(vocabulary_path, "r", encoding="utf-8") as f:
                vocabulary_data = json.load(f)
        vocab_objects, vocab_predicates = [], []
        for graph in vocabulary_data.values():
            for obj in graph.get("objects", {}).values():
                vocab_objects.append(_norm(obj.get("name", "object")))
                vocab_predicates.extend(
                    _norm(rel.get("name", "related to"))
                    for rel in obj.get("relations", [])
                )
        self.obj_vocab = _build_vocab(vocab_objects or obj_names)
        self.pred_vocab = _build_vocab(vocab_predicates or pred_names)
        self.ontology_id = _ontology_id("gqa", self.obj_vocab, self.pred_vocab)
        self.num_entity_classes = len(self.obj_vocab) + 1
        self.num_predicate_classes = len(self.pred_vocab) + 1
        self.sgg_dict = {
            "idx_to_label": {str(idx): name for name, idx in self.obj_vocab.items()},
            "idx_to_predicate": {str(idx): name for name, idx in self.pred_vocab.items()},
        }
        self.items = parsed
        print(f"  [GQA] graphs={len(self.items)} objects={len(self.obj_vocab)} "
              f"predicates={len(self.pred_vocab)} "
              f"skipped_missing_images={skipped_missing_images}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_id, nodes, boxes, rels = self.items[idx]
        result = _scene_to_batch(
            image_id, nodes, boxes, rels, self.obj_vocab, self.pred_vocab,
            "gqa", self.ontology_id, self.include_proxy_features,
        )
        if self.image_root:
            candidates = [
                self.image_root / f"{image_id}.jpg",
                self.image_root / str(image_id),
            ]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is not None:
                result["image_path"] = str(image_path)
            if self.include_raw_images:
                image = load_image_tensor(image_path)
                if image is not None:
                    result["image"] = image
        return result


class PSGSceneGraphDataset(Dataset):
    def __init__(self, annotation_path: str, num_samples: int = 500,
                 exclude_annotation_path: str | None = None,
                 image_root: str | None = None,
                 panoptic_root: str | None = None,
                 include_proxy_features: bool = True,
                 include_raw_images: bool = True):
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root) if image_root else None
        self.panoptic_root = Path(panoptic_root) if panoptic_root else None
        self.include_proxy_features = bool(include_proxy_features)
        self.include_raw_images = bool(include_raw_images)
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            records, meta = raw, {}
        else:
            records = raw.get("data", raw.get("annotations", []))
            meta = raw
        if isinstance(records, dict):
            records = list(records.values())
        excluded_ids = set()
        if exclude_annotation_path:
            with open(exclude_annotation_path, "r", encoding="utf-8") as f:
                excluded_raw = json.load(f)
            if isinstance(excluded_raw, list):
                excluded_records = excluded_raw
            else:
                excluded_records = excluded_raw.get(
                    "data", excluded_raw.get("annotations", [])
                )
            if isinstance(excluded_records, dict):
                excluded_records = list(excluded_records.values())
            excluded_ids = {
                str(record.get("image_id", record.get("coco_image_id")))
                for record in excluded_records
            }
            records = [
                record for record in records
                if str(record.get("image_id", record.get("coco_image_id")))
                not in excluded_ids
            ]

        predicate_classes = meta.get("predicate_classes", meta.get("predicate_categories", meta.get("rel_classes", [])))
        thing_classes = meta.get("thing_classes", [])
        stuff_classes = meta.get("stuff_classes", [])
        object_classes = meta.get("object_classes", thing_classes + stuff_classes)

        def seg_name(seg: dict) -> str:
            raw_name = seg.get("category_name", seg.get("name", seg.get("category", None)))
            if raw_name is None:
                cid = seg.get("category_id", "object")
                raw_name = object_classes[cid] if isinstance(cid, int) and 0 <= cid < len(object_classes) else cid
            return _norm(raw_name)

        def pred_name(pred) -> str:
            if isinstance(pred, int) and 0 <= pred < len(predicate_classes):
                return _norm(predicate_classes[pred])
            return _norm(pred)

        obj_names, pred_names = [], []
        parsed = []
        skipped_missing_images = 0
        sample_limit = None if num_samples is None or num_samples <= 0 else int(num_samples)
        for idx, rec in enumerate(records):
            width = rec.get("width", rec.get("image_width", 1))
            height = rec.get("height", rec.get("image_height", 1))
            segments = rec.get("segments_info", rec.get("annotations", []))
            box_annotations = rec.get("annotations", segments)
            if isinstance(segments, dict):
                segments = list(segments.values())
            if isinstance(box_annotations, dict):
                box_annotations = list(box_annotations.values())
            relations = rec.get("relations", rec.get("relation_annotations", []))
            if not segments or not relations:
                continue

            segment_by_ref = {}
            for i, seg in enumerate(segments):
                segment_by_ref[str(i)] = i
                for key in ("id", "segment_id", "annotation_id"):
                    if key in seg:
                        segment_by_ref[str(seg[key])] = i

            nodes = [seg_name(seg) for seg in segments]
            boxes = [
                _box_from_annotation(
                    box_annotations[i] if i < len(box_annotations) else seg,
                    width, height,
                )
                for i, seg in enumerate(segments)
            ]
            obj_names.extend(nodes)
            rels = []
            for rel in relations:
                if isinstance(rel, dict):
                    s_ref = rel.get("subject_id", rel.get("subject", rel.get("subj_id", 0)))
                    o_ref = rel.get("object_id", rel.get("object", rel.get("obj_id", 0)))
                    pred = rel.get("predicate", rel.get("predicate_name", rel.get("relation", "related to")))
                else:
                    if len(rel) < 3:
                        continue
                    s_ref, o_ref, pred = rel[0], rel[1], rel[2]
                if str(s_ref) not in segment_by_ref or str(o_ref) not in segment_by_ref:
                    continue
                pname = pred_name(pred)
                rels.append((segment_by_ref[str(s_ref)], segment_by_ref[str(o_ref)], pname))
                pred_names.append(pname)
            if rels:
                file_name = rec.get("file_name")
                if self.image_root is not None:
                    image_candidates = [] if not file_name else [
                        self.image_root / file_name,
                        self.image_root / Path(file_name).name,
                        self.image_root / "train2017" / Path(file_name).name,
                        self.image_root / "val2017" / Path(file_name).name,
                    ]
                    if not any(path.is_file() for path in image_candidates):
                        skipped_missing_images += 1
                        continue
                panoptic_name = rec.get(
                    "pan_seg_file_name", rec.get("panoptic_file_name")
                )
                segment_ids = [
                    seg.get("id", seg.get("segment_id")) for seg in segments
                ]
                parsed.append((
                    rec.get("image_id", idx), nodes, boxes, rels,
                    file_name, panoptic_name, segment_ids,
                ))
                if sample_limit is not None and len(parsed) >= sample_limit:
                    break

        if not parsed:
            raise ValueError(f"No PSG scene graphs parsed from {annotation_path}")

        if object_classes:
            self.obj_vocab = {
                _norm(name): i + 1 for i, name in enumerate(object_classes)
            }
        else:
            self.obj_vocab = _build_vocab(obj_names)
        if predicate_classes:
            self.pred_vocab = {
                _norm(name): i + 1 for i, name in enumerate(predicate_classes)
            }
        else:
            self.pred_vocab = _build_vocab(pred_names)
        self.ontology_id = _ontology_id("psg", self.obj_vocab, self.pred_vocab)
        self.num_entity_classes = len(self.obj_vocab) + 1
        self.num_predicate_classes = len(self.pred_vocab) + 1
        self.sgg_dict = {
            "idx_to_label": {str(idx): name for name, idx in self.obj_vocab.items()},
            "idx_to_predicate": {str(idx): name for name, idx in self.pred_vocab.items()},
        }
        self.items = parsed
        print(f"  [PSG] graphs={len(self.items)} objects={len(self.obj_vocab)} "
              f"predicates={len(self.pred_vocab)} excluded_ids={len(excluded_ids)} "
              f"skipped_missing_images={skipped_missing_images}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_id, nodes, boxes, rels, file_name, panoptic_name, segment_ids = self.items[idx]
        result = _scene_to_batch(
            image_id, nodes, boxes, rels, self.obj_vocab, self.pred_vocab,
            "psg", self.ontology_id, self.include_proxy_features,
        )
        if self.image_root and file_name:
            candidates = [
                self.image_root / file_name,
                self.image_root / Path(file_name).name,
                self.image_root / "train2017" / Path(file_name).name,
                self.image_root / "val2017" / Path(file_name).name,
            ]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is not None:
                result["image_path"] = str(image_path)
            if self.include_raw_images:
                image = load_image_tensor(image_path)
                if image is not None:
                    result["image"] = image
        if self.panoptic_root is not None and panoptic_name:
            candidates = [
                self.panoptic_root / panoptic_name,
                self.panoptic_root / Path(panoptic_name).name,
                self.panoptic_root / "panoptic_train2017" / Path(panoptic_name).name,
                self.panoptic_root / "panoptic_val2017" / Path(panoptic_name).name,
            ]
            panoptic_path = next((path for path in candidates if path.is_file()), None)
            if panoptic_path is not None and all(value is not None for value in segment_ids):
                try:
                    from PIL import Image
                    rgb = np.asarray(Image.open(panoptic_path).convert("RGB"), dtype=np.int64)
                    ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]
                    result["masks"] = torch.stack([
                        torch.from_numpy((ids == int(segment_id)).copy())
                        for segment_id in segment_ids
                    ])
                    result["panoptic_path"] = str(panoptic_path)
                    result["segment_ids"] = torch.tensor(
                        [int(value) for value in segment_ids], dtype=torch.long
                    )
                except (OSError, ValueError):
                    pass
        return result


def build_gqa_loader(scene_graph_path: str, num_samples: int = 500,
                     batch_size: int = 1,
                     vocabulary_path: str | None = None,
                     image_root: str | None = None,
                     include_proxy_features: bool = True,
                     include_raw_images: bool = True) -> DataLoader:
    dataset = GQASceneGraphDataset(
        scene_graph_path=scene_graph_path,
        num_samples=num_samples,
        vocabulary_path=vocabulary_path,
        image_root=image_root,
        include_proxy_features=include_proxy_features,
        include_raw_images=include_raw_images,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)


def build_psg_loader(annotation_path: str, num_samples: int = 500,
                     batch_size: int = 1,
                     exclude_annotation_path: str | None = None,
                     image_root: str | None = None,
                     panoptic_root: str | None = None,
                     include_proxy_features: bool = True,
                     include_raw_images: bool = True) -> DataLoader:
    dataset = PSGSceneGraphDataset(
        annotation_path=annotation_path,
        num_samples=num_samples,
        exclude_annotation_path=exclude_annotation_path,
        image_root=image_root,
        panoptic_root=panoptic_root,
        include_proxy_features=include_proxy_features,
        include_raw_images=include_raw_images,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)


def check_gqa_file(scene_graph_path: str) -> dict:
    ok = bool(scene_graph_path) and Path(scene_graph_path).exists()
    return {"ok": ok, "missing": [] if ok else [scene_graph_path or "--gqa_scene_graph"]}


def check_psg_file(annotation_path: str) -> dict:
    ok = bool(annotation_path) and Path(annotation_path).exists()
    return {"ok": ok, "missing": [] if ok else [annotation_path or "--psg_ann"]}
