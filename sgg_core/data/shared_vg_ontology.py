"""Project external scene graphs onto an explicitly shared VG-150 ontology.

Only normalized exact label matches are retained. This deliberately avoids
semantic synonym expansion, which would introduce an unreviewed source of
label noise into the external evaluation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


def normalize_label(value: str) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def load_vg_ontology(dictionary_path: str | Path) -> dict:
    path = Path(dictionary_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = {
        normalize_label(name): int(index)
        for name, index in payload["label_to_idx"].items()
    }
    predicates = {
        normalize_label(name): int(index)
        for name, index in payload["predicate_to_idx"].items()
    }
    canonical = json.dumps(
        {"objects": objects, "predicates": predicates},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ontology_id": "vg150:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16],
        "objects": objects,
        "predicates": predicates,
        "num_objects": max(objects.values()) + 1,
        "num_predicates": max(predicates.values()) + 1,
    }


def build_exact_mapping(dataset, vg_ontology: dict) -> dict:
    source_objects = dataset.sgg_dict["idx_to_label"]
    source_predicates = dataset.sgg_dict["idx_to_predicate"]
    object_map = {
        int(index): vg_ontology["objects"][normalize_label(name)]
        for index, name in source_objects.items()
        if normalize_label(name) in vg_ontology["objects"]
    }
    predicate_map = {
        int(index): vg_ontology["predicates"][normalize_label(name)]
        for index, name in source_predicates.items()
        if normalize_label(name) in vg_ontology["predicates"]
    }
    return {
        "policy": "normalized_exact_match_only",
        "object_map": object_map,
        "predicate_map": predicate_map,
        "source_object_classes": len(source_objects),
        "source_predicate_classes": len(source_predicates),
        "mapped_object_classes": len(object_map),
        "mapped_predicate_classes": len(predicate_map),
        "object_class_coverage": (
            len(object_map) / len(source_objects) if source_objects else 0.0
        ),
        "predicate_class_coverage": (
            len(predicate_map) / len(source_predicates)
            if source_predicates else 0.0
        ),
    }


def project_batch_to_vg(batch: dict, mapping: dict, vg_ontology: dict) -> tuple[dict | None, dict]:
    """Retain mapped relation endpoints and predicates, preserving box order."""
    labels = batch["entity_labels"].long()
    pairs = batch["rel_pairs"].long()
    predicates = batch["rel_labels"].long()
    object_map = mapping["object_map"]
    predicate_map = mapping["predicate_map"]
    mapped_objects = {
        index: object_map[int(label)]
        for index, label in enumerate(labels.tolist())
        if int(label) in object_map
    }
    retained_relations = []
    skipped = Counter()
    for pair, predicate in zip(pairs.tolist(), predicates.tolist()):
        subject, obj = map(int, pair)
        if subject not in mapped_objects or obj not in mapped_objects:
            skipped["unmapped_endpoint"] += 1
            continue
        if int(predicate) not in predicate_map:
            skipped["unmapped_predicate"] += 1
            continue
        retained_relations.append((subject, obj, predicate_map[int(predicate)]))
    if not retained_relations:
        return None, {
            "source_objects": int(labels.numel()),
            "source_relations": int(predicates.numel()),
            "retained_objects": 0,
            "retained_relations": 0,
            "skipped_relations": dict(skipped),
        }

    retained_indices = sorted({
        index for subject, obj, _ in retained_relations for index in (subject, obj)
    })
    old_to_new = {old: new for new, old in enumerate(retained_indices)}
    projected_pairs = torch.tensor([
        [old_to_new[subject], old_to_new[obj]]
        for subject, obj, _ in retained_relations
    ], dtype=torch.long)
    projected_predicates = torch.tensor(
        [predicate for _, _, predicate in retained_relations], dtype=torch.long
    )
    projected_labels = torch.tensor(
        [mapped_objects[index] for index in retained_indices], dtype=torch.long
    )
    adjacency = torch.zeros(
        (len(retained_indices), len(retained_indices)), dtype=torch.float32
    )
    for subject, obj in projected_pairs.tolist():
        adjacency[subject, obj] = 1.0
        adjacency[obj, subject] = 1.0

    projected = {
        "boxes": batch["boxes"][retained_indices].float(),
        "entity_labels": projected_labels,
        "rel_pairs": projected_pairs,
        "rel_labels": projected_predicates,
        "graph_adj": adjacency,
        "num_nodes": len(retained_indices),
        "image_id": str(batch["image_id"]),
        "image": batch.get("image"),
        "image_path": batch.get("image_path"),
        "feature_source": "raw_image_gt_boxes_shared_vg_ontology",
        "dataset": str(batch.get("dataset", "external")),
        "source_ontology_id": batch.get("ontology_id"),
        "ontology_id": vg_ontology["ontology_id"],
        "num_entity_classes": vg_ontology["num_objects"],
        "num_predicate_classes": vg_ontology["num_predicates"],
    }
    if not isinstance(projected["image"], torch.Tensor):
        return None, {
            "source_objects": int(labels.numel()),
            "source_relations": int(predicates.numel()),
            "retained_objects": 0,
            "retained_relations": 0,
            "skipped_relations": dict(skipped),
            "missing_image_tensor": True,
        }
    return projected, {
        "source_objects": int(labels.numel()),
        "source_relations": int(predicates.numel()),
        "retained_objects": len(retained_indices),
        "retained_relations": len(retained_relations),
        "skipped_relations": dict(skipped),
    }
