"""Experiment I-B: VG-150 PredCls relation-depth component study.

This replaces the former relation-token classifier.  Every graph node is one
ground-truth object and every supervised example is an ordered object pair.
The experiment is intentionally PredCls-only and does not test object
recognition. Experiment I-A is the paper's object-grounding study; complete
PredCls/SGCls/SGDet evaluation belongs to Experiment IV.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from sgg_core.backbones.perception import PerceptionModule
from sgg_core.backbones.reasoning import GATLayer, GCNLayer, TransformerLayer
from sgg_core.audits.feature_audit import dirichlet_energy, effective_rank
from sgg_core.audits.physical_consistency import predicate_violation, summarise_pvr
from sgg_core.data.data_utils import build_vg_test_loader


RECALL_KS = (1, 5, 10, 20, 50, 100)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _all_ordered_pairs(num_nodes: int) -> torch.Tensor:
    return torch.tensor(
        [(s, o) for s in range(num_nodes) for o in range(num_nodes) if s != o],
        dtype=torch.long,
    )


def _graph_adjacency(num_nodes: int, rel_pairs: torch.Tensor) -> torch.Tensor:
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if rel_pairs.numel():
        valid = rel_pairs[
            (rel_pairs[:, 0] >= 0) & (rel_pairs[:, 0] < num_nodes)
            & (rel_pairs[:, 1] >= 0) & (rel_pairs[:, 1] < num_nodes)
        ]
        if valid.numel():
            adjacency[valid[:, 0], valid[:, 1]] = 1.0
            adjacency[valid[:, 1], valid[:, 0]] = 1.0
    return adjacency


def _union_boxes(boxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    subject = boxes[pairs[:, 0]]
    obj = boxes[pairs[:, 1]]
    return torch.stack(
        (
            torch.minimum(subject[:, 0], obj[:, 0]),
            torch.minimum(subject[:, 1], obj[:, 1]),
            torch.maximum(subject[:, 2], obj[:, 2]),
            torch.maximum(subject[:, 3], obj[:, 3]),
        ),
        dim=1,
    )


class FrozenROIEncoder(nn.Module):
    """Frozen raw-image backbone with backbone-specific preprocessing."""

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, backbone: str, device: torch.device,
                 roi_chunk_size: int = 512):
        super().__init__()
        self.backbone_name = backbone
        self.device = device
        self.encoder = PerceptionModule(
            backbone_type=backbone,
            output_dim=PerceptionModule.get_output_dim_for(backbone),
            signal_threshold=0.0,
        ).to(device)
        for parameter in self.encoder.backbone.parameters():
            parameter.requires_grad = False
        self.encoder.backbone.eval()
        self.output_dim = self.encoder.native_dim
        self.input_size = self.encoder.BACKBONE_REGISTRY[backbone][2]
        if roi_chunk_size < 1:
            raise ValueError("roi_chunk_size must be positive")
        self.roi_chunk_size = int(roi_chunk_size)
        if backbone.startswith("clip_"):
            mean, std = self.CLIP_MEAN, self.CLIP_STD
            self.preprocess_id = "openai_clip_rgb_norm"
        elif backbone.startswith("siglip2_"):
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            self.preprocess_id = "siglip2_rgb_norm"
        elif "radio" in backbone:
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            self.preprocess_id = "radio_internal_rgb_norm"
        else:
            mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD
            self.preprocess_id = "imagenet_rgb_norm"
        self.register_buffer(
            "mean", torch.tensor(mean, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, device=device).view(1, 3, 1, 1)
        )
        self._provenance = None

    @torch.no_grad()
    def extract_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """Return one normalized, frozen spatial feature map."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = F.interpolate(
            image.to(device=self.device, dtype=torch.float32),
            size=(self.input_size, self.input_size),
            mode="bicubic", align_corners=False,
        )
        image = (image - self.mean) / self.std
        return self.encoder._extract_feature_map(image)

    @staticmethod
    def _pool_boxes(feature_map: torch.Tensor, boxes: torch.Tensor,
                    roi_chunk_size: int, output_size: int = 1) -> torch.Tensor:
        try:
            from torchvision.ops import roi_align
        except ImportError as exc:
            raise ImportError(
                "raw_backbone mode requires torchvision with compiled ROIAlign"
            ) from exc
        _, _, height, width = feature_map.shape
        boxes = boxes.to(device=feature_map.device, dtype=torch.float32)
        chunks = []
        for start in range(0, boxes.size(0), roi_chunk_size):
            scaled = boxes[start:start + roi_chunk_size].to(
                device=feature_map.device, dtype=feature_map.dtype
            ).clone()
            scaled[:, (0, 2)] *= width
            scaled[:, (1, 3)] *= height
            roi_boxes = torch.cat(
                (torch.zeros(
                    scaled.size(0), 1, device=feature_map.device,
                    dtype=scaled.dtype,
                ), scaled),
                dim=1,
            )
            pooled = roi_align(
                feature_map, roi_boxes,
                output_size=(output_size, output_size),
                spatial_scale=1.0, aligned=True,
            )
            chunks.append(pooled.float().mean(dim=(-2, -1)).cpu())
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _pool_masks(feature_map: torch.Tensor, masks: torch.Tensor,
                    fallback: torch.Tensor) -> torch.Tensor:
        masks = masks.to(device=feature_map.device, dtype=torch.float32)
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape [N,H,W], got {tuple(masks.shape)}")
        weights = F.interpolate(
            masks[:, None], size=feature_map.shape[-2:], mode="area"
        ).clamp(0.0, 1.0)
        spatial = feature_map.expand(weights.size(0), -1, -1, -1)
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        pooled = (spatial * weights).sum(dim=(-2, -1)) / denominator
        empty = weights.sum(dim=(-2, -1)).squeeze(1) <= 1e-6
        if bool(empty.any()):
            pooled[empty] = fallback.to(pooled.device)[empty]
        return pooled.float().cpu()

    @torch.no_grad()
    def encode_object_views(self, image: torch.Tensor, boxes: torch.Tensor,
                            masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Extract comparable box and optional mask features for Experiment I-A."""
        feature_map = self.extract_feature_map(image)
        box_features = self._pool_boxes(
            feature_map, boxes, self.roi_chunk_size, output_size=7
        )
        views = {"box": box_features}
        if masks is not None:
            views["gt_mask"] = self._pool_masks(feature_map, masks, box_features)
        return views

    @torch.no_grad()
    def encode(self, image: torch.Tensor, boxes: torch.Tensor,
               pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.extract_feature_map(image)
        boxes = boxes.to(device=feature_map.device, dtype=torch.float32)
        pairs = pairs.to(device=feature_map.device, dtype=torch.long)
        union_boxes = _union_boxes(boxes, pairs)
        return (
            self._pool_boxes(feature_map, boxes, self.roi_chunk_size, output_size=1),
            self._pool_boxes(
                feature_map, union_boxes, self.roi_chunk_size, output_size=1
            ),
        )

    def provenance(self) -> dict:
        if self._provenance is not None:
            return dict(self._provenance)
        state = self.encoder.backbone.state_dict()
        weight_digest = hashlib.sha256()
        for key, value in state.items():
            tensor = value.detach().contiguous().cpu()
            weight_digest.update(key.encode("utf-8"))
            weight_digest.update(str(tensor.dtype).encode("ascii"))
            weight_digest.update(str(tuple(tensor.shape)).encode("ascii"))
            weight_digest.update(tensor.numpy().tobytes())
        self._provenance = {
            "backbone": self.backbone_name,
            "native_feature_dim": self.output_dim,
            "input_size": self.input_size,
            "preprocess": self.preprocess_id,
            "roi_chunk_size": self.roi_chunk_size,
            "state_dict_sha256": weight_digest.hexdigest(),
            "weights_source": getattr(
                self.encoder, "pretrained_source", "library_pretrained_weights"
            ),
        }
        return dict(self._provenance)


class ObjectPairReasoner(nn.Module):
    """Object-node GNN followed by a directed pair predicate head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_predicates: int,
                 depth: int, mode: str):
        super().__init__()
        self.depth = int(depth)
        self.mode = mode
        self.object_projection = nn.Linear(input_dim, hidden_dim)
        self.union_projection = nn.Linear(input_dim, hidden_dim)
        layers = []
        for _ in range(self.depth):
            if mode == "gcn":
                layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif mode == "gat":
                layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4))
            elif mode == "transformer":
                layers.append(TransformerLayer(hidden_dim, num_heads=8))
            else:
                raise ValueError(f"Unsupported reasoning mode: {mode}")
        self.layers = nn.ModuleList(layers)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates),
        )

    @staticmethod
    def geometry(boxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        subject, obj = boxes[pairs[:, 0]], boxes[pairs[:, 1]]
        sw = (subject[:, 2] - subject[:, 0]).clamp_min(1e-4)
        sh = (subject[:, 3] - subject[:, 1]).clamp_min(1e-4)
        ow = (obj[:, 2] - obj[:, 0]).clamp_min(1e-4)
        oh = (obj[:, 3] - obj[:, 1]).clamp_min(1e-4)
        return torch.stack(
            (
                obj[:, 0] - subject[:, 0], obj[:, 1] - subject[:, 1],
                obj[:, 2] - subject[:, 2], obj[:, 3] - subject[:, 3],
                torch.log(ow / sw), torch.log(oh / sh), sw * sh, ow * oh,
            ), dim=1,
        )

    def encode_nodes(self, object_features: torch.Tensor,
                     adjacency: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        node = self.object_projection(object_features)
        probes = [object_features, node]
        for layer in self.layers:
            node = layer(node, adjacency)
            probes.append(node)
        return node, probes

    def score_pairs(self, node: torch.Tensor, union_features: torch.Tensor,
                    boxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        union = self.union_projection(union_features)
        pair_repr = torch.cat(
            (node[pairs[:, 0]], node[pairs[:, 1]], union,
             self.geometry(boxes, pairs)), dim=1,
        )
        return self.pair_head(pair_repr)

    def forward(self, object_features: torch.Tensor, union_features: torch.Tensor,
                boxes: torch.Tensor, adjacency: torch.Tensor,
                pairs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        node, probes = self.encode_nodes(object_features, adjacency)
        return self.score_pairs(node, union_features, boxes, pairs), probes


def _record_from_batch(batch: dict, encoder: FrozenROIEncoder | None,
                       allow_proxy: bool) -> dict:
    boxes = batch["boxes"].float().cpu()
    entities = batch["entity_labels"].long().cpu()
    gt_pairs = batch["rel_pairs"].long().cpu()
    gt_labels = batch["rel_labels"].long().cpu()
    all_pairs = _all_ordered_pairs(boxes.size(0))
    if all_pairs.numel() == 0:
        raise ValueError("An object graph requires at least two nodes")

    if encoder is not None:
        image = batch.get("image")
        if not isinstance(image, torch.Tensor):
            raise RuntimeError(
                "raw_backbone mode requires batch['image']; "
                f"image_id={batch.get('image_id', 'unknown')} "
                f"image_path={batch.get('image_path', 'unresolved')}"
            )
        object_features, union_features = encoder.encode(image, boxes, all_pairs)
        feature_source = f"raw_backbone:{encoder.backbone_name}"
    else:
        feature_source = str(batch.get("feature_source", "unknown"))
        if feature_source not in {
            "roi_hdf5", "official_precomputed_features", "detector_roi_features"
        } and not allow_proxy:
            raise RuntimeError(
                f"Paper mode rejects feature_source={feature_source!r}; use raw_backbone "
                "or official detector features. --allow_proxy is smoke-test only."
            )
        object_features = batch["visual_features"].float().cpu()
        union_features = (
            object_features[all_pairs[:, 0]] + object_features[all_pairs[:, 1]]
        ) / 2.0

    positives = defaultdict(list)
    for pair, label in zip(gt_pairs.tolist(), gt_labels.tolist()):
        if label > 0:
            positives[tuple(pair)].append(int(label))
    return {
        "image_id": str(batch.get("image_id", batch.get("img_id", "unknown"))),
        "boxes": boxes,
        "entity_labels": entities,
        "gt_pairs": gt_pairs,
        "gt_labels": gt_labels,
        "all_pairs": all_pairs,
        "object_features": object_features,
        "union_features": union_features,
        "adjacency": _graph_adjacency(boxes.size(0), gt_pairs),
        "positive_map": dict(positives),
        "feature_source": feature_source,
    }


def materialize(loader, encoder, cache_path: Path, allow_proxy: bool,
                cache_dtype: torch.dtype = torch.float16) -> list[dict]:
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, batch in enumerate(loader):
        try:
            record = _record_from_batch(batch, encoder, allow_proxy)
            record["object_features"] = record["object_features"].to(cache_dtype)
            record["union_features"] = record["union_features"].to(cache_dtype)
            records.append(record)
        except ValueError as exc:
            if not records and index < 3:
                print(f"[skip] {exc}")
            continue
        except RuntimeError as exc:
            if encoder is not None:
                raise RuntimeError(
                    "Raw-backbone feature materialization failed at "
                    f"loader_index={index}, "
                    f"image_id={batch.get('image_id', 'unknown')}, "
                    f"image_path={batch.get('image_path', 'unresolved')}: {exc}"
                ) from exc
            if not records and index < 3:
                print(f"[skip] {exc}")
            continue
    if not records:
        raise RuntimeError("No valid object graphs were materialized")
    torch.save(records, cache_path)
    return records


def _training_examples(record: dict, negative_ratio: int,
                       generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs, labels, source_indices = [], [], []
    pair_to_index = {tuple(pair): i for i, pair in enumerate(record["all_pairs"].tolist())}
    for pair, predicates in record["positive_map"].items():
        if pair not in pair_to_index:
            continue
        for predicate in predicates:
            pairs.append(pair)
            labels.append(predicate)
            source_indices.append(pair_to_index[pair])
    negative_indices = [
        i for i, pair in enumerate(record["all_pairs"].tolist())
        if tuple(pair) not in record["positive_map"]
    ]
    max_negatives = min(len(negative_indices), max(len(labels) * negative_ratio, 1))
    if max_negatives:
        selected = torch.randperm(len(negative_indices), generator=generator)[:max_negatives]
        for offset in selected.tolist():
            index = negative_indices[offset]
            pairs.append(tuple(record["all_pairs"][index].tolist()))
            labels.append(0)
            source_indices.append(index)
    return (
        torch.tensor(pairs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(source_indices, dtype=torch.long),
    )


def train_model(model, records, epochs, learning_rate, weight_decay,
                negative_ratio, seed, device, gradient_accumulation_steps=4,
                amp=True):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(records), generator=generator).tolist()
        loss_sum = 0.0
        relation_count = 0
        optimizer_steps = 0
        accumulation_count = 0
        optimizer.zero_grad(set_to_none=True)
        for index in order:
            record = records[index]
            pairs, labels, pair_indices = _training_examples(
                record, negative_ratio, generator
            )
            if not labels.numel():
                continue
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, _ = model(
                    record["object_features"].to(device, dtype=torch.float32),
                    record["union_features"][pair_indices].to(
                        device, dtype=torch.float32
                    ),
                    record["boxes"].to(device), record["adjacency"].to(device),
                    pairs.to(device),
                )
                loss = F.cross_entropy(
                    logits, labels.to(device), label_smoothing=0.05
                )
            scaler.scale(loss / gradient_accumulation_steps).backward()
            accumulation_count += 1
            if accumulation_count == gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                optimizer_steps += 1
            loss_sum += float(loss.item()) * labels.numel()
            relation_count += labels.numel()
        if accumulation_count:
            correction = gradient_accumulation_steps / accumulation_count
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        history.append({
            "epoch": epoch + 1,
            "cross_entropy": loss_sum / max(relation_count, 1),
            "training_pairs": relation_count,
            "optimizer_steps": optimizer_steps,
            "image_batch_size": 1,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "amp": amp_enabled,
        })
        print(json.dumps(history[-1], sort_keys=True))
    return history


@torch.no_grad()
def evaluate_model(model, records, seen_triplets, num_predicates, device,
                   predicate_names, minimum_pvr_checked,
                   pair_chunk_size=512, amp=True):
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive")
    model.eval()
    amp_enabled = bool(amp and device.type == "cuda")
    image_recalls = {k: [] for k in RECALL_KS}
    class_hits = {k: Counter() for k in RECALL_KS}
    class_totals = Counter()
    zero_hits = {k: 0 for k in RECALL_KS}
    zero_total = 0
    layer_rank, layer_energy = defaultdict(list), defaultdict(list)
    pvr_image_counts = []
    pvr_total_predictions = 0

    for record in records:
        pairs = record["all_pairs"].to(device)
        object_features = record["object_features"].to(device, dtype=torch.float32)
        boxes = record["boxes"].to(device)
        adjacency = record["adjacency"].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=amp_enabled,
        ):
            node, probes = model.encode_nodes(object_features, adjacency)
        logit_chunks = []
        for start in range(0, pairs.size(0), pair_chunk_size):
            stop = start + pair_chunk_size
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp_enabled,
            ):
                chunk = model.score_pairs(
                    node,
                    record["union_features"][start:stop].to(
                        device, dtype=torch.float32
                    ),
                    boxes,
                    pairs[start:stop],
                )
            logit_chunks.append(chunk.float().cpu())
        logits = torch.cat(logit_chunks, dim=0)
        candidate_scores = logits[:, 1:num_predicates].reshape(-1)
        pair_lookup = {tuple(pair): i for i, pair in enumerate(record["all_pairs"].tolist())}
        top1_predicates = logits.argmax(dim=-1).cpu()
        pvr_row = {"checked": 0, "violations": 0}
        for pair in record["positive_map"]:
            pair_index = pair_lookup.get(pair)
            if pair_index is None:
                continue
            pvr_total_predictions += 1
            predicate_name = predicate_names.get(int(top1_predicates[pair_index]))
            if predicate_name is None:
                continue
            violation = predicate_violation(
                predicate_name, record["boxes"][pair[0]], record["boxes"][pair[1]]
            )
            if violation is not None:
                pvr_row["checked"] += 1
                pvr_row["violations"] += int(violation)
        if pvr_row["checked"]:
            pvr_image_counts.append(pvr_row)
        gt = []
        for pair, predicates in record["positive_map"].items():
            pair_index = pair_lookup.get(pair)
            if pair_index is None:
                continue
            for predicate in predicates:
                candidate_index = pair_index * (num_predicates - 1) + predicate - 1
                triplet = (
                    int(record["entity_labels"][pair[0]]), predicate,
                    int(record["entity_labels"][pair[1]]),
                )
                gt.append((candidate_index, predicate, triplet))
                class_totals[predicate] += 1
                if triplet not in seen_triplets:
                    zero_total += 1
        if not gt:
            continue
        for k in RECALL_KS:
            top = set(candidate_scores.topk(min(k, candidate_scores.numel())).indices.tolist())
            hits = sum(candidate_index in top for candidate_index, _, _ in gt)
            image_recalls[k].append(hits / len(gt))
            for candidate_index, predicate, triplet in gt:
                if candidate_index in top:
                    class_hits[k][predicate] += 1
                    if triplet not in seen_triplets:
                        zero_hits[k] += 1
        for layer_index, features in enumerate(probes):
            centered = features.float() - features.float().mean(dim=0, keepdim=True)
            layer_rank[layer_index].append(effective_rank(centered.cpu()))
            layer_energy[layer_index].append(
                dirichlet_energy(features.float().cpu(), record["adjacency"])
            )

    metrics = {}
    for k in RECALL_KS:
        recalls = image_recalls[k]
        per_class = [
            class_hits[k][predicate] / total
            for predicate, total in class_totals.items() if total
        ]
        metrics[f"R@{k}"] = float(np.mean(recalls)) if recalls else float("nan")
        metrics[f"mR@{k}"] = float(np.mean(per_class)) if per_class else float("nan")
        metrics[f"zR@{k}"] = zero_hits[k] / zero_total if zero_total else float("nan")
    return {
        "metric_scope": "VG-150 PredCls, image-level triplet ranking",
        "metrics": metrics,
        "zero_shot_support": zero_total,
        "predicate_support": dict(class_totals),
        "layer_diagnostics": {
            str(layer): {
                "effective_rank": float(np.mean(layer_rank[layer])),
                "dirichlet_energy": float(np.mean(layer_energy[layer])),
                "num_graphs": len(layer_rank[layer]),
            }
            for layer in sorted(layer_rank)
        },
        "physical_consistency": summarise_pvr(
            pvr_image_counts, pvr_total_predictions,
            min_checked=minimum_pvr_checked,
        ),
    }


def seen_triplets(records) -> set[tuple[int, int, int]]:
    seen = set()
    for record in records:
        for pair, predicates in record["positive_map"].items():
            subject = int(record["entity_labels"][pair[0]])
            obj = int(record["entity_labels"][pair[1]])
            seen.update((subject, predicate, obj) for predicate in predicates)
    return seen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vg_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_mode", choices=("raw_backbone", "official_features", "proxy_smoke"), default="raw_backbone")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--reasoning", choices=("gcn", "gat", "transformer"), default="gcn")
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--negative_ratio", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--roi_chunk_size", type=int, default=512)
    parser.add_argument("--eval_pair_chunk_size", type=int, default=512)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use CUDA mixed precision; ignored on CPU.",
    )
    parser.add_argument("--minimum_pvr_checked", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or output_dir / "feature_cache")
    device = torch.device(args.device)

    use_raw = args.feature_mode == "raw_backbone"
    encoder = FrozenROIEncoder(
        args.backbone, device, roi_chunk_size=args.roi_chunk_size
    ) if use_raw else None
    backbone_provenance = encoder.provenance() if encoder else None
    allow_proxy = args.feature_mode == "proxy_smoke"
    train_loader = build_vg_test_loader(
        args.vg_root, args.train_samples, split=0,
        include_proxy_features=not use_raw, include_raw_images=use_raw,
    )
    test_loader = build_vg_test_loader(
        args.vg_root, args.test_samples, split=2,
        include_proxy_features=not use_raw, include_raw_images=use_raw,
    )
    ontology_id = train_loader.dataset.ontology_id
    cache_key = hashlib.sha256(
        json.dumps({
            "protocol_version": 6,
            "ontology_id": ontology_id,
            "vg_box_encoding": "boxes_1024_cxcywh_max_side_normalized_xyxy_v2",
            "feature_mode": args.feature_mode,
            "backbone": args.backbone,
            "backbone_provenance": backbone_provenance,
            "cache_dtype": "float16",
        }, sort_keys=True).encode()
    ).hexdigest()[:12]
    train_records = materialize(
        train_loader, encoder,
        cache_dir / f"train_{args.train_samples}_{cache_key}.pt", allow_proxy,
    )
    test_records = materialize(
        test_loader, encoder,
        cache_dir / f"test_{args.test_samples}_{cache_key}.pt", allow_proxy,
    )
    predicate_count = int(train_loader.dataset.num_predicate_classes)
    sgg_dict = getattr(train_loader.dataset, "sgg_dict", {}) or {}
    predicate_names = {
        int(key): value
        for key, value in sgg_dict.get("idx_to_predicate", {}).items()
    }
    input_dim = int(train_records[0]["object_features"].size(1))
    seen = seen_triplets(train_records)
    runs = []

    for seed in args.seeds:
        for depth in args.depths:
            seed_everything(seed)
            model = ObjectPairReasoner(
                input_dim, args.hidden_dim, predicate_count, depth, args.reasoning
            ).to(device)
            history = train_model(
                model, train_records, args.epochs, args.learning_rate,
                args.weight_decay, args.negative_ratio, seed, device,
                args.gradient_accumulation_steps, args.amp,
            )
            evaluation = evaluate_model(
                model, test_records, seen, predicate_count, device,
                predicate_names, args.minimum_pvr_checked,
                args.eval_pair_chunk_size, args.amp,
            )
            run_name = f"{args.backbone}_{args.reasoning}_L{depth}_seed{seed}"
            checkpoint = output_dir / f"{run_name}.pth"
            torch.save({
                "state_dict": model.state_dict(), "args": vars(args),
                "ontology_id": ontology_id, "seed": seed, "depth": depth,
            }, checkpoint)
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result = {
                "run_name": run_name,
                "seed": seed,
                "depth": depth,
                "history": history,
                "evaluation": evaluation,
                "checkpoint": checkpoint.name,
                "checkpoint_path_base": "summary_directory",
                "checkpoint_sha256": checkpoint_digest,
                "feature_source": train_records[0]["feature_source"],
                "paper_eligible": not allow_proxy,
            }
            runs.append(result)
            with open(output_dir / f"{run_name}.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

    protocol = {
        "experiment": "I-B_VG150_PredCls_relation_depth_component_study",
        "ontology_id": ontology_id,
        "metric_scope": (
            "VG-150 PredCls relation-depth component study; does not test "
            "object recognition, segmentation, SGCls, or SGDet"
        ),
        "recall_ks": list(RECALL_KS),
        "backbone_provenance": backbone_provenance,
        "memory_protocol": {
            "image_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "roi_chunk_size": args.roi_chunk_size,
            "eval_pair_chunk_size": args.eval_pair_chunk_size,
            "feature_cache_dtype": "float16",
            "amp_requested": args.amp,
        },
        "runs": runs,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)


if __name__ == "__main__":
    main()
