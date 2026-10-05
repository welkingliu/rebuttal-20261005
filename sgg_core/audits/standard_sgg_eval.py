"""Standard image-level SGG evaluation.

The pair audit measures predicate Hit@K on annotated object pairs. This module
implements the separate benchmark protocol used for PredCls, SGCls and SGDet:
all predicted triplets in an image are ranked jointly and matched to unique
ground-truth triplets. SGDet additionally requires subject and object IoU.

Models must expose ``predict_scene_graph(batch, task)`` and return:
  pred_boxes:          [N, 4] normalized xyxy
  pred_entity_scores:  [N, C_obj] logits or probabilities
  pred_rel_pairs:      [M, 2] indices into pred_boxes
  pred_rel_scores:     [M, C_rel] logits or probabilities
Optional:
  pred_box_scores:     [N]

For PredCls, ground-truth boxes and object labels are used by definition. For
SGCls, ground-truth boxes are used while object labels are predicted. SGDet
uses all predicted boxes, labels and relations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_SGG_KS = (1, 5, 10, 20, 50, 100)
VALID_TASKS = ("predcls", "sgcls", "sgdet")


def _normalise_ks(ks: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not values:
        raise ValueError("ks must contain at least one positive integer")
    return values


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized or absolute xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=torch.float32,
            device=boxes1.device,
        )
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
             * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
             * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-12)


def _as_probabilities(scores: torch.Tensor, *, independent: bool = False) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return scores
    if independent:
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Independent probabilities contain non-finite values")
        if not bool(((scores >= 0) & (scores <= 1)).all()):
            raise ValueError("Independent probabilities must lie in [0,1]")
        return scores
    row_sums = scores.sum(dim=-1)
    looks_like_probability = (
        bool((scores >= 0).all())
        and bool((scores <= 1).all())
        and bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))
    )
    return scores if looks_like_probability else torch.softmax(scores, dim=-1)


def _labels_and_scores(entity_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = _as_probabilities(entity_scores)
    if probs.size(1) <= 1:
        return (
            torch.zeros(probs.size(0), dtype=torch.long, device=probs.device),
            torch.ones(probs.size(0), device=probs.device),
        )
    foreground = probs[:, 1:]
    values, labels = foreground.max(dim=1)
    return labels + 1, values


@dataclass
class RankedTriplets:
    scores: torch.Tensor
    subj_boxes: torch.Tensor
    obj_boxes: torch.Tensor
    subj_labels: torch.Tensor
    predicates: torch.Tensor
    obj_labels: torch.Tensor

    def top(self, k: int) -> "RankedTriplets":
        n = min(int(k), int(self.scores.numel()))
        if n == 0:
            idx = torch.zeros(0, dtype=torch.long, device=self.scores.device)
        else:
            idx = torch.argsort(self.scores, descending=True)[:n]
        return RankedTriplets(
            self.scores[idx], self.subj_boxes[idx], self.obj_boxes[idx],
            self.subj_labels[idx], self.predicates[idx], self.obj_labels[idx],
        )


def _build_ranked_triplets(prediction: dict,
                           batch: dict,
                           task: str,
                           graph_constraint: bool) -> RankedTriplets:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown SGG task: {task}")
    required = ("pred_rel_pairs", "pred_rel_scores")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError(f"Scene-graph prediction missing: {missing}")

    rel_pairs = prediction["pred_rel_pairs"].long()
    relation_score_mode = str(
        prediction.get("pred_rel_score_mode", "categorical")
    ).lower()
    if relation_score_mode not in {"categorical", "independent_probabilities"}:
        raise ValueError(f"Unknown pred_rel_score_mode: {relation_score_mode}")
    rel_probs = _as_probabilities(
        prediction["pred_rel_scores"],
        independent=relation_score_mode == "independent_probabilities",
    )
    if rel_pairs.ndim != 2 or rel_pairs.size(-1) != 2:
        raise ValueError("pred_rel_pairs must have shape [M, 2]")
    if rel_probs.ndim != 2 or rel_probs.size(0) != rel_pairs.size(0):
        raise ValueError("pred_rel_scores must have shape [M, C_rel]")

    if task in ("predcls", "sgcls"):
        boxes = batch["boxes"].float()
    else:
        if "pred_boxes" not in prediction:
            raise KeyError("SGDet requires pred_boxes")
        boxes = prediction["pred_boxes"].float()

    if task == "predcls":
        entity_labels = batch["entity_labels"].long()
        entity_conf = torch.ones_like(entity_labels, dtype=torch.float32)
    else:
        if "pred_entity_scores" not in prediction:
            raise KeyError(f"{task} requires pred_entity_scores")
        entity_labels, entity_conf = _labels_and_scores(prediction["pred_entity_scores"])

    if task == "sgdet" and "pred_box_scores" in prediction:
        entity_conf = entity_conf * prediction["pred_box_scores"].float()

    if boxes.size(0) != entity_labels.numel():
        raise ValueError(
            f"box/entity count mismatch: {boxes.size(0)} vs {entity_labels.numel()}"
        )
    if rel_pairs.numel() and (
        int(rel_pairs.min()) < 0 or int(rel_pairs.max()) >= boxes.size(0)
    ):
        raise ValueError("pred_rel_pairs contains an out-of-range entity index")

    if rel_probs.size(1) <= 1 or rel_pairs.numel() == 0:
        empty = torch.zeros(0, device=boxes.device)
        return RankedTriplets(
            empty, boxes.new_zeros((0, 4)), boxes.new_zeros((0, 4)),
            empty.long(), empty.long(), empty.long(),
        )

    predicate_probs = rel_probs[:, 1:]
    if graph_constraint:
        pred_conf, predicates = predicate_probs.max(dim=1)
        predicates = predicates + 1
        expanded_pairs = rel_pairs
    else:
        num_predicates = predicate_probs.size(1)
        expanded_pairs = rel_pairs.repeat_interleave(num_predicates, dim=0)
        pred_conf = predicate_probs.reshape(-1)
        predicates = torch.arange(
            1, num_predicates + 1, device=rel_pairs.device
        ).repeat(rel_pairs.size(0))

    subj_idx = expanded_pairs[:, 0]
    obj_idx = expanded_pairs[:, 1]
    scores = entity_conf[subj_idx] * pred_conf * entity_conf[obj_idx]
    return RankedTriplets(
        scores=scores,
        subj_boxes=boxes[subj_idx],
        obj_boxes=boxes[obj_idx],
        subj_labels=entity_labels[subj_idx],
        predicates=predicates.long(),
        obj_labels=entity_labels[obj_idx],
    )


def _ground_truth(batch: dict) -> dict:
    boxes = batch["boxes"].float()
    labels = batch["entity_labels"].long()
    pairs = batch["rel_pairs"].long()
    predicates = batch["rel_labels"].long()
    valid = (
        (predicates > 0)
        & (pairs[:, 0] >= 0) & (pairs[:, 0] < boxes.size(0))
        & (pairs[:, 1] >= 0) & (pairs[:, 1] < boxes.size(0))
    )
    pairs = pairs[valid]
    predicates = predicates[valid]
    return {
        "subj_boxes": boxes[pairs[:, 0]],
        "obj_boxes": boxes[pairs[:, 1]],
        "subj_labels": labels[pairs[:, 0]],
        "predicates": predicates,
        "obj_labels": labels[pairs[:, 1]],
    }


def _matched_gt_indices(predictions: RankedTriplets,
                        gt: dict,
                        iou_threshold: float) -> Set[int]:
    """Return the union of GT relations matched by ranked predictions.

    The reference SGG protocol permits one prediction to match every duplicate
    GT row with the same semantic triplet and compatible boxes. Recall is then
    computed from the union of those GT indices.
    """
    if predictions.scores.numel() == 0 or gt["predicates"].numel() == 0:
        return set()
    subj_iou = box_iou(predictions.subj_boxes, gt["subj_boxes"])
    obj_iou = box_iou(predictions.obj_boxes, gt["obj_boxes"])
    matched: Set[int] = set()
    for pred_idx in range(predictions.scores.numel()):
        compatible = (
            (gt["subj_labels"] == predictions.subj_labels[pred_idx])
            & (gt["predicates"] == predictions.predicates[pred_idx])
            & (gt["obj_labels"] == predictions.obj_labels[pred_idx])
            & (subj_iou[pred_idx] >= iou_threshold)
            & (obj_iou[pred_idx] >= iou_threshold)
        )
        matched.update(
            int(i) for i in compatible.nonzero(as_tuple=False).flatten().tolist()
        )
    return matched


@dataclass
class _Accumulator:
    ks: Tuple[int, ...]
    seen_triplets: Optional[Set[Tuple[int, int, int]]] = None
    total_gt: int = 0
    total_zero_shot: int = 0
    predicate_total: Counter = field(default_factory=Counter)
    zero_predicate_total: Counter = field(default_factory=Counter)
    hits: Dict[str, Counter] = field(default_factory=dict)
    predicate_hits: Dict[str, Dict[int, Counter]] = field(default_factory=dict)
    predicate_image_recalls: Dict[str, Dict[int, Dict[int, list]]] = field(
        default_factory=dict
    )
    zero_hits: Dict[str, Counter] = field(default_factory=dict)
    image_recalls: Dict[str, Dict[int, list]] = field(default_factory=dict)
    image_zero_recalls: Dict[str, Dict[int, list]] = field(default_factory=dict)

    def __post_init__(self):
        for mode in ("graph", "ng"):
            self.hits[mode] = Counter()
            self.predicate_hits[mode] = {k: Counter() for k in self.ks}
            self.predicate_image_recalls[mode] = {
                k: {} for k in self.ks
            }
            self.zero_hits[mode] = Counter()
            self.image_recalls[mode] = {k: [] for k in self.ks}
            self.image_zero_recalls[mode] = {k: [] for k in self.ks}

    def update(self, prediction: dict, batch: dict, task: str, iou_threshold: float):
        gt = _ground_truth(batch)
        num_gt = int(gt["predicates"].numel())
        if num_gt == 0:
            return
        self.total_gt += num_gt
        self.predicate_total.update(gt["predicates"].cpu().tolist())

        zero_ids: Set[int] = set()
        if self.seen_triplets is not None:
            for idx, triplet in enumerate(zip(
                gt["subj_labels"].cpu().tolist(),
                gt["predicates"].cpu().tolist(),
                gt["obj_labels"].cpu().tolist(),
            )):
                if tuple(map(int, triplet)) not in self.seen_triplets:
                    zero_ids.add(idx)
                    self.zero_predicate_total[int(triplet[1])] += 1
            self.total_zero_shot += len(zero_ids)

        for mode, graph_constraint in (("graph", True), ("ng", False)):
            ranked = _build_ranked_triplets(prediction, batch, task, graph_constraint)
            for k in self.ks:
                matched = _matched_gt_indices(ranked.top(k), gt, iou_threshold)
                self.hits[mode][k] += len(matched)
                self.image_recalls[mode][k].append(len(matched) / num_gt)
                for idx in matched:
                    self.predicate_hits[mode][k][int(gt["predicates"][idx])] += 1
                for predicate in torch.unique(gt["predicates"]).tolist():
                    predicate = int(predicate)
                    predicate_indices = {
                        int(index) for index in
                        (gt["predicates"] == predicate).nonzero(
                            as_tuple=False
                        ).flatten().tolist()
                    }
                    recall = len(matched & predicate_indices) / len(predicate_indices)
                    self.predicate_image_recalls[mode][k].setdefault(
                        predicate, []
                    ).append(recall)
                if zero_ids:
                    zero_matched = len(matched & zero_ids)
                    self.zero_hits[mode][k] += zero_matched
                    self.image_zero_recalls[mode][k].append(
                        zero_matched / len(zero_ids)
                    )

    @staticmethod
    def _mean_class_recall(hits: Counter, totals: Counter) -> float:
        values = [hits[c] / n for c, n in totals.items() if n > 0]
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _bootstrap_ci(values: Iterable[float], seed: int = 17,
                      trials: int = 1000) -> list:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(seed)
        means = np.empty(trials, dtype=np.float64)
        for i in range(trials):
            means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
        return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]

    def summary(self) -> dict:
        result = {
            "num_ground_truth_relations": self.total_gt,
            "num_zero_shot_relations": self.total_zero_shot,
            "zero_shot_status": "ok" if self.seen_triplets is not None else "not_configured",
            "metrics": {},
            "per_predicate_recall": {},
            "bootstrap_95ci": {},
        }
        for mode, prefix in (("graph", ""), ("ng", "ng")):
            for k in self.ks:
                image_values = self.image_recalls[mode][k]
                recall = float(np.mean(image_values)) if image_values else float("nan")
                micro_recall = (
                    self.hits[mode][k] / self.total_gt
                    if self.total_gt else float("nan")
                )
                mean_recall = self._mean_class_recall(
                    self.predicate_hits[mode][k], self.predicate_total
                )
                image_class_values = [
                    float(np.mean(values))
                    for values in self.predicate_image_recalls[mode][k].values()
                    if values
                ]
                image_mean_recall = (
                    float(np.mean(image_class_values))
                    if image_class_values else float("nan")
                )
                result["metrics"][f"{prefix}R@{k}"] = recall
                result["metrics"][f"{prefix}microR@{k}"] = micro_recall
                result["metrics"][f"{prefix}mR@{k}"] = mean_recall
                # EGTR and several one-stage implementations first average
                # recall over images containing each predicate, then average
                # predicates. Report it separately instead of conflating the
                # two accepted mR aggregation protocols.
                result["metrics"][f"{prefix}imR@{k}"] = image_mean_recall
                if np.isfinite(recall) and np.isfinite(mean_recall) and recall + mean_recall > 0:
                    result["metrics"][f"{prefix}F@{k}"] = 2 * recall * mean_recall / (recall + mean_recall)
                else:
                    result["metrics"][f"{prefix}F@{k}"] = float("nan")
                if self.seen_triplets is not None and self.total_zero_shot:
                    zero_values = self.image_zero_recalls[mode][k]
                    result["metrics"][f"{prefix}zR@{k}"] = float(np.mean(zero_values))
                    result["metrics"][f"{prefix}microzR@{k}"] = (
                        self.zero_hits[mode][k] / self.total_zero_shot
                    )
                else:
                    result["metrics"][f"{prefix}zR@{k}"] = float("nan")
                    result["metrics"][f"{prefix}microzR@{k}"] = float("nan")
                result["bootstrap_95ci"][f"{prefix}R@{k}"] = self._bootstrap_ci(
                    self.image_recalls[mode][k]
                )
                result["per_predicate_recall"][f"{prefix}R@{k}"] = {
                    str(c): self.predicate_hits[mode][k][c] / total
                    for c, total in self.predicate_total.items() if total > 0
                }
        return result


class StandardSGGAudit:
    """Run standard SGG metrics for models implementing the official contract."""

    def __init__(self,
                 ks: Sequence[int] = DEFAULT_SGG_KS,
                 tasks: Sequence[str] = VALID_TASKS,
                 iou_threshold: float = 0.5,
                 seen_triplets: Optional[Set[Tuple[int, int, int]]] = None,
                 device: str = "cpu"):
        self.ks = _normalise_ks(ks)
        self.tasks = tuple(task.lower() for task in tasks)
        invalid = [task for task in self.tasks if task not in VALID_TASKS]
        if invalid:
            raise ValueError(f"Unsupported SGG tasks: {invalid}")
        self.iou_threshold = float(iou_threshold)
        self.seen_triplets = seen_triplets
        self.device = device

    def run(self, models: dict, test_loader) -> dict:
        return {
            name: self._evaluate_model(name, model, test_loader)
            for name, model in models.items()
        }

    def _evaluate_model(self, name: str, model, test_loader) -> dict:
        if not getattr(model, "supports_standard_sgg", False):
            return {
                "status": "unsupported_model_contract",
                "reason": "model must implement predict_scene_graph(batch, task)",
                "implementation_kind": getattr(model, "implementation_kind", "unknown"),
            }

        supported = tuple(getattr(model, "supported_tasks", self.tasks))
        model_tasks = tuple(task for task in self.tasks if task in supported)
        unsupported_tasks = [task for task in self.tasks if task not in supported]
        if not model_tasks:
            return {
                "status": "unsupported_model_contract",
                "reason": "model supports none of the requested SGG tasks",
                "requested_tasks": list(self.tasks),
                "unsupported_tasks": unsupported_tasks,
                "implementation_kind": getattr(model, "implementation_kind", "unknown"),
            }

        model.eval()
        if getattr(model, "supports_joint_task_inference", False):
            return self._evaluate_model_joint(
                name, model, test_loader, model_tasks, unsupported_tasks
            )
        task_results = {}
        num_images = len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader)
        for task in model_tasks:
            accumulator = _Accumulator(self.ks, self.seen_triplets)
            errors = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"  [StandardSGG:{task}] {name}", leave=False):
                    device_batch = {
                        key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    try:
                        prediction = model.predict_scene_graph(device_batch, task=task)
                        accumulator.update(
                            prediction, device_batch, task, self.iou_threshold
                        )
                    except Exception as exc:
                        if len(errors) < 5:
                            errors.append(f"{type(exc).__name__}: {exc}")
            summary = accumulator.summary()
            summary.update({
                "status": "ok" if summary["num_ground_truth_relations"] > 0 and not errors else (
                    "partial" if summary["num_ground_truth_relations"] > 0 else "failed"
                ),
                "task": task,
                "iou_threshold": self.iou_threshold,
                "ks": list(self.ks),
                "num_images": int(num_images),
                "errors": errors,
            })
            task_results[task] = summary
        return {
            "status": "ok" if all(v["status"] == "ok" for v in task_results.values()) else "partial",
            "tasks": task_results,
            "requested_tasks": list(self.tasks),
            "unsupported_tasks": unsupported_tasks,
            "implementation_kind": getattr(model, "implementation_kind", "unknown"),
        }

    def _evaluate_model_joint(
        self, name: str, model, test_loader, model_tasks, unsupported_tasks
    ) -> dict:
        """Evaluate all tasks in one loader pass when the adapter supports it."""
        accumulators = {
            task: _Accumulator(self.ks, self.seen_triplets) for task in model_tasks
        }
        errors = {task: [] for task in model_tasks}
        with torch.no_grad():
            for batch in tqdm(
                test_loader, desc=f"  [StandardSGG:joint] {name}", leave=False
            ):
                device_batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                try:
                    predictions = model.predict_scene_graph_tasks(
                        device_batch, tasks=model_tasks,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    for task in model_tasks:
                        if len(errors[task]) < 5:
                            errors[task].append(message)
                    continue
                for task in model_tasks:
                    try:
                        accumulators[task].update(
                            predictions[task], device_batch, task, self.iou_threshold,
                        )
                    except Exception as exc:
                        if len(errors[task]) < 5:
                            errors[task].append(f"{type(exc).__name__}: {exc}")

        task_results = {}
        num_images = len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader)
        for task, accumulator in accumulators.items():
            summary = accumulator.summary()
            summary.update({
                "status": (
                    "ok" if summary["num_ground_truth_relations"] > 0 and not errors[task]
                    else "partial" if summary["num_ground_truth_relations"] > 0
                    else "failed"
                ),
                "task": task,
                "iou_threshold": self.iou_threshold,
                "ks": list(self.ks),
                "num_images": int(num_images),
                "errors": errors[task],
            })
            task_results[task] = summary
        return {
            "status": "ok" if all(
                value["status"] == "ok" for value in task_results.values()
            ) else "partial",
            "tasks": task_results,
            "requested_tasks": list(self.tasks),
            "unsupported_tasks": unsupported_tasks,
            "implementation_kind": getattr(model, "implementation_kind", "unknown"),
            "joint_task_inference": True,
        }
