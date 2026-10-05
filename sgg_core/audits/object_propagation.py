"""Measure how endpoint identity errors propagate into relation prediction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from sgg_core.audits.error_decomposition import _greedy_matches


def _labels(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 1:
        return scores.long()
    return scores.argmax(dim=-1)


def _bootstrap_group(rows, group: str, trials=2000, seed=911):
    eligible = [row[group] for row in rows if row[group]["total"] > 0]
    if len(eligible) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(trials)):
        sampled = [eligible[index] for index in rng.integers(0, len(eligible), len(eligible))]
        total = sum(item["total"] for item in sampled)
        values.append(sum(item["hit"] for item in sampled) / total)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


class ObjectErrorPropagationAudit:
    """Stratify relation Hit@K by endpoint object-recognition correctness.

    SGCls uses the GT-box-aligned entity rows. SGDet first performs one-to-one
    IoU matching and only evaluates GT relations whose two endpoints and pair
    row are present. Coverage is reported explicitly, so missed detections are
    not silently converted into relation mistakes or silently discarded.
    """

    def __init__(self, ks=(1, 5), iou_threshold=0.5, device="cpu",
                 bootstrap_trials=2000):
        self.ks = tuple(sorted({int(value) for value in ks}))
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.bootstrap_trials = int(bootstrap_trials)

    def run(self, models: dict, loader) -> dict:
        return {
            name: self._run_model(name, model, loader)
            for name, model in models.items()
        }

    def _run_model(self, name, model, loader):
        supported = set(getattr(model, "supported_tasks", ()))
        tasks = [task for task in ("sgcls", "sgdet") if task in supported]
        if not tasks:
            return {"status": "sgcls_or_sgdet_required", "tasks": {}}
        task_results = {}
        for task in tasks:
            task_results[task] = self._run_task(name, model, loader, task)
        valid = [value for value in task_results.values() if value["status"] == "ok"]
        return {
            "status": "ok" if valid else "failed",
            "tasks": task_results,
            "interpretation": (
                "Association only: endpoint errors and relation errors share upstream "
                "causes. Use Experiment II interventions for stronger evidence."
            ),
        }

    def _run_task(self, name, model, loader, task):
        groups = ("both_correct", "one_wrong", "both_wrong")
        image_rows = []
        totals = defaultdict(int)
        errors = []
        model.eval()
        with torch.no_grad():
            for image_index, batch in enumerate(tqdm(
                loader, desc=f"  [ObjectPropagation:{task}] {name}", leave=False
            )):
                moved = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                try:
                    output = model.predict_scene_graph(moved, task=task)
                    row = self._image_row(output, moved, task)
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"image={image_index} {type(exc).__name__}: {exc}")
                    continue
                image_rows.append(row)
                for key, value in row["counts"].items():
                    totals[key] += int(value)

        if not image_rows:
            return {"status": "failed", "errors": errors}
        result_groups = {}
        for group in groups:
            support = sum(row[group]["total"] for row in image_rows)
            result_groups[group] = {
                "support": support,
                **{
                    f"relation_Hit@{k}": (
                        sum(row[group][f"hit@{k}"] for row in image_rows) / support
                        if support else float("nan")
                    )
                    for k in self.ks
                },
                **{
                    f"bootstrap_95ci_Hit@{k}": _bootstrap_group(
                        [
                            {
                                group: {
                                    "hit": row[group][f"hit@{k}"],
                                    "total": row[group]["total"],
                                }
                            }
                            for row in image_rows
                        ],
                        group,
                        trials=self.bootstrap_trials,
                        seed=911 + k,
                    )
                    for k in self.ks
                },
            }
        gt_relations = totals["gt_relations"]
        return {
            "status": "ok" if gt_relations else "failed",
            "task": task,
            "num_images": len(image_rows),
            "counts": dict(totals),
            "endpoint_and_pair_coverage": (
                totals["evaluable_relations"] / gt_relations if gt_relations else 0.0
            ),
            "localized_endpoint_coverage": (
                totals["localized_endpoint_relations"] / gt_relations
                if gt_relations else 0.0
            ),
            "groups": result_groups,
            "errors": errors,
        }

    def _image_row(self, output, batch, task):
        gt_labels = batch["entity_labels"].long()
        gt_pairs = batch["rel_pairs"].long()
        gt_predicates = batch["rel_labels"].long()
        entity_scores = output["pred_entity_scores"]
        pred_labels = _labels(entity_scores)
        if task == "sgcls":
            if pred_labels.numel() != gt_labels.numel():
                raise ValueError("SGCls entity rows must align with GT entities")
            gt_to_pred = {index: index for index in range(gt_labels.numel())}
        else:
            pred_boxes = output["pred_boxes"].float()
            matches = _greedy_matches(
                pred_boxes, batch["boxes"].float(), self.iou_threshold
            )
            gt_to_pred = {gt: pred for pred, gt, _ in matches}

        pred_pairs = output["pred_rel_pairs"].long()
        rel_scores = output["pred_rel_scores"].float()
        if pred_pairs.size(0) != rel_scores.size(0):
            raise ValueError("Relation pair rows and score rows differ")
        pair_rows = defaultdict(list)
        for index, pair in enumerate(pred_pairs.tolist()):
            pair_rows[(int(pair[0]), int(pair[1]))].append(index)

        row = {
            group: {
                "total": 0,
                **{f"hit@{k}": 0 for k in self.ks},
            }
            for group in ("both_correct", "one_wrong", "both_wrong")
        }
        counts = {
            "gt_relations": 0,
            "localized_endpoint_relations": 0,
            "evaluable_relations": 0,
        }
        for pair, predicate in zip(gt_pairs.tolist(), gt_predicates.tolist()):
            subject, obj = map(int, pair)
            predicate = int(predicate)
            if predicate <= 0:
                continue
            counts["gt_relations"] += 1
            if subject not in gt_to_pred or obj not in gt_to_pred:
                continue
            counts["localized_endpoint_relations"] += 1
            pred_subject, pred_object = gt_to_pred[subject], gt_to_pred[obj]
            candidates = pair_rows.get((pred_subject, pred_object), [])
            if not candidates:
                continue
            counts["evaluable_relations"] += 1
            candidate_scores = rel_scores[candidates]
            foreground_confidence = candidate_scores[:, 1:].max(dim=1).values
            selected = candidate_scores[foreground_confidence.argmax()]
            endpoint_correct = (
                int(pred_labels[pred_subject]) == int(gt_labels[subject]),
                int(pred_labels[pred_object]) == int(gt_labels[obj]),
            )
            correct_count = sum(endpoint_correct)
            group = ("both_wrong", "one_wrong", "both_correct")[correct_count]
            row[group]["total"] += 1
            for k in self.ks:
                foreground = selected[1:]
                top = (
                    foreground.topk(min(k, foreground.numel())).indices + 1
                ).tolist()
                row[group][f"hit@{k}"] += int(predicate in top)
        row["counts"] = counts
        return row
