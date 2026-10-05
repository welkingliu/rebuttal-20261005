"""Cache-free validation for mitigation on the VG train holdout.

The formal prediction caches cover the untouched VG split-2 test set. They
must not be used to select a mitigation epoch on a split-0 holdout. This audit
therefore runs the live SGCls adapter with GT boxes and reports object identity
and relation metrics from the same forward pass.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from sgg_core.audits.standard_sgg_eval import _Accumulator, _as_probabilities


def _ece(confidences: list[float], correctness: list[bool], bins: int = 15) -> float:
    if not confidences:
        return float("nan")
    confidence = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correctness, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            value += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return value


class LiveSGClsValidationAudit:
    """Evaluate GT-box object identity and SGCls without static caches."""

    def __init__(self, ks, device, seen_triplets=None):
        self.ks = tuple(sorted({int(value) for value in ks}))
        self.device = torch.device(device)
        self.seen_triplets = seen_triplets

    def run(self, model, loader) -> dict:
        relation = _Accumulator(self.ks, self.seen_triplets)
        class_total = Counter()
        class_correct = Counter()
        confidences: list[float] = []
        correctness: list[bool] = []
        errors: list[str] = []
        object_count = 0
        valid_images = 0

        model.eval()
        with torch.no_grad():
            for raw_batch in tqdm(
                loader, desc=f"  [LiveValidation:sgcls] {model.name}", leave=False
            ):
                batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor)
                    else value
                    for key, value in raw_batch.items()
                }
                try:
                    prediction = model.forward_grounding(batch)
                    entity_scores = prediction["pred_entity_scores"]
                    targets = batch["entity_labels"].long()
                    if entity_scores.ndim != 2 or entity_scores.size(0) != targets.numel():
                        raise ValueError(
                            "Live object logits must align with GT-box entities"
                        )
                    probabilities = _as_probabilities(entity_scores)
                    if probabilities.size(1) <= 1:
                        raise ValueError("Live object logits have no foreground classes")
                    foreground = probabilities[:, 1:]
                    predicted_confidence, predicted_label = foreground.max(dim=1)
                    predicted_label = predicted_label + 1
                    valid = (targets > 0) & (targets < probabilities.size(1))
                    if not bool(valid.any()):
                        raise ValueError("Validation image has no valid object labels")

                    selected_targets = targets[valid]
                    selected_predictions = predicted_label[valid]
                    selected_confidence = predicted_confidence[valid]
                    selected_correct = selected_predictions.eq(selected_targets)
                    object_count += int(valid.sum().item())
                    class_total.update(
                        int(value) for value in selected_targets.detach().cpu().tolist()
                    )
                    class_correct.update(
                        int(target)
                        for target, is_correct in zip(
                            selected_targets.detach().cpu().tolist(),
                            selected_correct.detach().cpu().tolist(),
                        )
                        if is_correct
                    )
                    confidences.extend(
                        float(value)
                        for value in selected_confidence.detach().cpu().tolist()
                    )
                    correctness.extend(
                        bool(value)
                        for value in selected_correct.detach().cpu().tolist()
                    )
                    relation.update(prediction, batch, "sgcls", iou_threshold=1.0)
                    valid_images += 1
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"{type(exc).__name__}: {exc}")

        relation_summary = relation.summary()
        relation_summary.update({
            "status": (
                "ok" if relation_summary["num_ground_truth_relations"] > 0 and not errors
                else "partial" if relation_summary["num_ground_truth_relations"] > 0
                else "failed"
            ),
            "task": "sgcls",
            "iou_threshold": 1.0,
            "ks": list(self.ks),
            "num_images": int(valid_images),
            "errors": list(errors),
            "evaluation_source": "live_forward_grounding",
            "box_protocol": "ground_truth_boxes",
        })
        top1 = (
            float(np.mean(correctness)) if correctness else float("nan")
        )
        per_class = {
            class_id: class_correct[class_id] / count
            for class_id, count in class_total.items() if count
        }
        macro = (
            float(np.mean(list(per_class.values())))
            if per_class else float("nan")
        )
        object_identity = {
            "top1_accuracy_given_localized": top1,
            "macro_accuracy_given_localized": macro,
            "ece_15": _ece(confidences, correctness, bins=15),
            "localized_objects": object_count,
            "correct_objects": int(sum(correctness)),
            "classes_with_support": len(per_class),
            "box_protocol": "ground_truth_boxes",
            "evaluation_source": "live_forward_grounding",
        }
        status = (
            "ok" if valid_images and not errors and np.isfinite(top1)
            else "partial" if valid_images else "failed"
        )
        return {
            "standard_sgg": {
                "status": relation_summary["status"],
                "tasks": {"sgcls": relation_summary},
                "requested_tasks": ["sgcls"],
                "unsupported_tasks": [],
                "validation_protocol": "live_sgcls_on_disjoint_split0_holdout",
            },
            "grounding_error_decomposition": {
                "status": status,
                "num_images": int(valid_images),
                "errors": list(errors),
                "iou_threshold": 1.0,
                "object_identity": object_identity,
                "metrics": {
                    "localization_recall": 1.0 if object_count else float("nan"),
                    "recognition_accuracy_given_localized": top1,
                    "grounded_object_recall": top1,
                    "mean_box_iou": 1.0 if object_count else float("nan"),
                    "mean_mask_iou": float("nan"),
                },
                "counts": {
                    "gt_objects": object_count,
                    "localized_objects": object_count,
                    "recognized_objects": int(sum(correctness)),
                },
                "interpretation": (
                    "GT-box live validation measures object classification and "
                    "SGCls; it does not measure autonomous localization or SGDet."
                ),
            },
        }
