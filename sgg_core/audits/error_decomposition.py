"""Object grounding error decomposition for complete SGG models."""

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sgg_core.audits.standard_sgg_eval import box_iou


def _greedy_matches(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                    threshold: float) -> list[tuple[int, int, float]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []
    ious = box_iou(pred_boxes.float(), gt_boxes.float())
    flat = torch.argsort(ious.reshape(-1), descending=True)
    used_pred, used_gt, matches = set(), set(), []
    num_gt = gt_boxes.size(0)
    for flat_index in flat.tolist():
        pred_index, gt_index = divmod(flat_index, num_gt)
        value = float(ious[pred_index, gt_index].item())
        if value < threshold:
            break
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, value))
    return matches


def _mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred = pred_mask.float()
    gt = gt_mask.float()
    if pred.shape != gt.shape:
        pred = F.interpolate(
            pred[None, None], size=gt.shape[-2:], mode="bilinear",
            align_corners=False,
        )[0, 0]
    pred = pred > 0.5
    gt = gt > 0.5
    union = (pred | gt).sum().item()
    return float((pred & gt).sum().item() / union) if union else 1.0


def _bootstrap_image_means(rows, key, seed=181, trials=2000):
    values = np.asarray([row[key] for row in rows if np.isfinite(row[key])])
    if values.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [
        values[rng.integers(0, values.size, values.size)].mean()
        for _ in range(int(trials))
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _bootstrap_object_rate(rows, value_key: str, seed=197, trials=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["image_index"])].append(float(row[value_key]))
    image_means = np.asarray([
        np.mean(grouped[key]) for key in sorted(grouped)
    ], dtype=np.float64)
    if image_means.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = [
        image_means[rng.integers(0, image_means.size, image_means.size)].mean()
        for _ in range(int(trials))
    ]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


class GroundingErrorDecompositionAudit:
    """Separate localization, recognition, and mask-quality failures.

    Relation reasoning remains quantified by standard PredCls/SGCls/SGDet. This
    audit supplies the missing object-level denominators needed to interpret
    the gaps between those tasks.
    """

    def __init__(self, iou_threshold=0.5, device="cpu", class_frequency=None):
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.class_frequency = {
            int(key): int(value) for key, value in (class_frequency or {}).items()
        }

    def _frequency_groups(self):
        ordered = sorted(
            self.class_frequency,
            key=lambda value: (-self.class_frequency[value], value),
        )
        groups = np.array_split(np.asarray(ordered, dtype=np.int64), 3)
        return {
            name: {int(value) for value in group.tolist()}
            for name, group in zip(("head", "body", "tail"), groups)
        }

    @staticmethod
    def _ece(confidence, correct, bins=15):
        if not confidence:
            return float("nan")
        confidence = np.asarray(confidence, dtype=np.float64)
        correct = np.asarray(correct, dtype=np.float64)
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        value = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (confidence > lower) & (confidence <= upper)
            if selected.any():
                value += selected.mean() * abs(
                    correct[selected].mean() - confidence[selected].mean()
                )
        return float(value)

    def run(self, models: dict, loader) -> dict:
        return {
            name: self._run_model(name, model, loader)
            for name, model in models.items()
        }

    def _run_model(self, name, model, loader):
        if not getattr(model, "supports_standard_sgg", False):
            return {"status": "standard_sgdet_required"}
        if "sgdet" not in getattr(model, "supported_tasks", ("sgdet",)):
            return {"status": "sgdet_not_supported"}
        rows, errors = [], []
        object_rows = []
        totals = defaultdict(float)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  [GroundingDecomp] {name}", leave=False):
                moved = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                gt_boxes = moved.get("boxes")
                gt_labels = moved.get("entity_labels")
                if gt_boxes is None or gt_labels is None or not gt_labels.numel():
                    continue
                try:
                    output = model.predict_scene_graph(moved, task="sgdet")
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                pred_boxes = output.get("pred_boxes")
                entity_scores = output.get("pred_entity_scores")
                if pred_boxes is None or entity_scores is None:
                    errors.append("missing pred_boxes or pred_entity_scores")
                    continue
                if entity_scores.ndim == 2:
                    probabilities = entity_scores.float()
                    if (
                        float(probabilities.min()) < 0.0
                        or float(probabilities.max()) > 1.0
                        or not torch.allclose(
                            probabilities.sum(dim=1),
                            torch.ones(probabilities.size(0), device=probabilities.device),
                            atol=1e-3,
                        )
                    ):
                        probabilities = probabilities.softmax(dim=1)
                    if probabilities.size(1) <= 1:
                        confidence = torch.ones(
                            probabilities.size(0), device=probabilities.device
                        )
                        pred_labels = torch.zeros(
                            probabilities.size(0), dtype=torch.long,
                            device=probabilities.device,
                        )
                    else:
                        confidence, pred_labels = probabilities[:, 1:].max(dim=-1)
                        pred_labels = pred_labels + 1
                else:
                    pred_labels = entity_scores.long()
                    confidence = torch.ones_like(pred_labels, dtype=torch.float32)
                matches = _greedy_matches(
                    pred_boxes, gt_boxes, self.iou_threshold
                )
                recognized = sum(
                    int(pred_labels[pred_index]) == int(gt_labels[gt_index])
                    for pred_index, gt_index, _ in matches
                )
                num_gt = int(gt_labels.numel())
                localized = len(matches)
                row = {
                    "localization_recall": localized / num_gt,
                    "recognition_accuracy_given_localized": (
                        recognized / localized if localized else float("nan")
                    ),
                    "grounded_object_recall": recognized / num_gt,
                    "mean_box_iou": (
                        float(np.mean([value for _, _, value in matches]))
                        if matches else float("nan")
                    ),
                    "mean_mask_iou": float("nan"),
                }
                pred_masks, gt_masks = output.get("pred_masks"), moved.get("masks")
                mask_by_match = {}
                if isinstance(pred_masks, torch.Tensor) and isinstance(gt_masks, torch.Tensor):
                    mask_by_match = {
                        (pred_index, gt_index): _mask_iou(
                            pred_masks[pred_index], gt_masks[gt_index]
                        )
                        for pred_index, gt_index, _ in matches
                        if pred_index < pred_masks.size(0) and gt_index < gt_masks.size(0)
                    }
                    mask_values = list(mask_by_match.values())
                    if mask_values:
                        row["mean_mask_iou"] = float(np.mean(mask_values))
                        totals["mask_matches"] += len(mask_values)
                for pred_index, gt_index, box_value in matches:
                    target = int(gt_labels[gt_index])
                    predicted = int(pred_labels[pred_index])
                    box = gt_boxes[gt_index].float()
                    area = float(
                        (box[2] - box[0]).clamp_min(0)
                        * (box[3] - box[1]).clamp_min(0)
                    )
                    object_rows.append({
                        "image_index": len(rows),
                        "target": target,
                        "prediction": predicted,
                        "correct": int(predicted == target),
                        "confidence": float(confidence[pred_index]),
                        "box_iou": float(box_value),
                        "mask_iou": float(mask_by_match.get(
                            (pred_index, gt_index), float("nan")
                        )),
                        "area": area,
                    })
                rows.append(row)
                totals["gt_objects"] += num_gt
                totals["localized_objects"] += localized
                totals["recognized_objects"] += recognized

        if not rows:
            return {"status": "no_valid_images", "errors": errors}
        metrics = {}
        for key in (
            "localization_recall", "recognition_accuracy_given_localized",
            "grounded_object_recall", "mean_box_iou", "mean_mask_iou",
        ):
            values = [row[key] for row in rows if np.isfinite(row[key])]
            metrics[key] = float(np.mean(values)) if values else float("nan")
        class_correct = Counter()
        class_total = Counter()
        confusion = Counter()
        for row in object_rows:
            class_total[row["target"]] += 1
            class_correct[row["target"]] += row["correct"]
            if not row["correct"]:
                confusion[(row["target"], row["prediction"])] += 1
        groups = self._frequency_groups()
        macro = [
            class_correct[key] / value for key, value in class_total.items() if value
        ]
        identity = {
            "support_localized_objects": len(object_rows),
            "top1_accuracy_given_localized": (
                float(np.mean([row["correct"] for row in object_rows]))
                if object_rows else float("nan")
            ),
            "macro_accuracy_given_localized": (
                float(np.mean(macro)) if macro else float("nan")
            ),
            "head_body_tail_macro_accuracy": {
                name: (
                    float(np.mean([
                        class_correct[key] / class_total[key]
                        for key in classes if class_total[key] > 0
                    ])) if any(class_total[key] > 0 for key in classes)
                    else float("nan")
                )
                for name, classes in groups.items()
            },
            "ece_15": self._ece(
                [row["confidence"] for row in object_rows],
                [row["correct"] for row in object_rows],
            ),
            "top_confusions": [
                {"target": target, "prediction": prediction, "count": count}
                for (target, prediction), count in confusion.most_common(25)
            ],
            "class_support": {
                str(key): value for key, value in sorted(class_total.items())
            },
            "mask_iou_bins": {},
            "wrong_given_good_mask": {},
        }
        boundaries = (0.0, 0.5, 0.7, 0.85, 0.95, 1.000001)
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            selected = [
                row for row in object_rows
                if np.isfinite(row["mask_iou"])
                and lower <= row["mask_iou"] < upper
            ]
            identity["mask_iou_bins"][f"[{lower:.2f},{min(upper, 1.0):.2f}]"] = {
                "support": len(selected),
                "top1_accuracy": (
                    float(np.mean([row["correct"] for row in selected]))
                    if selected else float("nan")
                ),
            }
        for threshold in (0.75, 0.85, 0.90, 0.95):
            selected = [
                row for row in object_rows
                if np.isfinite(row["mask_iou"]) and row["mask_iou"] >= threshold
            ]
            identity["wrong_given_good_mask"][f"iou>={threshold:.2f}"] = {
                "support": len(selected),
                "error_rate": (
                    float(np.mean([1 - row["correct"] for row in selected]))
                    if selected else float("nan")
                ),
                "bootstrap_95ci": _bootstrap_object_rate(
                    [{**row, "error": 1 - row["correct"]} for row in selected],
                    "error", seed=197 + int(round(threshold * 100)),
                ) if selected else [float("nan"), float("nan")],
            }
        return {
            "status": "ok" if not errors else "partial",
            "iou_threshold": self.iou_threshold,
            "num_images": len(rows),
            "counts": dict(totals),
            "metrics": metrics,
            "object_identity": identity,
            "bootstrap_95ci": {
                key: _bootstrap_image_means(rows, key)
                for key in metrics
            },
            "interpretation": (
                "Use with PredCls/SGCls/SGDet gaps; localization alone does not "
                "establish object recognition quality."
            ),
            "errors": errors,
        }
