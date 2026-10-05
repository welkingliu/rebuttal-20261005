"""Conservative 2D physical-consistency audit for predicted predicates."""

from __future__ import annotations

import re
import numpy as np
import torch
from tqdm import tqdm


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def predicate_violation(predicate: str, subject_box, object_box,
                        tolerance: float = 0.02) -> bool | None:
    """Return violation, consistency, or None when 2D boxes cannot decide."""
    predicate = _normalise(predicate)
    subject = np.asarray(subject_box, dtype=np.float64)
    obj = np.asarray(object_box, dtype=np.float64)
    sx, sy = (subject[:2] + subject[2:]) / 2.0
    ox, oy = (obj[:2] + obj[2:]) / 2.0
    tol = float(tolerance)

    if predicate in {"left of", "to the left of", "left"}:
        return not (sx < ox - tol)
    if predicate in {"right of", "to the right of", "right"}:
        return not (sx > ox + tol)
    if predicate in {"above", "over"}:
        return not (sy < oy - tol)
    if predicate in {"below", "under"}:
        return not (sy > oy + tol)
    if predicate in {"inside", "in", "inside of"}:
        consistent = (
            subject[0] >= obj[0] - tol and subject[1] >= obj[1] - tol
            and subject[2] <= obj[2] + tol and subject[3] <= obj[3] + tol
        )
        return not consistent
    if predicate in {"contains", "around", "surrounding"}:
        consistent = (
            obj[0] >= subject[0] - tol and obj[1] >= subject[1] - tol
            and obj[2] <= subject[2] + tol and obj[3] <= subject[3] + tol
        )
        return not consistent
    if predicate in {"overlapping", "overlaps", "intersecting"}:
        intersection_w = max(0.0, min(subject[2], obj[2]) - max(subject[0], obj[0]))
        intersection_h = max(0.0, min(subject[3], obj[3]) - max(subject[1], obj[1]))
        return not (intersection_w * intersection_h > 0.0)
    # Depth, contact, support, distance, and action predicates are not
    # identifiable from two-dimensional boxes alone.
    return None


def _vocabulary(loader) -> dict[int, str]:
    dataset = getattr(loader, "dataset", None)
    sgg_dict = getattr(dataset, "sgg_dict", {}) or {}
    return {
        int(key): value
        for key, value in sgg_dict.get("idx_to_predicate", {}).items()
    }


def summarise_pvr(image_counts, total_predictions: int,
                  min_checked: int = 100, seed: int = 211,
                  trials: int = 2000) -> dict:
    checked = sum(row["checked"] for row in image_counts)
    violations = sum(row["violations"] for row in image_counts)
    coverage = checked / max(int(total_predictions), 1)
    if checked < int(min_checked):
        return {
            "PVR": None,
            "pvr_status": "insufficient_support" if checked else "undefined",
            "pvr_checked": checked,
            "pvr_violations": violations,
            "coverage": coverage,
            "minimum_checked": int(min_checked),
            "bootstrap_95ci": [None, None],
        }
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(int(trials)):
        sampled = [
            image_counts[index]
            for index in rng.integers(0, len(image_counts), len(image_counts))
        ]
        denominator = sum(row["checked"] for row in sampled)
        if denominator:
            rates.append(sum(row["violations"] for row in sampled) / denominator)
    return {
        "PVR": violations / checked,
        "pvr_status": "ok",
        "pvr_checked": checked,
        "pvr_violations": violations,
        "coverage": coverage,
        "minimum_checked": int(min_checked),
        "bootstrap_95ci": (
            [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]
            if rates else [None, None]
        ),
    }


class PhysicalConsistencyAudit:
    def __init__(self, min_checked=100, device="cpu"):
        self.min_checked = int(min_checked)
        self.device = device

    def run(self, models: dict, loader) -> dict:
        names = _vocabulary(loader)
        return {
            name: self._run_model(model, loader, names)
            for name, model in models.items()
        }

    def _run_model(self, model, loader, names):
        image_counts, total_predictions, errors = [], 0, []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="  [PVR]", leave=False):
                moved = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                boxes, pairs = moved.get("boxes"), moved.get("rel_pairs")
                if boxes is None or pairs is None or not pairs.numel():
                    continue
                try:
                    scores = model.predict(moved)["pred_rel_scores"]
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                predictions = scores.argmax(dim=-1)
                row = {"checked": 0, "violations": 0}
                for pair, predicate_id in zip(pairs.tolist(), predictions.tolist()):
                    total_predictions += 1
                    predicate = names.get(int(predicate_id))
                    if predicate is None:
                        continue
                    subject, obj = map(int, pair)
                    if not (0 <= subject < boxes.size(0) and 0 <= obj < boxes.size(0)):
                        continue
                    violation = predicate_violation(
                        predicate, boxes[subject].cpu(), boxes[obj].cpu()
                    )
                    if violation is None:
                        continue
                    row["checked"] += 1
                    row["violations"] += int(violation)
                if row["checked"]:
                    image_counts.append(row)
        result = summarise_pvr(
            image_counts, total_predictions, min_checked=self.min_checked
        )
        result.update({
            "metric_definition": "2D-box-identifiable predicate violation rate",
            "unidentifiable_predicates_excluded": True,
            "total_predictions": total_predictions,
            "num_checked_images": len(image_counts),
            "errors": errors,
        })
        return result
