"""Object-recognition metrics for segmentation-conditioned grounding audits."""

from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearObjectProbe(nn.Module):
    """A deliberately small probe over a frozen visual representation."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def deterministic_image_split(image_ids, validation_fraction=0.1, seed=17):
    """Split complete images so objects from one image never cross partitions."""
    unique = sorted({str(value) for value in image_ids})
    validation_images = set()
    for image_id in unique:
        digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < float(validation_fraction):
            validation_images.add(image_id)
    if unique and not validation_images:
        validation_images.add(unique[0])
    if len(validation_images) == len(unique) and len(unique) > 1:
        validation_images.remove(unique[-1])
    validation = torch.tensor(
        [str(value) in validation_images for value in image_ids], dtype=torch.bool
    )
    return ~validation, validation


def frequency_groups(labels: torch.Tensor, num_classes: int) -> dict[str, list[int]]:
    """Create head/body/tail class thirds from training frequency only."""
    counts = Counter(int(value) for value in labels.tolist())
    ordered = sorted(range(num_classes), key=lambda c: (-counts[c], c))
    nonempty = [value for value in ordered if counts[value] > 0]
    groups = np.array_split(np.asarray(nonempty, dtype=np.int64), 3)
    return {
        name: [int(value) for value in group.tolist()]
        for name, group in zip(("head", "body", "tail"), groups)
    }


def train_linear_probe(features: torch.Tensor, labels: torch.Tensor,
                       train_mask: torch.Tensor, validation_mask: torch.Tensor,
                       num_classes: int, seed: int, device: str,
                       epochs: int = 100, batch_size: int = 512,
                       learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4,
                       early_stopping_patience: int = 10,
                       early_stopping_min_delta: float = 0.0,
                       ) -> tuple[LinearObjectProbe, list[dict]]:
    torch.manual_seed(int(seed))
    model = LinearObjectProbe(features.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_indices = train_mask.nonzero(as_tuple=False).flatten()
    validation_indices = validation_mask.nonzero(as_tuple=False).flatten()
    if not train_indices.numel() or not validation_indices.numel():
        raise ValueError("Probe training requires non-empty image-disjoint train/validation sets")
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        order = train_indices[
            torch.randperm(train_indices.numel(), generator=generator)
        ]
        loss_sum = 0.0
        count = 0
        for start in range(0, order.numel(), int(batch_size)):
            index = order[start:start + int(batch_size)]
            logits = model(features[index].to(device, dtype=torch.float32))
            target = labels[index].to(device)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * index.numel()
            count += index.numel()
        model.eval()
        with torch.no_grad():
            validation_logits = []
            for start in range(0, validation_indices.numel(), int(batch_size)):
                index = validation_indices[start:start + int(batch_size)]
                validation_logits.append(
                    model(features[index].to(device, dtype=torch.float32)).cpu()
                )
            val_logits = torch.cat(validation_logits)
            val_loss = float(F.cross_entropy(
                val_logits, labels[validation_indices]
            ).item())
        record = {
            "epoch": epoch + 1,
            "train_cross_entropy": loss_sum / max(count, 1),
            "validation_cross_entropy": val_loss,
        }
        improved = val_loss < best_loss - float(early_stopping_min_delta)
        record["improved"] = bool(improved)
        history.append(record)
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if (
                int(early_stopping_patience) > 0
                and epochs_without_improvement >= int(early_stopping_patience)
            ):
                break
    if best_state is None:
        raise RuntimeError("Probe training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def batched_logits(model: nn.Module, features: torch.Tensor, device: str,
                   batch_size: int = 512) -> torch.Tensor:
    model.eval()
    values = []
    for start in range(0, features.size(0), int(batch_size)):
        values.append(model(
            features[start:start + int(batch_size)].to(device, dtype=torch.float32)
        ).float().cpu())
    return torch.cat(values, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_steps: int = 100) -> float:
    """Fit one positive temperature on validation data only."""
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=int(max_steps), line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits.float() / temperature, labels.long())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def expected_calibration_error(probabilities: torch.Tensor,
                               labels: torch.Tensor, bins: int = 15,
                               adaptive: bool = False) -> float:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    if adaptive:
        order = confidence.argsort()
        partitions = torch.tensor_split(order, int(bins))
    else:
        edges = torch.linspace(0.0, 1.0, int(bins) + 1)
        partitions = [
            ((confidence > edges[index]) & (confidence <= edges[index + 1])).nonzero(
                as_tuple=False
            ).flatten()
            for index in range(int(bins))
        ]
    total = max(int(labels.numel()), 1)
    error = 0.0
    for index in partitions:
        if not index.numel():
            continue
        error += index.numel() / total * abs(
            float(correct[index].mean()) - float(confidence[index].mean())
        )
    return float(error)


def _bootstrap_image_accuracy(correct: torch.Tensor, image_ids,
                              seed=181, trials=2000) -> list[float]:
    """Cluster-bootstrap the object-micro accuracy by resampling images."""
    grouped = {}
    for value, image_id in zip(correct.tolist(), image_ids):
        row = grouped.setdefault(str(image_id), [0.0, 0])
        row[0] += float(value)
        row[1] += 1
    rows = np.asarray([
        grouped[key] for key in sorted(grouped)
    ], dtype=np.float64)
    if rows.shape[0] < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    estimates = [
        (lambda sample: sample[:, 0].sum() / sample[:, 1].sum())(
            rows[rng.integers(0, rows.shape[0], rows.shape[0])]
        )
        for _ in range(int(trials))
    ]
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _prediction_concentration(prediction: torch.Tensor) -> dict:
    counts = Counter(int(value) for value in prediction.tolist())
    total = max(int(prediction.numel()), 1)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
    entropy = float(-(probabilities * np.log(probabilities.clip(1e-12))).sum())
    class_id, count = counts.most_common(1)[0] if counts else (-1, 0)
    return {
        "unique_predicted_classes": len(counts),
        "most_predicted_class": int(class_id),
        "most_predicted_fraction": float(count / total),
        "prediction_entropy_nats": entropy,
    }


def _macro_accuracy(correct: torch.Tensor, labels: torch.Tensor,
                    classes=None) -> float:
    allowed = set(int(value) for value in classes) if classes is not None else None
    values = []
    for class_id in labels.unique().tolist():
        class_id = int(class_id)
        if allowed is not None and class_id not in allowed:
            continue
        mask = labels == class_id
        if bool(mask.any()):
            values.append(float(correct[mask].float().mean()))
    return float(np.mean(values)) if values else float("nan")


def evaluate_object_logits(logits: torch.Tensor, labels: torch.Tensor,
                           image_ids, groups: dict[str, list[int]],
                           areas: torch.Tensor | None = None,
                           mask_iou: torch.Tensor | None = None,
                           temperature: float = 1.0) -> dict:
    logits = logits.float() / float(temperature)
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    top_k = min(5, logits.size(1))
    top5 = logits.topk(top_k, dim=1).indices.eq(labels[:, None]).any(dim=1)
    class_counts = Counter(int(value) for value in labels.tolist())
    confusion = Counter(
        (int(target), int(predicted))
        for target, predicted in zip(labels.tolist(), prediction.tolist())
        if target != predicted
    )
    result = {
        "num_objects": int(labels.numel()),
        "num_images": len({str(value) for value in image_ids}),
        "top1_accuracy": float(correct.float().mean()),
        "top5_accuracy": float(top5.float().mean()),
        "macro_accuracy": _macro_accuracy(correct, labels),
        "head_body_tail_accuracy": {
            name: _macro_accuracy(correct, labels, classes)
            for name, classes in groups.items()
        },
        "ece_15": expected_calibration_error(probabilities, labels, 15, False),
        "adaptive_ece_15": expected_calibration_error(probabilities, labels, 15, True),
        "nll": float(F.nll_loss(probabilities.clamp_min(1e-9).log(), labels)),
        "brier": float((
            probabilities.square().sum(dim=1)
            - 2.0 * probabilities[torch.arange(labels.numel()), labels]
            + 1.0
        ).mean()),
        "temperature": float(temperature),
        "mean_confidence": float(confidence.mean()),
        "prediction_concentration": _prediction_concentration(prediction),
        "bootstrap_95ci": {
            "top1_accuracy": _bootstrap_image_accuracy(correct, image_ids),
        },
        "bootstrap_unit": "image",
        "bootstrap_estimand": "object_micro_accuracy",
        "class_support": {str(key): value for key, value in sorted(class_counts.items())},
        "top_confusions": [
            {"target": target, "prediction": predicted, "count": count}
            for (target, predicted), count in confusion.most_common(25)
        ],
    }
    if areas is not None:
        area_groups = {
            "small": areas < 0.02,
            "medium": (areas >= 0.02) & (areas < 0.10),
            "large": areas >= 0.10,
        }
        result["area_accuracy"] = {
            name: {
                "support": int(mask.sum()),
                "top1_accuracy": (
                    float(correct[mask].float().mean()) if bool(mask.any()) else float("nan")
                ),
            }
            for name, mask in area_groups.items()
        }
    if mask_iou is not None:
        finite = torch.isfinite(mask_iou)
        boundaries = (0.0, 0.5, 0.7, 0.85, 0.95, 1.000001)
        bins = {}
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            selected = finite & (mask_iou >= lower) & (mask_iou < upper)
            key = f"[{lower:.2f},{min(upper, 1.0):.2f}{']' if upper > 1.0 else ')'}"
            bins[key] = {
                "support": int(selected.sum()),
                "classes": int(labels[selected].unique().numel()) if bool(selected.any()) else 0,
                "top1_accuracy": (
                    float(correct[selected].float().mean())
                    if bool(selected.any()) else float("nan")
                ),
                "ece_15": (
                    expected_calibration_error(probabilities[selected], labels[selected], 15)
                    if int(selected.sum()) >= 2 else float("nan")
                ),
            }
        result["mask_iou_bins"] = bins
        result["wrong_given_good_mask"] = {}
        for threshold in (0.75, 0.85, 0.90, 0.95):
            selected = finite & (mask_iou >= threshold)
            selected_image_ids = [
                image_id for image_id, keep in zip(image_ids, selected.tolist()) if keep
            ]
            selected_errors = (~correct[selected]).float()
            result["wrong_given_good_mask"][f"iou>={threshold:.2f}"] = {
                "support": int(selected.sum()),
                "error_rate": (
                    float(selected_errors.mean())
                    if bool(selected.any()) else float("nan")
                ),
                "bootstrap_95ci": (
                    _bootstrap_image_accuracy(
                        selected_errors, selected_image_ids,
                        seed=781 + int(round(threshold * 100)),
                    ) if bool(selected.any()) else [float("nan"), float("nan")]
                ),
                "bootstrap_unit": "image",
                "bootstrap_estimand": "object_micro_error_rate",
            }
    return result


def paired_accuracy_delta(prediction_a: torch.Tensor,
                          prediction_b: torch.Tensor,
                          labels: torch.Tensor, image_ids,
                          seed=211, trials=2000) -> dict:
    """Image-clustered paired delta, B minus A."""
    a = prediction_a.eq(labels).float()
    b = prediction_b.eq(labels).float()
    grouped = {}
    for value_a, value_b, image_id in zip(a.tolist(), b.tolist(), image_ids):
        row = grouped.setdefault(str(image_id), [0.0, 0])
        row[0] += float(value_b) - float(value_a)
        row[1] += 1
    rows = np.asarray([
        grouped[key]
        for key in sorted(grouped)
    ], dtype=np.float64)
    if rows.shape[0] < 2:
        ci = [float("nan"), float("nan")]
    else:
        rng = np.random.default_rng(int(seed))
        values = [
            (lambda sample: sample[:, 0].sum() / sample[:, 1].sum())(
                rows[rng.integers(0, rows.shape[0], rows.shape[0])]
            )
            for _ in range(int(trials))
        ]
        ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "delta_top1": float(b.mean() - a.mean()),
        "bootstrap_95ci": ci,
        "bootstrap_unit": "image",
        "bootstrap_estimand": "paired_object_micro_accuracy_delta",
        "num_images": len(grouped),
    }


def _endpoint_count_summary(correct_counts: list[int]) -> tuple[dict, dict]:
    counts = Counter(
        {2: "both_correct", 1: "one_wrong", 0: "both_wrong"}[value]
        for value in correct_counts
    )
    ordered_counts = {
        key: int(counts.get(key, 0))
        for key in ("both_correct", "one_wrong", "both_wrong")
    }
    total = sum(ordered_counts.values())
    fractions = {
        key: value / total if total else float("nan")
        for key, value in ordered_counts.items()
    }
    return ordered_counts, fractions


def relationship_endpoint_summary(prediction: torch.Tensor, labels: torch.Tensor,
                                  graph_records: list[dict],
                                  mask_iou: torch.Tensor | None = None,
                                  thresholds=(0.75, 0.85, 0.90, 0.95),
                                  bootstrap_seed: int = 941) -> dict:
    """Audit endpoint identity, optionally when both endpoint masks are good."""
    rows = []
    for graph in graph_records:
        start = int(graph["object_start"])
        stop = int(graph["object_stop"])
        local_correct = prediction[start:stop].eq(labels[start:stop])
        local_iou = mask_iou[start:stop] if mask_iou is not None else None
        for pair in graph["rel_pairs"].tolist():
            subject, obj = map(int, pair)
            if not (0 <= subject < local_correct.numel() and 0 <= obj < local_correct.numel()):
                continue
            correct_count = int(local_correct[subject]) + int(local_correct[obj])
            endpoint_iou = None
            if local_iou is not None:
                endpoint_iou = min(
                    float(local_iou[subject]), float(local_iou[obj])
                )
            rows.append({
                "image_id": str(graph["image_id"]),
                "correct_count": correct_count,
                "minimum_endpoint_mask_iou": endpoint_iou,
            })
    counts, fractions = _endpoint_count_summary([
        row["correct_count"] for row in rows
    ])
    endpoint_disagreements = torch.tensor([
        row["correct_count"] < 2 for row in rows
    ], dtype=torch.float32)
    endpoint_image_ids = [row["image_id"] for row in rows]
    result = {
        "num_relations": len(rows),
        "counts": counts,
        "fractions": fractions,
        "endpoint_failure_rate": (
            fractions["one_wrong"] + fractions["both_wrong"]
            if rows else float("nan")
        ),
        "bootstrap_95ci": (
            _bootstrap_image_accuracy(
                endpoint_disagreements, endpoint_image_ids,
                seed=int(bootstrap_seed),
            ) if endpoint_disagreements.numel() else [float("nan"), float("nan")]
        ),
        "bootstrap_unit": "image",
        "bootstrap_estimand": "relation_micro_endpoint_identity_disagreement_rate",
        "scope": "endpoint identity only; relation correctness is Experiment II/IV",
    }
    if mask_iou is None:
        return result

    conditioned = {}
    for threshold in thresholds:
        selected = [
            row for row in rows
            if row["minimum_endpoint_mask_iou"] is not None
            and math.isfinite(row["minimum_endpoint_mask_iou"])
            and row["minimum_endpoint_mask_iou"] >= float(threshold)
        ]
        selected_counts, selected_fractions = _endpoint_count_summary([
            row["correct_count"] for row in selected
        ])
        failures = torch.tensor([
            row["correct_count"] < 2 for row in selected
        ], dtype=torch.float32)
        image_ids = [row["image_id"] for row in selected]
        failure_rate = (
            float(failures.mean()) if failures.numel() else float("nan")
        )
        conditioned[f"both_iou>={float(threshold):.2f}"] = {
            "support": len(selected),
            "num_images": len(set(image_ids)),
            "coverage": len(selected) / len(rows) if rows else float("nan"),
            "counts": selected_counts,
            "fractions": selected_fractions,
            "endpoint_failure_rate": failure_rate,
            "bootstrap_95ci": (
                _bootstrap_image_accuracy(
                    failures, image_ids,
                    seed=int(bootstrap_seed) + int(round(float(threshold) * 100)),
                ) if failures.numel() else [float("nan"), float("nan")]
            ),
            "bootstrap_unit": "image",
            "bootstrap_estimand": "relation_micro_endpoint_failure_rate",
        }
    result["conditioned_on_both_endpoint_mask_iou"] = conditioned
    result["conditioning_note"] = (
        "A relation is eligible only when both GT-box-prompted predicted masks "
        "meet the stated IoU threshold against their ground-truth masks."
    )
    return result
