"""Grounding-aware objective for joint object/relation fine-tuning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a.float(), dim=-1)
    log_b = F.log_softmax(logits_b.float(), dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    mixture = 0.5 * (prob_a + prob_b)
    log_mixture = mixture.clamp_min(1e-8).log()
    return 0.5 * (
        F.kl_div(log_mixture, prob_a, reduction="batchmean")
        + F.kl_div(log_mixture, prob_b, reduction="batchmean")
    )


class GroundingDependencyRegularizer(nn.Module):
    """Preserve labels under mild noise and suppress evidence-free confidence.

    The destructive intervention has no counterfactual class label. The loss
    therefore does not force an arbitrary alternative predicate. It requires a
    minimum clean/ablated distribution shift and caps confidence after union
    evidence removal. Object supervision and box/mask consistency directly
    target the identity-grounding failure measured by Experiment I-A.
    """

    def __init__(self, mild_weight=0.5, dependency_weight=0.5,
                 uncertainty_weight=0.5, dependency_margin=0.05,
                 max_ablated_confidence=0.5, object_weight=1.0,
                 object_consistency_weight=0.25,
                 object_calibration_weight=0.05,
                 object_focal_gamma=0.0,
                 object_margin_weight=0.0,
                 object_margin=0.1,
                 hard_object_fraction=1.0,
                 detach_clean_consistency_target=True):
        super().__init__()
        self.mild_weight = float(mild_weight)
        self.dependency_weight = float(dependency_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.dependency_margin = float(dependency_margin)
        self.max_ablated_confidence = float(max_ablated_confidence)
        self.object_weight = float(object_weight)
        self.object_consistency_weight = float(object_consistency_weight)
        self.object_calibration_weight = float(object_calibration_weight)
        self.object_focal_gamma = float(object_focal_gamma)
        self.object_margin_weight = float(object_margin_weight)
        self.object_margin = float(object_margin)
        self.hard_object_fraction = float(hard_object_fraction)
        if not 0.0 < self.hard_object_fraction <= 1.0:
            raise ValueError("hard_object_fraction must be in (0, 1]")
        self.detach_clean_consistency_target = bool(
            detach_clean_consistency_target
        )

    def forward(self, clean_logits, mild_logits, ablated_logits, targets,
                object_logits=None, object_targets=None,
                mask_object_logits=None, object_sample_weights=None):
        if not (
            clean_logits.ndim == mild_logits.ndim == ablated_logits.ndim == 2
            and clean_logits.shape == mild_logits.shape == ablated_logits.shape
        ):
            raise ValueError("Relation logits must have one shared [R,C] shape")
        if targets.ndim != 1 or targets.numel() != clean_logits.size(0):
            raise ValueError("Relation logits must align one-to-one with targets")
        valid = (targets > 0) & (targets < clean_logits.size(-1))
        if not bool(valid.any()):
            raise ValueError("No valid foreground predicate targets in this batch")
        clean = clean_logits[valid]
        mild = mild_logits[valid]
        ablated = ablated_logits[valid]
        target = targets[valid]

        supervised = F.cross_entropy(clean, target)
        clean_consistency_target = (
            clean.detach() if self.detach_clean_consistency_target else clean
        )
        mild_consistency = _js_divergence(clean_consistency_target, mild)
        dependency_js = _js_divergence(clean, ablated)
        dependency = F.relu(clean.new_tensor(self.dependency_margin) - dependency_js)
        ablated_confidence = F.softmax(ablated.float(), dim=-1).max(dim=-1).values
        uncertainty = F.relu(
            ablated_confidence - self.max_ablated_confidence
        ).square().mean()
        object_supervised = clean.new_zeros(())
        object_consistency = clean.new_zeros(())
        object_calibration = clean.new_zeros(())
        object_margin_loss = clean.new_zeros(())
        num_objects = 0
        if object_logits is not None or object_targets is not None:
            if object_logits is None or object_targets is None:
                raise ValueError(
                    "object_logits and object_targets must be provided together"
                )
            if object_logits.ndim != 2 or object_targets.ndim != 1:
                raise ValueError("Object logits must be [N,C] and targets [N]")
            if object_logits.size(0) != object_targets.numel():
                raise ValueError("Object logits/targets have different lengths")
            object_valid = (
                (object_targets > 0)
                & (object_targets < object_logits.size(-1))
            )
            if not bool(object_valid.any()):
                raise ValueError("No valid foreground object targets in this batch")
            selected_logits = object_logits[object_valid]
            selected_targets = object_targets[object_valid]
            object_losses = F.cross_entropy(
                selected_logits, selected_targets, reduction="none"
            )
            target_probability = F.softmax(
                selected_logits.float(), dim=-1
            ).gather(1, selected_targets[:, None]).squeeze(1)
            if self.object_focal_gamma > 0.0:
                object_losses = object_losses * (
                    1.0 - target_probability
                ).pow(self.object_focal_gamma)
            if object_sample_weights is not None:
                if (
                    object_sample_weights.ndim != 1
                    or object_sample_weights.numel() != object_targets.numel()
                ):
                    raise ValueError(
                        "object_sample_weights must align with object_targets"
                    )
                selected_weights = object_sample_weights[object_valid].to(
                    device=object_losses.device, dtype=object_losses.dtype
                )
                object_losses = object_losses * selected_weights
                object_supervised = object_losses.sum() / selected_weights.sum()
            else:
                object_supervised = object_losses.mean()
                selected_weights = None
            target_logits = selected_logits.float().gather(
                1, selected_targets[:, None]
            ).squeeze(1)
            wrong_logits = selected_logits.float().clone()
            wrong_logits.scatter_(1, selected_targets[:, None], float("-inf"))
            strongest_wrong_logits = wrong_logits.max(dim=1).values
            margin_losses = F.relu(
                self.object_margin - target_logits + strongest_wrong_logits
            )
            hard_count = max(
                1,
                int(math.ceil(
                    margin_losses.numel() * self.hard_object_fraction
                )),
            )
            hard_indices = margin_losses.topk(
                hard_count, largest=True, sorted=False
            ).indices
            hard_losses = margin_losses[hard_indices]
            if selected_weights is not None:
                hard_weights = selected_weights[hard_indices]
                object_margin_loss = (
                    (hard_losses * hard_weights).sum()
                    / hard_weights.sum().clamp_min(1e-8)
                )
            else:
                object_margin_loss = hard_losses.mean()
            probabilities = F.softmax(selected_logits.float(), dim=-1)
            one_hot = F.one_hot(
                selected_targets, num_classes=selected_logits.size(-1)
            ).float()
            object_calibration = (probabilities - one_hot).square().sum(dim=-1).mean()
            if mask_object_logits is not None:
                if mask_object_logits.shape != object_logits.shape:
                    raise ValueError(
                        "mask_object_logits must match object_logits shape"
                    )
                object_consistency = _js_divergence(
                    selected_logits, mask_object_logits[object_valid]
                )
            num_objects = int(object_valid.sum().item())
        total = (
            supervised
            + self.mild_weight * mild_consistency
            + self.dependency_weight * dependency
            + self.uncertainty_weight * uncertainty
            + self.object_weight * object_supervised
            + self.object_consistency_weight * object_consistency
            + self.object_calibration_weight * object_calibration
            + self.object_margin_weight * object_margin_loss
        )
        return {
            "loss": total,
            "supervised": supervised.detach(),
            "mild_consistency": mild_consistency.detach(),
            "dependency_penalty": dependency.detach(),
            "dependency_js": dependency_js.detach(),
            "ablated_uncertainty": uncertainty.detach(),
            "ablated_mean_confidence": ablated_confidence.mean().detach(),
            "object_supervised": object_supervised.detach(),
            "object_consistency": object_consistency.detach(),
            "object_calibration_proxy": object_calibration.detach(),
            "object_margin": object_margin_loss.detach(),
            "num_relations": int(valid.sum().item()),
            "num_objects": num_objects,
        }
