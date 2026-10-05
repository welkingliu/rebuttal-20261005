"""Dose-response and negative-control audit for relation grounding."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Sequence

import numpy as np
import torch
from tqdm import tqdm

from sgg_core.audits.pair_audit import (
    VisualPerturbation, derive_batch_seed, recall_at_k,
)


DEFAULT_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (17, 29, 43)


def _bootstrap_ci(values, seed=97, trials=1000):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(arr, size=arr.size, replace=True).mean()
        for _ in range(trials)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _new_calibration(num_bins=15):
    return {
        "num_bins": int(num_bins),
        "count": np.zeros(num_bins, dtype=np.int64),
        "confidence": np.zeros(num_bins, dtype=np.float64),
        "correct": np.zeros(num_bins, dtype=np.float64),
        "nll": 0.0,
        "brier": 0.0,
        "total": 0,
    }


def _update_calibration(stats, scores, targets):
    probabilities = torch.softmax(scores.float(), dim=-1)
    valid = (targets > 0) & (targets < probabilities.size(1))
    if not bool(valid.any()):
        return
    probabilities = probabilities[valid]
    targets = targets[valid].long()
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(targets).float()
    bins = torch.clamp(
        (confidence * stats["num_bins"]).long(), max=stats["num_bins"] - 1
    )
    for bin_index in range(stats["num_bins"]):
        mask = bins == bin_index
        if bool(mask.any()):
            stats["count"][bin_index] += int(mask.sum().item())
            stats["confidence"][bin_index] += float(confidence[mask].sum().item())
            stats["correct"][bin_index] += float(correct[mask].sum().item())
    row = torch.arange(targets.numel(), device=targets.device)
    target_probability = probabilities[row, targets].clamp_min(1e-8)
    stats["nll"] += float((-target_probability.log()).sum().item())
    stats["brier"] += float(
        (probabilities.square().sum(dim=-1) - 2 * target_probability + 1).sum().item()
    )
    stats["total"] += int(targets.numel())


def _summarise_calibration(stats):
    total = stats["total"]
    if not total:
        return {"status": "no_valid_relations"}
    ece = 0.0
    for count, confidence_sum, correct_sum in zip(
        stats["count"], stats["confidence"], stats["correct"]
    ):
        if count:
            ece += (count / total) * abs(correct_sum / count - confidence_sum / count)
    return {
        "status": "ok",
        "num_relations": total,
        "top1_accuracy": float(stats["correct"].sum() / total),
        "ece_15": float(ece),
        "nll": float(stats["nll"] / total),
        "brier": float(stats["brier"] / total),
    }


class PerturbationSweepAudit:
    """Evaluate gradual interventions and matched negative controls.

    The reported Hit@K remains a GT-pair diagnostic. Standard graph-level
    performance is reported separately by ``StandardSGGAudit``.
    """

    available_strategies = (
        "visual_noise",
        "color_jitter",
        "union_attenuation",
        "on_manifold_replacement",
        "random_node_mask",
        "key_node_mask",
        "unrelated_node_mask",
    )
    def __init__(self,
                 recall_ks: Sequence[int],
                 levels: Sequence[float] = DEFAULT_LEVELS,
                 seeds: Sequence[int] = DEFAULT_SEEDS,
                 strategies: Sequence[str] | None = None,
                 noise_std: float = 1.0,
                 device: str = "cpu"):
        self.recall_ks = tuple(sorted({int(k) for k in recall_ks if int(k) > 0}))
        self.levels = tuple(sorted({float(np.clip(level, 0.0, 1.0)) for level in levels}))
        if not self.levels or self.levels[0] != 0.0 or self.levels[-1] != 1.0:
            raise ValueError("perturbation levels must include both 0.0 and 1.0")
        self.seeds = tuple(int(seed) for seed in seeds)
        if not self.seeds:
            raise ValueError("at least one perturbation seed is required")
        selected = tuple(dict.fromkeys(str(value) for value in (
            strategies or self.available_strategies
        )))
        unknown = sorted(set(selected) - set(self.available_strategies))
        if unknown or not selected:
            raise ValueError(f"Invalid perturbation strategies: {unknown or selected}")
        self.strategies = selected
        self.required_paper_strategies = selected
        self.perturb = VisualPerturbation(noise_std=noise_std)
        self.device = device

    def run(self, models: dict, test_loader) -> dict:
        return {
            name: self._run_model(name, model, test_loader)
            for name, model in models.items()
        }

    def _apply(self, strategy: str, batch: dict, level: float, seed: int) -> dict:
        if strategy == "visual_noise":
            return self.perturb.inject_visual_noise(batch, level, seed)
        if strategy == "color_jitter":
            return self.perturb.color_jitter(batch, level, seed)
        if strategy == "union_attenuation":
            return self.perturb.attenuate_union_features(batch, level)
        if strategy == "on_manifold_replacement":
            return self.perturb.on_manifold_replace(batch, level, seed)
        if strategy == "random_node_mask":
            return self.perturb.mask_nodes(batch, "random", level, seed)
        if strategy == "key_node_mask":
            return self.perturb.mask_nodes(batch, "key", level, seed)
        if strategy == "unrelated_node_mask":
            return self.perturb.mask_nodes(batch, "unrelated", level, seed)
        raise ValueError(strategy)

    @staticmethod
    def _transition_counts(clean_scores, perturbed_scores, gt):
        clean_prob = torch.softmax(clean_scores.float(), dim=-1)
        pert_prob = torch.softmax(perturbed_scores.float(), dim=-1)
        clean_pred = clean_prob.argmax(dim=-1)
        pert_pred = pert_prob.argmax(dim=-1)
        valid = (gt > 0) & (gt < clean_prob.size(1))
        clean_correct = clean_pred == gt
        pert_correct = pert_pred == gt
        names = {
            "correct_stable": valid & clean_correct & pert_correct,
            "wrong_stable": valid & (~clean_correct) & (~pert_correct) & (clean_pred == pert_pred),
            "correct_to_wrong": valid & clean_correct & (~pert_correct),
            "wrong_to_correct": valid & (~clean_correct) & pert_correct,
            "wrong_changed": valid & (~clean_correct) & (~pert_correct) & (clean_pred != pert_pred),
        }
        counts = {name: int(mask.sum().item()) for name, mask in names.items()}
        wrong_stable = names["wrong_stable"]
        counts["wrong_stable_confidence_sum"] = (
            float(pert_prob[wrong_stable].max(dim=-1).values.sum().item())
            if bool(wrong_stable.any()) else 0.0
        )
        counts["valid"] = int(valid.sum().item())
        return counts

    def _run_model(self, name, model, test_loader):
        support_method = getattr(model, "supports_perturbation", None)
        fingerprint_method = getattr(model, "diagnostic_input_fingerprint", None)
        strategy_support = {
            strategy: (
                bool(support_method(strategy)) if callable(support_method) else True
            )
            for strategy in self.strategies
        }
        clean_values = {k: [] for k in self.recall_ks}
        values = {
            strategy: {
                level: {k: [] for k in self.recall_ks}
                for level in self.levels
            }
            for strategy in self.strategies
        }
        paired_deltas = {
            strategy: {
                level: {k: [] for k in self.recall_ks}
                for level in self.levels
            }
            for strategy in self.strategies
        }
        seed_counts = {
            strategy: {level: [] for level in self.levels}
            for strategy in self.strategies
        }
        transitions = {
            strategy: {level: Counter() for level in self.levels if level > 0}
            for strategy in self.strategies
        }
        clean_calibration = _new_calibration()
        calibrations = {
            strategy: {level: _new_calibration() for level in self.levels}
            for strategy in self.strategies
        }
        errors = []
        strategy_errors = {strategy: [] for strategy in self.strategies}
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(
                    test_loader, desc=f"  [DoseAudit] {name}", leave=False)):
                batch = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                gt = batch.get("rel_labels")
                if gt is None or gt.numel() == 0:
                    continue
                try:
                    clean_scores = model.predict(batch)["pred_rel_scores"]
                    clean_fingerprint = (
                        fingerprint_method(batch)
                        if callable(fingerprint_method) else None
                    )
                    clean_batch = {
                        k: recall_at_k(clean_scores, gt, k)
                        for k in self.recall_ks
                    }
                    _update_calibration(clean_calibration, clean_scores, gt)
                    for k in self.recall_ks:
                        clean_values[k].append(clean_batch[k])
                    for strategy in self.strategies:
                        if not strategy_support[strategy]:
                            continue
                        for level in self.levels:
                            if level == 0.0:
                                for k in self.recall_ks:
                                    values[strategy][level][k].append(clean_batch[k])
                                    paired_deltas[strategy][level][k].append(0.0)
                                seed_counts[strategy][level].append(1)
                                _update_calibration(
                                    calibrations[strategy][level], clean_scores, gt
                                )
                                continue
                            image_seed_values = {
                                k: [] for k in self.recall_ks
                            }
                            successful_seeds = 0
                            for seed in self.seeds:
                                try:
                                    effective_seed = derive_batch_seed(seed, batch_index)
                                    perturbed = self._apply(
                                        strategy, batch, level, effective_seed
                                    )
                                    if clean_fingerprint is not None:
                                        perturbed_fingerprint = fingerprint_method(perturbed)
                                        if perturbed_fingerprint == clean_fingerprint:
                                            raise RuntimeError(
                                                "intervention did not change the model-consumed "
                                                "visual input fingerprint for this image"
                                            )
                                    perturbed_scores = model.predict(perturbed)["pred_rel_scores"]
                                except Exception as exc:
                                    if len(strategy_errors[strategy]) < 5:
                                        strategy_errors[strategy].append(
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                    continue
                                successful_seeds += 1
                                for k in self.recall_ks:
                                    perturbed_value = recall_at_k(perturbed_scores, gt, k)
                                    image_seed_values[k].append(perturbed_value)
                                _update_calibration(
                                    calibrations[strategy][level], perturbed_scores, gt
                                )
                                transitions[strategy][level].update(
                                    self._transition_counts(clean_scores, perturbed_scores, gt)
                                )
                            if successful_seeds:
                                seed_counts[strategy][level].append(successful_seeds)
                                for k in self.recall_ks:
                                    image_mean = float(np.mean(image_seed_values[k]))
                                    values[strategy][level][k].append(image_mean)
                                    paired_deltas[strategy][level][k].append(
                                        clean_batch[k] - image_mean
                                    )
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"{type(exc).__name__}: {exc}")

        if not any(clean_values.values()):
            return {"status": "failed", "errors": errors or ["no valid batches"]}

        clean_mean = {str(k): float(np.mean(clean_values[k])) for k in self.recall_ks}
        strategy_results: Dict[str, dict] = {}
        for strategy in self.strategies:
            if not strategy_support[strategy]:
                strategy_results[strategy] = {
                    "status": "unsupported_input_contract",
                    "curve": {},
                    "auc": {},
                    "errors": [],
                }
                continue
            curves = {}
            auc = {}
            for level in self.levels:
                level_key = f"{level:.3f}"
                curves[level_key] = {}
                for k in self.recall_ks:
                    samples = values[strategy][level][k]
                    deltas = paired_deltas[strategy][level][k]
                    mean_value = float(np.mean(samples)) if samples else float("nan")
                    mean_drop = float(np.mean(deltas)) if deltas else float("nan")
                    curves[level_key][str(k)] = {
                        "mean": mean_value,
                        "std": float(np.std(samples)) if samples else float("nan"),
                        "absolute_drop": mean_drop,
                        "retention_ratio": (
                            mean_value / clean_mean[str(k)]
                            if clean_mean[str(k)] > 1e-8 else float("nan")
                        ),
                        "bootstrap_95ci": _bootstrap_ci(samples),
                        "paired_drop_bootstrap_95ci": _bootstrap_ci(deltas),
                        "n_paired": len(deltas),
                        "bootstrap_unit": "image_after_averaging_perturbation_seeds",
                        "successful_seeds_per_image": {
                            "minimum": (
                                int(min(seed_counts[strategy][level]))
                                if seed_counts[strategy][level] else 0
                            ),
                            "mean": (
                                float(np.mean(seed_counts[strategy][level]))
                                if seed_counts[strategy][level] else 0.0
                            ),
                        },
                    }
                curves[level_key]["calibration"] = _summarise_calibration(
                    calibrations[strategy][level]
                )
                if level > 0:
                    counts = transitions[strategy][level]
                    valid = counts.get("valid", 0)
                    wrong_stable = counts.get("wrong_stable", 0)
                    curves[level_key]["transitions"] = {
                        key: counts.get(key, 0) / valid if valid else float("nan")
                        for key in (
                            "correct_stable", "wrong_stable", "correct_to_wrong",
                            "wrong_to_correct", "wrong_changed",
                        )
                    }
                    curves[level_key]["transitions"]["wrong_stable_mean_confidence"] = (
                        counts.get("wrong_stable_confidence_sum", 0.0) / wrong_stable
                        if wrong_stable else float("nan")
                    )
            x = np.asarray(self.levels, dtype=np.float64)
            for k in self.recall_ks:
                y = np.asarray([
                    curves[f"{level:.3f}"][str(k)]["mean"] for level in self.levels
                ])
                if hasattr(np, "trapezoid"):
                    auc[str(k)] = float(np.trapezoid(y, x))
                else:
                    auc[str(k)] = float(np.trapz(y, x))
            has_perturbed_samples = any(
                values[strategy][level][k]
                for level in self.levels if level > 0
                for k in self.recall_ks
            )
            strategy_results[strategy] = {
                "status": "ok" if has_perturbed_samples else "unsupported_input_contract",
                "curve": curves,
                "auc": auc,
                "errors": strategy_errors[strategy],
            }

        complete = not errors and all(
            strategy_results[strategy]["status"] == "ok"
            for strategy in self.required_paper_strategies
        )
        return {
            "status": "ok" if complete else "partial",
            "metric_definition": "macro_image_predicate_hit_rate_on_ground_truth_pairs",
            "recall_ks": list(self.recall_ks),
            "levels": list(self.levels),
            "seeds": list(self.seeds),
            "inference_unit": "image_seed_pair",
            "confidence_interval": "paired bootstrap over clean-minus-perturbed image/seed effects",
            "clean": clean_mean,
            "clean_calibration": _summarise_calibration(clean_calibration),
            "strategies": strategy_results,
            "unsupported_strategies": [
                strategy for strategy, supported in strategy_support.items()
                if not supported
            ],
            "required_paper_strategies": list(self.required_paper_strategies),
            "errors": errors,
        }
