"""
audits/pair_audit.py
=====================
Step 3: Pair-Level Audit — Bias Dependency

Protocol — Expanded Perturbation Suite
---------------------------------------
Previous version only perturbed batch["boxes"], which models largely ignore
because their visual encoder operates on batch["visual_features"] (RoI-pool
or proxy features). Result: BRR = 1.0 for all models (covert immunity).

FIX — Anomaly 2 (BRR = 1.0):
  Root cause: spatial bounding box perturbation had zero effect because all
  SGG models (MOTIFS, GPS-Net, PE-Net, …) feed `visual_features` and
  `entity_labels` through their encoders; `boxes` only enters through an
  optional positional encoding added in Transformer/DINOv2 variants.
  Perturbing only boxes therefore touches a near-zero-weight input path.

  Fix: Expand the perturbation chain to directly attack visual_features
  with three complementary strategies:

  Strategy 1 — Visual Feature Noise Injection (primary):
    Add Gaussian noise to visual_features with a large std (noise_std).
    This degrades the quality of RoI embeddings while preserving
    their statistical structure, forcing the model to rely on priors.

  Strategy 2 — Pair-Targeted Visual Swap (counterfactual):
    For each relation pair (s, o), swap the visual features of s and o.
    A model doing genuine visual grounding must predict differently;
    a bias-dependent model predicts the same (driven by label priors).

  Strategy 3 — Relation-Feature Zero-Masking (ablation):
    Zero the union_features for all pairs. This removes the explicit
    pair-level visual signal that most SGG heads condition on.

  All three are combined into a single "full perturbation" that
  constitutes an unambiguous visual grounding stress test.

Metric: Bias-Recall Ratio (BRR)
--------------------------------
    BRR = R@k_perturbed / R@k_original

    BRR ≈ 1 → model barely changes under direct visual perturbation
              → prediction is driven by frequency priors / label bias
    BRR ≈ 0 → model relies on visual features for correct prediction

Strategy-level BRR breakdown also reported for interpretability.
"""

from collections import Counter
from typing import Dict, Tuple, List, Sequence
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# ── Recall metrics ────────────────────────────────────────────────────────────

DEFAULT_RECALL_KS = (1, 5, 10)
DEFAULT_PERTURBATION_SEEDS = (17, 29, 43)


def derive_batch_seed(base_seed: int, batch_index: int) -> int:
    """Derive a reproducible per-image RNG seed without resetting the stream."""
    modulus = 2**63 - 1
    return int((int(base_seed) + 1_000_003 * (int(batch_index) + 1)) % modulus)


def _normalise_recall_ks(ks: Sequence[int]) -> Tuple[int, ...]:
    normalised = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not normalised:
        raise ValueError("recall_ks must contain at least one positive integer")
    return normalised

def recall_at_k(pred_scores: torch.Tensor,
                gt_labels:   torch.Tensor,
                k: int = 20) -> float:
    """Predicate Hit@K on ground-truth relation pairs.

    This helper does not implement image-level SGG Recall@K because candidate
    object pairs and triplet scores are not available in this audit interface.
    """
    if pred_scores.numel() == 0 or gt_labels.numel() == 0:
        return 0.0
    if pred_scores.size(-1) <= 1:
        return 0.0
    foreground = pred_scores[:, 1:]
    top_k = min(k, foreground.size(-1))
    top_p = foreground.topk(top_k, dim=-1).indices + 1   # [M, K]
    gt_exp   = gt_labels.unsqueeze(1).expand_as(top_p)
    hits     = (top_p == gt_exp).any(dim=-1).float()
    valid    = gt_labels > 0
    if valid.sum() == 0:
        return 0.0
    return float(hits[valid].mean().item())


def mean_recall_at_k(pred_scores: torch.Tensor,
                     gt_labels:   torch.Tensor,
                     k: int = 20,
                     num_classes: int = 51) -> float:
    """Per-class mean predicate Hit@K — less dominated by head predicates."""
    if pred_scores.numel() == 0 or gt_labels.numel() == 0:
        return 0.0
    if pred_scores.size(-1) <= 1:
        return 0.0
    foreground = pred_scores[:, 1:]
    top_k = min(k, foreground.size(-1))
    top_p = foreground.topk(top_k, dim=-1).indices + 1
    per_cls  = []
    for c in range(1, num_classes):
        mask = gt_labels == c
        if mask.sum() == 0:
            continue
        gt_e = gt_labels[mask].unsqueeze(1).expand(mask.sum(), top_k)
        hits = (top_p[mask] == gt_e).any(dim=-1).float()
        per_cls.append(hits.mean().item())
    return float(np.mean(per_cls)) if per_cls else 0.0


def _new_recall_stats(ks: Sequence[int]) -> dict:
    ks = _normalise_recall_ks(ks)
    return {
        "ks": ks,
        "hits": {k: 0 for k in ks},
        "total": 0,
        "class_hits": {k: Counter() for k in ks},
        "class_total": Counter(),
        "invalid_labels": 0,
    }


def _update_recall_stats(stats: dict,
                         pred_scores: torch.Tensor,
                         gt_labels: torch.Tensor) -> None:
    """Accumulate relation-level hits without averaging batch means."""
    if pred_scores.ndim != 2:
        raise ValueError(f"pred_rel_scores must be [M, C], got {tuple(pred_scores.shape)}")
    if gt_labels.ndim != 1:
        gt_labels = gt_labels.reshape(-1)
    if pred_scores.size(0) != gt_labels.numel():
        raise ValueError(
            "Prediction/label row mismatch: "
            f"{pred_scores.size(0)} scores vs {gt_labels.numel()} labels"
        )

    num_classes = pred_scores.size(1)
    valid = (gt_labels > 0) & (gt_labels < num_classes)
    stats["invalid_labels"] += int((~valid).sum().item())
    if not bool(valid.any()):
        return

    labels = gt_labels[valid].long()
    scores = pred_scores[valid, 1:]
    max_k = min(max(stats["ks"]), num_classes - 1)
    top_predictions = scores.topk(max_k, dim=-1).indices + 1

    stats["total"] += int(labels.numel())
    stats["class_total"].update(labels.detach().cpu().tolist())
    for k in stats["ks"]:
        effective_k = min(k, num_classes - 1)
        hits = (top_predictions[:, :effective_k] == labels[:, None]).any(dim=1)
        stats["hits"][k] += int(hits.sum().item())
        hit_labels = labels[hits].detach().cpu().tolist()
        stats["class_hits"][k].update(hit_labels)


def _summarise_recall_stats(stats: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    recall = {}
    mean_recall = {}
    total = stats["total"]
    for k in stats["ks"]:
        key = str(k)
        recall[key] = round(stats["hits"][k] / total, 4) if total else float("nan")
        class_values = [
            stats["class_hits"][k][class_id] / class_total
            for class_id, class_total in stats["class_total"].items()
            if class_total > 0
        ]
        mean_recall[key] = round(float(np.mean(class_values)), 4) \
            if class_values else float("nan")
    return recall, mean_recall


def _paired_bootstrap(clean_values, perturbed_values, minimum_clean: float,
                      seed: int = 101, trials: int = 2000) -> dict:
    """Cluster bootstrap over images for paired recall drops and BRR."""
    clean = np.asarray(clean_values, dtype=np.float64)
    perturbed = np.asarray(perturbed_values, dtype=np.float64)
    if clean.size < 2 or clean.size != perturbed.size:
        return {"status": "insufficient_images", "num_images": int(clean.size)}
    rng = np.random.default_rng(seed)
    drops, ratios = [], []
    for _ in range(int(trials)):
        indices = rng.integers(0, clean.size, size=clean.size)
        clean_mean = float(clean[indices].mean())
        perturbed_mean = float(perturbed[indices].mean())
        drops.append(clean_mean - perturbed_mean)
        if clean_mean >= minimum_clean:
            ratios.append(perturbed_mean / clean_mean)
    return {
        "status": "ok" if ratios else "clean_recall_below_threshold",
        "num_images": int(clean.size),
        "recall_drop_95ci": [
            float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))
        ],
        "BRR_95ci": (
            [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]
            if ratios else [float("nan"), float("nan")]
        ),
    }


# ── Perturbation strategies ───────────────────────────────────────────────────

class VisualPerturbation:
    """
    Expanded perturbation suite operating on visual_features directly.

    All strategies return a cloned batch — originals are never mutated.
    """

    def __init__(self, noise_std: float = 1.0):
        self.noise_std = noise_std

    # ── Strategy 1: Gaussian noise on visual features ────────────────────────
    @staticmethod
    def _generator(tensor: torch.Tensor, seed: int) -> torch.Generator:
        device = tensor.device.type if tensor.device.type != "mps" else "cpu"
        return torch.Generator(device=device).manual_seed(int(seed))

    def inject_visual_noise(self, batch: dict,
                            strength: float = 1.0,
                            seed: int = 17) -> dict:
        """
        Add Gaussian noise (std = noise_std × feature std) to all
        visual_features.  Std is computed per-batch to be scale-invariant.
        """
        b   = _clone(batch)
        vf = b.get("visual_features")
        if isinstance(vf, torch.Tensor):
            std = max(vf.std().item(), 0.01)
            generator = self._generator(vf, seed)
            noise = torch.randn(vf.shape, generator=generator,
                                device=vf.device, dtype=vf.dtype)
            b["visual_features"] = vf + (
                noise * (float(strength) * self.noise_std * std)
            )
        image = b.get("image")
        if isinstance(image, torch.Tensor):
            generator = self._generator(image, seed + 7919)
            noise = torch.randn(image.shape, generator=generator,
                                device=image.device, dtype=image.dtype)
            scale = max(float(image.std().item()), 1.0 / 255.0)
            perturbed = image + noise * float(strength) * self.noise_std * scale
            if float(image.min()) >= 0.0 and float(image.max()) <= 1.0:
                perturbed = perturbed.clamp(0.0, 1.0)
            b["image"] = perturbed
        if not isinstance(vf, torch.Tensor) and not isinstance(image, torch.Tensor):
            raise KeyError("visual noise requires visual_features or image")
        return b

    @staticmethod
    def _pixel_box(box: torch.Tensor, height: int, width: int):
        values = box.detach().float()
        if float(values.max()) <= 1.5:
            values = values * values.new_tensor([width, height, width, height])
        x1, y1, x2, y2 = values.round().long().tolist()
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        return x1, y1, x2, y2

    def _attenuate_image_nodes(self, batch: dict, indices: torch.Tensor,
                               strength: float) -> None:
        image = batch.get("image")
        boxes = batch.get("boxes")
        if not isinstance(image, torch.Tensor) or not isinstance(boxes, torch.Tensor):
            return
        if image.ndim != 3:
            raise ValueError("node image intervention requires image shape [C,H,W]")
        height, width = map(int, image.shape[-2:])
        alpha = float(np.clip(strength, 0.0, 1.0))
        fill = image.mean(dim=(-2, -1), keepdim=True)
        for index in indices.tolist():
            x1, y1, x2, y2 = self._pixel_box(boxes[int(index)], height, width)
            if x2 <= x1 or y2 <= y1:
                continue
            region = image[:, y1:y2, x1:x2]
            image[:, y1:y2, x1:x2] = (1.0 - alpha) * region + alpha * fill

    def _replace_image_nodes(self, batch: dict, permutation: torch.Tensor,
                             strength: float) -> None:
        image = batch.get("image")
        boxes = batch.get("boxes")
        if not isinstance(image, torch.Tensor) or not isinstance(boxes, torch.Tensor):
            return
        if image.ndim != 3:
            raise ValueError("node image replacement requires image shape [C,H,W]")
        import torch.nn.functional as functional
        height, width = map(int, image.shape[-2:])
        alpha = float(np.clip(strength, 0.0, 1.0))
        source_image = image.clone()
        for target_index, source_index in enumerate(permutation.tolist()):
            tx1, ty1, tx2, ty2 = self._pixel_box(
                boxes[target_index], height, width
            )
            sx1, sy1, sx2, sy2 = self._pixel_box(
                boxes[int(source_index)], height, width
            )
            if min(tx2 - tx1, ty2 - ty1, sx2 - sx1, sy2 - sy1) <= 0:
                continue
            source = source_image[:, sy1:sy2, sx1:sx2].unsqueeze(0)
            replacement = functional.interpolate(
                source, size=(ty2 - ty1, tx2 - tx1), mode="bilinear",
                align_corners=False,
            )[0]
            target = image[:, ty1:ty2, tx1:tx2]
            image[:, ty1:ty2, tx1:tx2] = (
                (1.0 - alpha) * target + alpha * replacement
            )

    # ── Strategy 2: Pair-targeted visual feature swap ────────────────────────
    def swap_pair_visual_features(self, batch: dict) -> dict:
        """
        For every relation pair (s, o) swap visual_features[s] ↔ [o].
        This is the visual analogue of bounding-box inversion.

        A truly visually-grounded model must now predict a different predicate
        (or at least score it differently) because the object identities have
        been exchanged. A bias-dependent model predicts the same because it
        relies on entity label priors, not visual appearance.
        """
        b        = _clone(batch)
        vf       = b["visual_features"].clone()           # [N, D]
        pairs    = b.get("rel_pairs", None)
        if pairs is None or pairs.numel() == 0:
            return b
        N = vf.size(0)
        used = set()
        swaps = []
        for pair in pairs:
            s = int(pair[0].clamp(0, N-1).item())
            o = int(pair[1].clamp(0, N-1).item())
            if s != o and s not in used and o not in used:
                swaps.append((s, o))
                used.update((s, o))
        source = vf.clone()
        for s, o in swaps:
            vf[s], vf[o] = source[o], source[s]
        b["visual_features"] = vf
        if swaps:
            permutation = torch.arange(N, device=vf.device)
            for s, o in swaps:
                permutation[s], permutation[o] = permutation[o].clone(), permutation[s].clone()
            self._replace_image_nodes(b, permutation, strength=1.0)
        return b

    # ── Strategy 3: Zero-mask union features ─────────────────────────────────
    def zero_union_features(self, batch: dict) -> dict:
        """
        Set all union_features to zero, removing the explicit pair-level
        visual signal that most relation heads condition on.
        """
        b = _clone(batch)
        uf = b.get("union_features", None)
        if uf is not None and uf.numel() > 0:
            b["union_features"] = torch.zeros_like(uf)
        # Raw-image models do not consume the proxy union tensor. Mask the
        # corresponding union regions as well so this intervention is visible
        # to the actual backbone.
        b = self.attenuate_union_features(b, strength=1.0)
        return b

    def attenuate_union_features(self, batch: dict, strength: float) -> dict:
        """Continuously remove pair-level visual evidence."""
        b = _clone(batch)
        uf = b.get("union_features", None)
        if uf is not None and uf.numel() > 0:
            b["union_features"] = uf * (1.0 - float(np.clip(strength, 0.0, 1.0)))
        image = b.get("image")
        boxes = b.get("boxes")
        pairs = b.get("rel_pairs")
        if all(isinstance(value, torch.Tensor) for value in (image, boxes, pairs)):
            if image.ndim != 3:
                raise ValueError("union image intervention requires image shape [C,H,W]")
            height, width = map(int, image.shape[-2:])
            alpha = float(np.clip(strength, 0.0, 1.0))
            fill = image.mean(dim=(-2, -1), keepdim=True)
            for subject, obj in pairs.long().tolist():
                union = torch.stack((
                    torch.minimum(boxes[subject, :2], boxes[obj, :2]),
                    torch.maximum(boxes[subject, 2:], boxes[obj, 2:]),
                )).reshape(-1)
                x1, y1, x2, y2 = self._pixel_box(union, height, width)
                if x2 > x1 and y2 > y1:
                    region = image[:, y1:y2, x1:x2]
                    image[:, y1:y2, x1:x2] = (
                        (1.0 - alpha) * region + alpha * fill
                    )
        return b

    def mask_nodes(self, batch: dict, mode: str, strength: float = 1.0,
                   seed: int = 17) -> dict:
        """Mask key, random, or graph-unrelated nodes without changing labels."""
        b = _clone(batch)
        vf = b["visual_features"].clone()
        n = vf.size(0)
        if n == 0:
            return b
        pairs = b.get("rel_pairs", torch.zeros(0, 2, device=vf.device)).long()
        degree = torch.zeros(n, device=vf.device)
        if pairs.numel() > 0:
            degree.scatter_add_(0, pairs[:, 0].clamp(0, n - 1), torch.ones(pairs.size(0), device=vf.device))
            degree.scatter_add_(0, pairs[:, 1].clamp(0, n - 1), torch.ones(pairs.size(0), device=vf.device))
        count = max(1, int(round(n * max(float(strength), 1.0 / n) * 0.25)))
        if mode == "key":
            indices = torch.argsort(degree, descending=True)[:count]
        elif mode == "unrelated":
            candidates = (degree == 0).nonzero(as_tuple=False).flatten()
            indices = candidates[:count]
        elif mode == "random":
            generator = self._generator(vf, seed)
            indices = torch.randperm(n, generator=generator, device=vf.device)[:count]
        else:
            raise ValueError(f"Unknown node mask mode: {mode}")
        if indices.numel() > 0:
            vf[indices] = vf[indices] * (1.0 - float(np.clip(strength, 0.0, 1.0)))
            self._attenuate_image_nodes(b, indices, strength)
        b["visual_features"] = vf
        return b

    def on_manifold_replace(self, batch: dict, strength: float = 1.0,
                            seed: int = 17) -> dict:
        """Replace features using observed node features instead of Gaussian noise."""
        b = _clone(batch)
        vf = b["visual_features"].clone()
        if vf.size(0) < 2:
            return b
        generator = self._generator(vf, seed)
        permutation = torch.randperm(vf.size(0), generator=generator, device=vf.device)
        alpha = float(np.clip(strength, 0.0, 1.0))
        b["visual_features"] = (1.0 - alpha) * vf + alpha * vf[permutation]
        self._replace_image_nodes(b, permutation, strength)
        uf = b.get("union_features", None)
        if isinstance(uf, torch.Tensor) and uf.size(0) > 1:
            up = torch.randperm(uf.size(0), generator=generator, device=uf.device)
            b["union_features"] = (1.0 - alpha) * uf + alpha * uf[up]
        return b

    def color_jitter(self, batch: dict, strength: float = 1.0,
                     seed: int = 17) -> dict:
        """Deterministic brightness/contrast/saturation perturbation on images."""
        b = _clone(batch)
        key = "images" if isinstance(b.get("images"), torch.Tensor) else "image"
        image = b.get(key)
        if not isinstance(image, torch.Tensor):
            raise KeyError("color_jitter requires batch['image'] or batch['images']")
        if image.ndim not in (3, 4):
            raise ValueError("image tensor must have shape [C,H,W] or [B,C,H,W]")
        alpha = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(int(seed))
        contrast = 1.0 + float(rng.uniform(-0.4, 0.4)) * alpha
        brightness = float(rng.uniform(-0.15, 0.15)) * alpha
        saturation = 1.0 + float(rng.uniform(-0.5, 0.5)) * alpha
        spatial_dims = (-2, -1)
        mean = image.mean(dim=spatial_dims, keepdim=True)
        jittered = mean + (image - mean) * contrast
        jittered = jittered + brightness
        channel_dim = 0 if image.ndim == 3 else 1
        if image.size(channel_dim) == 3:
            gray = jittered.mean(dim=channel_dim, keepdim=True)
            jittered = gray + (jittered - gray) * saturation
        if float(image.min()) >= 0.0 and float(image.max()) <= 1.0:
            jittered = jittered.clamp(0.0, 1.0)
        b[key] = jittered
        return b

    # ── Legacy box perturbation (kept for reference / ablation study) ─────────
    def perturb_boxes_only(self, batch: dict) -> dict:
        """Original box-only perturbation (diagnoses spatial-encoding sensitivity)."""
        b     = _clone(batch)
        boxes = b.get("boxes", None)
        if boxes is None or boxes.numel() == 0:
            return b
        # Inversion
        pairs = b.get("rel_pairs", None)
        if pairs is not None and pairs.numel() > 0:
            bx = boxes.clone()
            N  = boxes.size(0)
            for pair in pairs:
                s = int(pair[0].clamp(0, N-1).item())
                o = int(pair[1].clamp(0, N-1).item())
                bx[s], bx[o] = bx[o].clone(), bx[s].clone()
            b["boxes"] = bx
        # Noise
        generator = self._generator(b["boxes"], 17)
        noise = torch.randn(b["boxes"].shape, generator=generator,
                            device=b["boxes"].device, dtype=b["boxes"].dtype) * 0.3
        b["boxes"] = (b["boxes"] + noise).clamp(0.0, 1.0)
        return b

    # ── Full combined perturbation ────────────────────────────────────────────
    def full_perturbation(self, batch: dict, seed: int = 17) -> dict:
        """
        Apply all three visual perturbation strategies sequentially.
        This constitutes the definitive BRR measurement.
        """
        b = self.inject_visual_noise(batch, strength=1.0, seed=seed)
        b = self.swap_pair_visual_features(b)
        b = self.zero_union_features(b)
        return b


def _clone(batch: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── Main audit class ──────────────────────────────────────────────────────────

class PairLevelAudit:
    """
    Step 3: Pair-Level Audit — measures reliance on visual features vs. priors.

    Computes per-strategy recall drop and BRR to pinpoint exactly which
    visual signal each model depends on.

    Reported metrics:
      original_recall_at_k  — clean predicate Hit@K for every configured K
      perturbed_recall_at_k — full-perturbation predicate Hit@K
      recall_drop        — R_orig − R_pert
      BRR                — R_pert / R_orig  (high = bias-dependent)
      brr_noise          — BRR under visual noise only
      brr_swap           — BRR under pair-visual-swap only
      brr_union_zero     — BRR under union-feature zeroing only
      brr_boxes_only     — BRR under legacy box perturbation (reference)
    """

    def __init__(self,
                 noise_std: float = 1.0,
                 recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
                 primary_k: int = 20,
                 min_relations: int = 200,
                 min_clean_recall: float = 0.01,
                 perturbation_seeds: Sequence[int] = DEFAULT_PERTURBATION_SEEDS,
                 device:    str   = "cpu"):
        self.pert   = VisualPerturbation(noise_std=noise_std)
        self.recall_ks = _normalise_recall_ks(recall_ks)
        if max(self.recall_ks) > 10:
            raise ValueError(
                "GT-pair predicate Hit@K is only defined here for K<=10. "
                "Use StandardSGGAudit for image-level R@20/50/100."
            )
        if primary_k not in self.recall_ks:
            raise ValueError(
                f"primary_k={primary_k} must be included in recall_ks={self.recall_ks}"
            )
        self.primary_k = int(primary_k)
        self.min_relations = int(min_relations)
        self.min_clean_recall = float(min_clean_recall)
        self.perturbation_seeds = tuple(int(seed) for seed in perturbation_seeds)
        if not self.perturbation_seeds:
            raise ValueError("at least one perturbation seed is required")
        self.device = device

    def run(self, models: dict, test_loader) -> dict:
        results = {}
        for name, model in models.items():
            results[name] = self._audit_model(name, model, test_loader)
        return results

    def _audit_model(self, name: str, model, test_loader) -> dict:
        model.eval()

        strategies = ("orig", "full", "noise", "swap", "union_zero", "boxes_only")
        support_method = getattr(model, "supports_perturbation", None)
        strategy_support = {
            strategy: (
                True if strategy == "orig" or not callable(support_method)
                else bool(support_method(strategy))
            )
            for strategy in strategies
        }
        fingerprint_method = getattr(model, "diagnostic_input_fingerprint", None)
        if not strategy_support["full"]:
            return {
                "status": "unsupported_input_contract",
                "unsupported_strategies": ["full"],
                "recall_ks": list(self.recall_ks),
                "primary_k": self.primary_k,
            }
        acc = {strategy: _new_recall_stats(self.recall_ks) for strategy in strategies}
        num_batches = 0
        paired_primary = {"clean": [], "full": []}
        error_count = 0
        last_error = None

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(
                    test_loader, desc=f"  [PairAudit] {name}", leave=False)):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                gt = batch.get("rel_labels", None)
                if gt is None or gt.numel() == 0:
                    continue

                # ── Original prediction ───────────────────────────────────────
                try:
                    s_orig = model.predict(batch)["pred_rel_scores"]
                except Exception as e:
                    error_count += 1
                    last_error = f"{type(e).__name__}: {e}"
                    if error_count <= 3:
                        print(f"\n  [PairAudit:{name}] orig error: {last_error}")
                    continue
                if not torch.isfinite(s_orig).all():
                    print(f"\n  [PairAudit:{name}] non-finite original scores — skip")
                    continue
                if s_orig.numel() == 0:
                    continue
                clean_fingerprint = (
                    fingerprint_method(batch) if callable(fingerprint_method) else None
                )

                # ── Strategy-level predictions ────────────────────────────────
                def _score(perturbed_batch, label):
                    nonlocal error_count, last_error
                    try:
                        if clean_fingerprint is not None:
                            perturbed_fingerprint = fingerprint_method(perturbed_batch)
                            if perturbed_fingerprint == clean_fingerprint:
                                strategy_support[label] = False
                                last_error = (
                                    f"{label}: intervention did not change the "
                                    "model-consumed visual input fingerprint"
                                )
                                return None
                        sc = model.predict(perturbed_batch)["pred_rel_scores"]
                        return sc if torch.isfinite(sc).all() and sc.numel() > 0 \
                               else None
                    except Exception as e:
                        error_count += 1
                        last_error = f"{label}: {type(e).__name__}: {e}"
                        if error_count <= 3:
                            print(f"\n  [PairAudit:{name}] {last_error}")
                        return None

                full_scores = []
                noise_scores = []
                for base_seed in self.perturbation_seeds:
                    effective_seed = derive_batch_seed(base_seed, batch_index)
                    score = _score(
                        self.pert.full_perturbation(batch, effective_seed), "full"
                    )
                    if score is not None:
                        full_scores.append(score)
                    if strategy_support["noise"]:
                        score = _score(
                            self.pert.inject_visual_noise(
                                batch, strength=1.0, seed=effective_seed
                            ),
                            "noise",
                        )
                        if score is not None:
                            noise_scores.append(score)
                s_swap = (
                    _score(self.pert.swap_pair_visual_features(batch), "swap")
                    if strategy_support["swap"] else None
                )
                s_union0 = (
                    _score(self.pert.zero_union_features(batch), "union_zero")
                    if strategy_support["union_zero"] else None
                )
                s_boxes = (
                    _score(self.pert.perturb_boxes_only(batch), "boxes_only")
                    if strategy_support["boxes_only"] else None
                )

                if not full_scores:
                    continue

                paired_primary["clean"].append(
                    recall_at_k(s_orig, gt, self.primary_k)
                )
                paired_primary["full"].append(
                    float(np.mean([
                        recall_at_k(score, gt, self.primary_k)
                        for score in full_scores
                    ]))
                )

                # ── Accumulate ────────────────────────────────────────────────
                try:
                    _update_recall_stats(acc["orig"], s_orig, gt)
                    for score in full_scores:
                        _update_recall_stats(acc["full"], score, gt)
                except ValueError as e:
                    error_count += 1
                    last_error = f"metric: {e}"
                    if error_count <= 3:
                        print(f"\n  [PairAudit:{name}] {last_error}")
                    continue
                num_batches += 1

                strategy_scores = {
                    "noise": noise_scores,
                    "swap": [s_swap] if s_swap is not None else [],
                    "union_zero": [s_union0] if s_union0 is not None else [],
                    "boxes_only": [s_boxes] if s_boxes is not None else [],
                }
                for key, scores in strategy_scores.items():
                    for score in scores:
                        try:
                            _update_recall_stats(acc[key], score, gt)
                        except ValueError as e:
                            error_count += 1
                            last_error = f"metric/{key}: {e}"

        # ── Guard: no valid batches ───────────────────────────────────────────
        if acc["orig"]["total"] == 0:
            print(f"  [PairAudit:{name}] WARNING: no valid batches. "
                  f"Check model forward() for errors or NaN.")
            return {
                "original_recall":  0.0,
                "perturbed_recall": 0.0,
                "recall_drop": 0.0,
                "BRR": float("nan"),
                "error": "no_valid_batches",
                "status": "no_valid_batches",
                "error_count": error_count,
                "last_error": last_error,
                "recall_ks": list(self.recall_ks),
                "primary_k": self.primary_k,
            }

        enough_relations = acc["orig"]["total"] >= self.min_relations

        def _brr(r_p, r_o):
            if not enough_relations or r_o < self.min_clean_recall:
                return float("nan")
            return round(r_p / r_o, 4)

        recall_by_strategy = {}
        mean_recall_by_strategy = {}
        for strategy, strategy_stats in acc.items():
            recall_by_strategy[strategy], mean_recall_by_strategy[strategy] = \
                _summarise_recall_stats(strategy_stats)

        original = recall_by_strategy["orig"]
        full = recall_by_strategy["full"]
        mean_original = mean_recall_by_strategy["orig"]
        mean_full = mean_recall_by_strategy["full"]
        recall_drop_at_k = {
            str(k): round(original[str(k)] - full[str(k)], 4)
            for k in self.recall_ks
        }
        brr_at_k = {
            str(k): _brr(full[str(k)], original[str(k)])
            for k in self.recall_ks
        }
        mbrr_at_k = {
            str(k): _brr(mean_full[str(k)], mean_original[str(k)])
            for k in self.recall_ks
        }
        strategy_brr_at_k = {
            strategy: {
                str(k): _brr(recall_by_strategy[strategy][str(k)], original[str(k)])
                for k in self.recall_ks
            }
            for strategy in ("noise", "swap", "union_zero", "boxes_only")
        }

        primary = str(self.primary_k)
        R_orig = original[primary]
        R_full = full[primary]
        R_mr_o = mean_original[primary]
        R_mr_f = mean_full[primary]
        drop = recall_drop_at_k[primary]
        BRR = brr_at_k[primary]
        validity_status = "ok"
        if not enough_relations:
            validity_status = "insufficient_relation_support"
        elif R_orig < self.min_clean_recall:
            validity_status = "clean_recall_below_threshold"
        paired_ci = _paired_bootstrap(
            paired_primary["clean"], paired_primary["full"],
            self.min_clean_recall,
        )

        return {
            # Primary metrics
            "metric_definition":       "predicate_hit_rate_on_ground_truth_pairs",
            "recall_ks":               list(self.recall_ks),
            "primary_k":               self.primary_k,
            "perturbation_seeds":      list(self.perturbation_seeds),
            "stochastic_aggregation":  "image-level mean over perturbation seeds",
            "original_recall_at_k":    original,
            "perturbed_recall_at_k":   full,
            "recall_drop_at_k":        recall_drop_at_k,
            "brr_at_k":                brr_at_k,
            "original_recall":        R_orig,
            "perturbed_recall":       R_full,
            "recall_drop":            drop,
            "BRR":                    BRR,
            "bias_dependent":         (
                (BRR > 0.80) if validity_status == "ok" and not np.isnan(BRR)
                else None
            ),
            "validity_status":        validity_status,
            "minimum_relations":      self.min_relations,
            "minimum_clean_recall":   self.min_clean_recall,
            "paired_image_bootstrap": paired_ci,

            # Mean recall (per-class, less biased toward head predicates)
            "mean_recall_original_at_k":  mean_original,
            "mean_recall_perturbed_at_k": mean_full,
            "mbrr_at_k":                   mbrr_at_k,
            "mean_recall_original":   R_mr_o,
            "mean_recall_perturbed":  R_mr_f,
            "mBRR":                   _brr(R_mr_f, R_mr_o),

            # Per-strategy breakdown (diagnostic)
            "strategy_recall_at_k":    {
                strategy: recall_by_strategy[strategy]
                for strategy in ("noise", "swap", "union_zero", "boxes_only")
            },
            "strategy_brr_at_k":       strategy_brr_at_k,
            "brr_noise_only":         strategy_brr_at_k["noise"][primary],
            "brr_swap_only":          strategy_brr_at_k["swap"][primary],
            "brr_union_zero_only":    strategy_brr_at_k["union_zero"][primary],
            "brr_boxes_only":         strategy_brr_at_k["boxes_only"][primary],

            "recall_noise_only":      recall_by_strategy["noise"][primary],
            "recall_swap_only":       recall_by_strategy["swap"][primary],
            "recall_union_zero_only": recall_by_strategy["union_zero"][primary],
            "recall_boxes_only":      recall_by_strategy["boxes_only"][primary],

            "num_batches":            num_batches,
            "num_relations":          acc["orig"]["total"],
            "invalid_gt_labels":      acc["orig"]["invalid_labels"],
            "status": (
                validity_status
                if validity_status != "ok"
                else ("ok" if error_count == 0 else "partial")
            ),
            "strategy_status": {
                strategy: "ok" if strategy_support[strategy] else "unsupported_input_contract"
                for strategy in strategies if strategy != "orig"
            },
            "unsupported_strategies": [
                strategy for strategy in strategies
                if strategy != "orig" and not strategy_support[strategy]
            ],
            "error_count":            error_count,
            "last_error":             last_error,
        }
