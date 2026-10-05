"""
audits/graph_audit.py
======================
Step 4: Graph-Level Audit — Topological Hallucination

Protocol: Contextual Ablation Protocol
---------------------------------------
For each high-frequency training motif ⟨subj_cls, pred_cls, obj_cls⟩ found in
the test set, the terminal (object) node's visual evidence is removed while
its GT box, label, and graph topology remain fixed. Feature-native models
receive zeroed node/union features; raw-image models receive a mean-filled
GT-box region. The model is re-run on this ablated batch.

Metrics
-------
PIR is prediction invariance over every valid motif instance.  MAR is stricter:
it is conditioned on a clean-correct motif prediction and asks whether that
prediction persists after its terminal visual evidence is removed.  Stable
clean-wrong predictions are reported separately as WSR and never enter MAR.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm


# ── Vocabulary helpers ─────────────────────────────────────────────────────────

def _build_vocab_from_loader(test_loader) -> Tuple[Dict[int, str], Dict[int, str]]:
    ds  = getattr(test_loader, "dataset", None)
    if ds is None:
        return {}, {}
    sgg = getattr(ds, "sgg_dict", None)
    if sgg is None:
        return {}, {}
    ent_map  = {int(k): v for k, v in sgg.get("idx_to_label",     {}).items()}
    pred_map = {int(k): v for k, v in sgg.get("idx_to_predicate", {}).items()}
    return ent_map, pred_map


# ── Pass 0: motif mining ───────────────────────────────────────────────────────

def mine_motifs(motif_loader,
                top_k: int = 20,
                max_batches: int = 300,
                min_support: int = 20) -> List[Tuple[int, int, int]]:
    """
    Mine the top-K most frequent (subj_cls, pred_cls, obj_cls) triplets
    directly from the training motif loader using its integer ontology IDs.
    """
    counter: Counter = Counter()

    for i, batch in enumerate(motif_loader):
        if i >= max_batches:
            break

        entity_lbl = batch.get("entity_labels", None)
        rel_pairs  = batch.get("rel_pairs",     None)
        rel_labels = batch.get("rel_labels",    None)

        if entity_lbl is None or rel_pairs is None or rel_labels is None:
            continue
        if rel_pairs.numel() == 0 or rel_labels.numel() == 0:
            continue

        N = entity_lbl.size(0)
        for m in range(rel_pairs.size(0)):
            s_idx = int(rel_pairs[m, 0].item())
            o_idx = int(rel_pairs[m, 1].item())
            p_cls = int(rel_labels[m].item())
            if p_cls == 0 or s_idx >= N or o_idx >= N:
                continue
            s_cls = int(entity_lbl[s_idx].item())
            o_cls = int(entity_lbl[o_idx].item())
            if s_cls == 0 or o_cls == 0:
                continue
            counter[(s_cls, p_cls, o_cls)] += 1

    if not counter:
        return []

    return [
        triplet for triplet, count in counter.most_common()
        if count >= int(min_support)
    ][:top_k]


def _clustered_rate_ci(image_counts, numerator: str, denominator: str,
                       seed: int = 149, trials: int = 2000) -> list[float]:
    """Bootstrap images, preserving within-image motif dependence."""
    valid = [row for row in image_counts if row.get(denominator, 0) > 0]
    if len(valid) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(int(trials)):
        sample = [valid[index] for index in rng.integers(0, len(valid), len(valid))]
        den = sum(row.get(denominator, 0) for row in sample)
        if den:
            rates.append(sum(row.get(numerator, 0) for row in sample) / den)
    return (
        [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]
        if rates else [float("nan"), float("nan")]
    )


def _clustered_paired_delta(image_counts, terminal: str, control: str,
                            denominator: str, seed: int = 173,
                            trials: int = 2000) -> dict:
    """Image-clustered terminal-minus-control rate difference."""
    valid = [row for row in image_counts if row.get(denominator, 0) > 0]
    if not valid:
        return {
            "delta": float("nan"),
            "bootstrap_95ci": [float("nan"), float("nan")],
            "num_images": 0,
            "support": 0,
        }

    def difference(rows):
        den = sum(row[denominator] for row in rows)
        return (
            sum(row.get(terminal, 0) for row in rows) / den
            - sum(row.get(control, 0) for row in rows) / den
        )

    observed = difference(valid)
    if len(valid) < 2:
        interval = [float("nan"), float("nan")]
    else:
        rng = np.random.default_rng(seed)
        values = [
            difference([
                valid[index]
                for index in rng.integers(0, len(valid), len(valid))
            ])
            for _ in range(int(trials))
        ]
        interval = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return {
        "delta": float(observed),
        "bootstrap_95ci": interval,
        "num_images": len(valid),
        "support": int(sum(row[denominator] for row in valid)),
    }


# ── Batch motif matching ───────────────────────────────────────────────────────

def match_motifs_in_batch(entity_labels: torch.Tensor,
                           rel_pairs:     torch.Tensor,
                           rel_labels:    torch.Tensor,
                           motifs:        List[Tuple[int, int, int]]
                           ) -> List[Tuple[int, Tuple[int, int, int]]]:
    if rel_pairs.numel() == 0 or entity_labels.numel() == 0:
        return []

    N      = entity_labels.size(0)
    M      = rel_pairs.size(0)
    s_arr  = rel_pairs[:, 0].clamp(0, N-1).cpu().numpy()
    o_arr  = rel_pairs[:, 1].clamp(0, N-1).cpu().numpy()
    p_arr  = rel_labels.cpu().numpy() if rel_labels.numel() == M else np.zeros(M, int)
    e_arr  = entity_labels.cpu().numpy()
    motif_set = set(motifs)

    hits = []
    for m_idx in range(M):
        s_cls = int(e_arr[int(s_arr[m_idx])])
        o_cls = int(e_arr[int(o_arr[m_idx])])
        p_cls = int(p_arr[m_idx])
        if p_cls == 0:
            continue
        for motif in motif_set:
            ms, mp, mo = motif
            if (ms == -1 or ms == s_cls) and \
               (mp == -1 or mp == p_cls) and \
               (mo == -1 or mo == o_cls):
                hits.append((m_idx, motif))
                break
    return hits


# ── Terminal-node ablation ─────────────────────────────────────────────────────

def ablate_terminal_node(batch: dict,
                          obj_node_idx: int,
                          pair_indices: List[int]) -> dict:
    """
    Remove endpoint evidence in every representation consumed by supported
    models. Boxes, entity labels, relation pairs, and topology remain fixed.

    The image intervention is explicitly GT-box-conditioned. It is not an
    autonomous segmentation result and must be reported as such.
    """
    b = {k: v.clone() if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}

    visual = b.get("visual_features")
    if isinstance(visual, torch.Tensor) and 0 <= obj_node_idx < visual.size(0):
        visual[obj_node_idx].zero_()

    image = b.get("image")
    boxes = b.get("boxes")
    if (
        isinstance(image, torch.Tensor)
        and isinstance(boxes, torch.Tensor)
        and 0 <= obj_node_idx < boxes.size(0)
    ):
        if image.ndim == 4 and image.size(0) == 1:
            image_view = image[0]
        elif image.ndim == 3:
            image_view = image
        else:
            raise ValueError(
                "GT-box endpoint intervention requires image shape [C,H,W] "
                "or [1,C,H,W]"
            )
        height, width = map(int, image_view.shape[-2:])
        box = boxes[obj_node_idx].detach().float()
        if float(box.max()) <= 1.5:
            box = box * box.new_tensor([width, height, width, height])
        x1, y1, x2, y2 = box.round().long().tolist()
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 > x1 and y2 > y1:
            fill = image_view.mean(dim=(-2, -1), keepdim=True)
            image_view[:, y1:y2, x1:x2] = fill

    union = b.get("union_features", None)
    if union is not None and union.numel() > 0:
        M_u = union.size(0)
        for pi in pair_indices:
            if 0 <= pi < M_u:
                b["union_features"][pi].zero_()

    return b


def _matched_control_node(num_nodes: int, excluded: set[int],
                          image_index: int, terminal_node: int,
                          seed: int = 193,
                          boxes: Optional[torch.Tensor] = None) -> Optional[int]:
    candidates = [index for index in range(num_nodes) if index not in excluded]
    if not candidates:
        return None
    rng = np.random.default_rng(
        int(seed) + 1_000_003 * (int(image_index) + 1) + 97 * int(terminal_node)
    )
    if (
        not isinstance(boxes, torch.Tensor)
        or boxes.ndim != 2
        or boxes.size(0) != num_nodes
    ):
        return int(candidates[int(rng.integers(0, len(candidates)))])

    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
    areas = (sizes[:, 0] * sizes[:, 1]).detach().float().cpu().numpy()
    target = max(float(areas[terminal_node]), 1e-12)
    distances = np.asarray([
        abs(np.log(max(float(areas[index]), 1e-12)) - np.log(target))
        for index in candidates
    ])
    minimum = float(distances.min())
    closest = [
        index for index, distance in zip(candidates, distances)
        if bool(np.isclose(distance, minimum))
    ]
    return int(closest[int(rng.integers(0, len(closest)))])


# ── Hallucination criteria ─────────────────────────────────────────────────────

def _top_k_set(scores: torch.Tensor, row: int, k: int) -> set:
    """Return the set of top-k predicted class indices for one relation."""
    k_eff = min(k, scores.size(-1))
    return set(scores[row].topk(k_eff).indices.cpu().tolist())


def _compute_hall_flags(sc_full:  torch.Tensor,
                         sc_abl:   torch.Tensor,
                         m_idx:    int,
                         gt_cls:   int,
                         pred_cls: int,
                         topk:     int = 5) -> dict:
    """
    Compute all three hallucination criteria for a single relation pair.

    Returns dict with keys:
      hall_strict     — (NEW primary)  pred_after == pred_before
                        regardless of GT correctness.
                        Measures: does ablation change the model's output at all?

      hall_invariant  — pred_after == pred_before  (same as hall_strict, renamed
                        for backward-compat in reporting)

      hall_gt_and_inv — pred_after == pred_before AND pred_after == GT
                        (the OLD criterion — kept for reference only)

      hall_topk       — top-K set unchanged after ablation (softer signal, K=5)

      score_delta     — L1 distance between score vectors (continuous signal)
    """
    pred_before = int(sc_full[m_idx].argmax().item())
    pred_after  = int(sc_abl[m_idx].argmax().item())

    # ── PRIMARY: Visual Invariance (the fix) ──────────────────────────────────
    # Does the model produce the same top-1 prediction regardless of ablation?
    # A purely topology-driven model will: pred_after == pred_before always.
    hall_invariant = (pred_after == pred_before)

    # ── REFERENCE: Old strict criterion (GT-gated) ────────────────────────────
    # Only counts if the model also happened to predict the GT correctly.
    # Artificially low due to Top-1 Accuracy Ceiling (see module docstring).
    hall_gt_inv = (pred_after == pred_before and pred_after == gt_cls)

    # ── SOFT: Top-K set unchanged ──────────────────────────────────────────────
    topk_before = _top_k_set(sc_full, m_idx, topk)
    topk_after  = _top_k_set(sc_abl,  m_idx, topk)
    hall_topk   = (topk_before == topk_after)

    # ── CONTINUOUS: Score vector L1 delta ──────────────────────────────────────
    score_delta = float(
        (sc_full[m_idx] - sc_abl[m_idx]).abs().mean().item())

    return {
        "hall_invariant":  hall_invariant,    # PRIMARY (the fix)
        "hall_strict":     hall_invariant,    # alias used in MAR output
        "hall_gt_inv":     hall_gt_inv,       # OLD criterion (reference only)
        "hall_topk":       hall_topk,         # K=5 soft criterion
        "score_delta":     score_delta,       # continuous signal
        "pred_before":     pred_before,
        "pred_after":      pred_after,
    }


# ── Main audit class ───────────────────────────────────────────────────────────

class GraphLevelAudit:
    """
    Step 4: Graph-Level Audit.

    Pass 0 — motif mining   : mine top-K triplets from real batch IDs
    Pass 1 — ablation eval  : for each matched triplet, zero terminal-node
                              features and measure Visual Invariance

    Primary output metric: MAR = clean-correct predictions that persist after
    terminal evidence ablation / clean-correct motif predictions.  PIR reports
    unconditional prediction invariance and WSR reports clean-wrong stability.
    """

    def __init__(self,
                 top_k_motifs:  int   = 20,
                 mine_batches:  int   = 5000,
                 min_motif_support: int = 20,
                 min_eval_support: int = 100,
                 topk_soft:     int   = 5,
                 control_seed:  int   = 193,
                 device:        str   = "cpu"):
        self.top_k_motifs = top_k_motifs
        self.mine_batches = mine_batches
        self.min_motif_support = int(min_motif_support)
        self.min_eval_support = int(min_eval_support)
        self.topk_soft    = topk_soft
        self.control_seed = int(control_seed)
        self.device       = device
        self._motifs: Optional[List[Tuple[int, int, int]]] = None

    # ── Pass 0 ────────────────────────────────────────────────────────────────
    def _ensure_motifs(self, motif_loader):
        if self._motifs is not None:
            return
        if motif_loader is None:
            self._motifs = []
            return
        print(f"  [GraphAudit] Mining top-{self.top_k_motifs} motifs "
              f"from training data ({self.mine_batches} batches max)…")
        self._motifs = mine_motifs(
            motif_loader, self.top_k_motifs, self.mine_batches,
            self.min_motif_support,
        )

        ent_map, pred_map = _build_vocab_from_loader(motif_loader)
        if not self._motifs:
            print("  [GraphAudit] No motifs mined from the loader. "
                  "Graph audit will report NaN instead of fabricating wildcard motifs.")
            return
        print(f"  [GraphAudit] Mined {len(self._motifs)} motifs. Top 10:")
        for i, (s, p, o) in enumerate(self._motifs[:10]):
            sn = ent_map.get(s, str(s))
            pn = pred_map.get(p, str(p))
            on = ent_map.get(o, str(o))
            print(f"    [{i+1:2d}] ({sn}, {pn}, {on})  ids=({s},{p},{o})")

    def run(self, models: dict, test_loader, motif_loader=None) -> dict:
        self._ensure_motifs(motif_loader)
        ent_map, pred_map = _build_vocab_from_loader(test_loader)
        results = {}
        if not self._motifs:
            for name in models:
                status = (
                    "training_motif_loader_required"
                    if motif_loader is None else "no_motifs_mined"
                )
                results[name] = self._empty_result(status)
            return results
        for name, model in models.items():
            results[name] = self._audit_model(
                name, model, test_loader, ent_map, pred_map)
        return results

    @staticmethod
    def _empty_result(status: str) -> dict:
        return {
            "MAR": float("nan"),
            "PIR": float("nan"),
            "WSR": float("nan"),
            "MAR_gt_inv": float("nan"),
            "MAR_topk": float("nan"),
            "hallucination_rate": float("nan"),
            "mean_score_delta": float("nan"),
            "total_motif_pairs": 0,
            "clean_correct_support": 0,
            "clean_wrong_support": 0,
            "hallucinated_pairs": 0,
            "hallucinated_pairs_gt": 0,
            "hallucinated_pairs_topk": 0,
            "top_motif": "N/A",
            "top_motif_hall_rate": float("nan"),
            "per_motif_inv_rates": {},
            "per_motif_gt_rates": {},
            "motifs_used": 0,
            "matched_negative_control": {
                "status": "unavailable",
                "reason": status,
            },
            "topological_hallucinator": False,
            "motif_source": "training_split",
            "interpretation": "MAR is clean-correct persistence; PIR and clean-wrong stability are reported separately.",
            "status": status,
            "error_count": 0,
            "last_error": None,
        }

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    def _audit_model(self, name: str, model,
                     test_loader, ent_map, pred_map) -> dict:
        model.eval()
        fingerprint_method = getattr(model, "diagnostic_input_fingerprint", None)

        # Accumulators — three criteria
        total_pairs     = 0
        n_invariant     = 0    # PRIMARY: pred unchanged after ablation
        n_gt_inv        = 0    # clean-correct prediction persists
        n_clean_correct = 0
        n_clean_wrong   = 0
        n_wrong_stable  = 0
        n_topk          = 0    # soft: top-K set unchanged
        score_deltas    = []   # continuous: mean score shift
        error_count = 0
        last_error = None

        motif_counts:    Counter = Counter()
        motif_invariant: Counter = Counter()
        motif_gt_inv:    Counter = Counter()
        motif_clean_correct: Counter = Counter()
        motif_topk:      Counter = Counter()
        image_counts = []

        with torch.no_grad():
            for image_index, batch in enumerate(tqdm(
                    test_loader, desc=f"  [GraphAudit] {name}", leave=False)):
                image_stat = Counter()
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                entity_lbl = batch.get("entity_labels", None)
                rel_pairs  = batch.get("rel_pairs",     None)
                rel_labels = batch.get("rel_labels",    None)

                if entity_lbl is None or rel_pairs is None or rel_labels is None:
                    continue
                if rel_pairs.numel() == 0:
                    continue

                # Baseline (full features)
                try:
                    sc_full = model.predict(batch)["pred_rel_scores"]
                except Exception as e:
                    error_count += 1
                    last_error = f"baseline: {type(e).__name__}: {e}"
                    if error_count <= 3:
                        print(f"\n  [GraphAudit:{name}] {last_error}")
                    continue
                if not torch.isfinite(sc_full).all() or sc_full.numel() == 0:
                    continue
                clean_fingerprint = (
                    fingerprint_method(batch) if callable(fingerprint_method) else None
                )

                # Find motif-matching relations
                hits = match_motifs_in_batch(
                    entity_lbl, rel_pairs, rel_labels, self._motifs)
                if not hits:
                    continue

                # Group by terminal node → one forward pass per unique node
                node_to_hits: Dict = defaultdict(list)
                for (m_idx, motif) in hits:
                    o_idx = int(rel_pairs[m_idx, 1].clamp(
                        0, entity_lbl.size(0)-1).item())
                    node_to_hits[o_idx].append((m_idx, motif))

                for o_node, node_hits in node_to_hits.items():
                    pair_indices = [mi for mi, _ in node_hits]
                    batch_abl    = ablate_terminal_node(
                        batch, o_node, pair_indices)

                    excluded_nodes = {int(o_node)}
                    for pair_index in pair_indices:
                        if 0 <= pair_index < rel_pairs.size(0):
                            excluded_nodes.update(
                                int(value) for value in rel_pairs[pair_index].tolist()
                            )
                    control_node = _matched_control_node(
                        entity_lbl.size(0), excluded_nodes,
                        image_index, o_node, self.control_seed,
                        boxes=batch.get("boxes"),
                    )
                    batch_control = (
                        ablate_terminal_node(batch, control_node, pair_indices)
                        if control_node is not None else None
                    )

                    try:
                        if (
                            clean_fingerprint is not None
                            and fingerprint_method(batch_abl) == clean_fingerprint
                        ):
                            raise RuntimeError(
                                "terminal ablation did not change the model-consumed "
                                "visual input fingerprint"
                            )
                        sc_abl = model.predict(batch_abl)["pred_rel_scores"]
                    except Exception as e:
                        error_count += 1
                        last_error = f"ablation: {type(e).__name__}: {e}"
                        if error_count <= 3:
                            print(f"\n  [GraphAudit:{name}] {last_error}")
                        continue
                    if not torch.isfinite(sc_abl).all() or sc_abl.numel() == 0:
                        continue

                    sc_control = None
                    if batch_control is not None:
                        try:
                            if (
                                clean_fingerprint is not None
                                and fingerprint_method(batch_control) == clean_fingerprint
                            ):
                                raise RuntimeError(
                                    "matched control ablation did not change the "
                                    "model-consumed visual input fingerprint"
                                )
                            candidate = model.predict(batch_control)["pred_rel_scores"]
                            if torch.isfinite(candidate).all() and candidate.numel() > 0:
                                sc_control = candidate
                        except Exception as e:
                            error_count += 1
                            last_error = f"control_ablation: {type(e).__name__}: {e}"
                            if error_count <= 3:
                                print(f"\n  [GraphAudit:{name}] {last_error}")

                    M = rel_pairs.size(0)
                    for m_idx, motif in node_hits:
                        if m_idx >= M or m_idx >= sc_abl.size(0):
                            continue

                        gt_cls   = int(rel_labels[m_idx].item()) \
                                   if m_idx < rel_labels.numel() else -1
                        _, pred_cls, _ = motif
                        mk = f"({motif[0]},{motif[1]},{motif[2]})"

                        flags = _compute_hall_flags(
                            sc_full, sc_abl, m_idx, gt_cls, pred_cls,
                            topk=self.topk_soft)

                        control_flags = (
                            _compute_hall_flags(
                                sc_full, sc_control, m_idx, gt_cls, pred_cls,
                                topk=self.topk_soft,
                            )
                            if sc_control is not None and m_idx < sc_control.size(0)
                            else None
                        )

                        motif_counts[mk] += 1
                        total_pairs      += 1
                        image_stat["total"] += 1

                        # PRIMARY: Visual Invariance
                        if flags["hall_invariant"]:
                            n_invariant          += 1
                            motif_invariant[mk]  += 1
                            image_stat["invariant"] += 1

                        if flags["pred_before"] == gt_cls:
                            n_clean_correct += 1
                            motif_clean_correct[mk] += 1
                            image_stat["clean_correct"] += 1
                            if flags["hall_invariant"]:
                                n_gt_inv += 1
                                motif_gt_inv[mk] += 1
                                image_stat["correct_persistent"] += 1
                        else:
                            n_clean_wrong += 1
                            image_stat["clean_wrong"] += 1
                            if flags["hall_invariant"]:
                                n_wrong_stable += 1
                                image_stat["wrong_stable"] += 1

                        # SOFT: top-K unchanged
                        if flags["hall_topk"]:
                            n_topk               += 1
                            motif_topk[mk]       += 1

                        score_deltas.append(flags["score_delta"])
                        if control_flags is not None:
                            image_stat["paired_total"] += 1
                            image_stat["paired_terminal_invariant"] += int(
                                flags["hall_invariant"]
                            )
                            image_stat["paired_control_invariant"] += int(
                                control_flags["hall_invariant"]
                            )
                            if flags["pred_before"] == gt_cls:
                                image_stat["paired_clean_correct"] += 1
                                image_stat["paired_terminal_correct_persistent"] += int(
                                    flags["hall_invariant"]
                                )
                                image_stat["paired_control_correct_persistent"] += int(
                                    control_flags["hall_invariant"]
                                )
                            else:
                                image_stat["paired_clean_wrong"] += 1
                                image_stat["paired_terminal_wrong_stable"] += int(
                                    flags["hall_invariant"]
                                )
                                image_stat["paired_control_wrong_stable"] += int(
                                    control_flags["hall_invariant"]
                                )
                if image_stat["total"]:
                    image_counts.append(dict(image_stat))

        # ── Guard ──────────────────────────────────────────────────────────────
        if total_pairs == 0:
            print(f"  [GraphAudit:{name}] WARNING: zero motif pairs matched.")
            res = self._empty_result("zero_motif_pairs_matched")
            res["error_count"] = error_count
            res["last_error"] = last_error
            return res

        # ── Aggregate ──────────────────────────────────────────────────────────
        PIR = n_invariant / total_pairs
        MAR = (
            n_gt_inv / n_clean_correct
            if n_clean_correct >= self.min_eval_support else float("nan")
        )
        WSR = n_wrong_stable / n_clean_wrong if n_clean_wrong else float("nan")
        MAR_topk = n_topk      / total_pairs
        mean_delta = float(np.mean(score_deltas)) if score_deltas else 0.0
        control_comparison = {
            "strategy": (
                "mask the closest-area deterministic non-endpoint GT-box node "
                "while applying the same selected-pair union-feature ablation"
            ),
            "PIR_terminal_minus_control": _clustered_paired_delta(
                image_counts,
                "paired_terminal_invariant", "paired_control_invariant",
                "paired_total", seed=self.control_seed,
            ),
            "MAR_terminal_minus_control": _clustered_paired_delta(
                image_counts,
                "paired_terminal_correct_persistent",
                "paired_control_correct_persistent",
                "paired_clean_correct", seed=self.control_seed + 1,
            ),
            "WSR_terminal_minus_control": _clustered_paired_delta(
                image_counts,
                "paired_terminal_wrong_stable",
                "paired_control_wrong_stable",
                "paired_clean_wrong", seed=self.control_seed + 2,
            ),
            "interpretation": (
                "A terminal-minus-control delta near zero means terminal removal "
                "is no more influential than the matched non-endpoint control."
            ),
        }

        # Per-motif rates (primary criterion)
        per_motif_inv = {
            mk: round(motif_invariant[mk] / cnt, 4)
            for mk, cnt in motif_counts.items() if cnt > 0
        }
        per_motif_gt = {
            mk: round(motif_gt_inv[mk] / clean_count, 4)
            for mk, clean_count in motif_clean_correct.items() if clean_count > 0
        }

        # Top motif by invariant hallucination rate
        top_mk = max(motif_invariant, key=motif_invariant.get,
                     default="N/A")
        top_rate = per_motif_inv.get(top_mk, 0.0)

        # Decode top motif to human-readable names
        if top_mk != "N/A":
            try:
                ids = [int(x) for x in top_mk.strip("()").split(",")]
                sn  = ent_map.get(ids[0],  str(ids[0]))
                pn  = pred_map.get(ids[1], str(ids[1]))
                on  = ent_map.get(ids[2],  str(ids[2]))
                top_mk_human = f"({sn}, {pn}, {on})"
            except Exception:
                top_mk_human = top_mk
        else:
            top_mk_human = "N/A"

        return {
            # PRIMARY metric (Visual Invariance — the fix)
            "MAR":                    round(MAR, 4) if np.isfinite(MAR) else MAR,
            "PIR":                    round(PIR, 4),
            "WSR":                    round(WSR, 4) if np.isfinite(WSR) else WSR,
            "legacy_hallucination_rate_alias": round(MAR, 4) if np.isfinite(MAR) else MAR,

            # Breakdown
            "MAR_gt_inv":             round(MAR, 4) if np.isfinite(MAR) else MAR,
            "MAR_topk":               round(MAR_topk, 4),  # soft K=5 criterion
            "mean_score_delta":       round(mean_delta, 4),# continuous signal

            # Counts
            "total_motif_pairs":      total_pairs,
            "clean_correct_support":  n_clean_correct,
            "clean_wrong_support":    n_clean_wrong,
            "minimum_eval_support":   self.min_eval_support,
            "hallucinated_pairs":     n_invariant,
            "hallucinated_pairs_gt":  n_gt_inv,
            "wrong_stable_pairs":     n_wrong_stable,
            "hallucinated_pairs_topk": n_topk,

            # Top motif info
            "top_motif":              top_mk_human,
            "top_motif_hall_rate":    top_rate,
            "per_motif_inv_rates":    per_motif_inv,
            "per_motif_gt_rates":     per_motif_gt,
            "motifs_used":            len(self._motifs),

            # Threshold flag (PRIMARY criterion)
            "topological_hallucinator": (
                bool(MAR > 0.5) if np.isfinite(MAR) else None
            ),
            "bootstrap_95ci": {
                "PIR": _clustered_rate_ci(image_counts, "invariant", "total"),
                "MAR": _clustered_rate_ci(
                    image_counts, "correct_persistent", "clean_correct"
                ),
                "WSR": _clustered_rate_ci(
                    image_counts, "wrong_stable", "clean_wrong"
                ),
            },
            "matched_negative_control": control_comparison,
            "motif_source": "training_split",
            "intervention_contract": {
                "space": "raw_image_gt_box_and_proxy_features",
                "endpoint": "object_terminal",
                "fill": "per_image_channel_mean",
                "fixed": [
                    "gt_boxes", "entity_labels", "relation_pairs",
                    "relation_labels", "graph_topology",
                ],
                "consumed_input_fingerprint_required": True,
                "segmentation_claim": "none_gt_box_conditioned",
            },
            "interpretation": (
                "MAR is prediction persistence after terminal visual ablation, "
                "conditioned on a clean-correct motif prediction. It is not by "
                "itself evidence of hallucination because context can be sufficient."
            ),
            "status": (
                "ok" if n_clean_correct >= self.min_eval_support
                else "insufficient_clean_correct_support"
            ),
            "error_count": error_count,
            "last_error": last_error,
        }
