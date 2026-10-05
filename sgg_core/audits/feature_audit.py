"""
audits/feature_audit.py
========================
Step 2: Feature-Level Audit — Identity Annihilation

Metrics
-------
Effective Rank (R)
    Measures the intrinsic dimensionality of the feature space.
    Roy & Vetterli (2007): R = exp(H(σ̄)) where σ̄ are the normalised
    singular values and H is the Shannon entropy.
    Low R → representation collapse (all nodes map to a similar embedding).

Dirichlet Energy (E)
    Measures smoothness/over-smoothing of graph node features.
    E(F) = (1/|E|) Σ_{(i,j)∈E} ||F_i − F_j||²
    Low E → over-smoothing (neighbouring nodes become indistinguishable).

Both metrics are computed on the pre-classifier node feature matrices
extracted during inference on the test set.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def effective_rank(feat_matrix: torch.Tensor) -> float:
    """
    Compute Effective Rank from Roy & Vetterli (2007).

    Args:
        feat_matrix: [N, D] node feature matrix (float32)

    Returns:
        Effective rank (float, in [1, min(N,D)])
    """
    if feat_matrix.shape[0] < 2:
        return 1.0

    feat_f32 = feat_matrix.float()
    try:
        _, S, _ = torch.linalg.svd(feat_f32, full_matrices=False)
    except Exception:
        S = torch.linalg.svdvals(feat_f32)

    S = S.abs()
    S_sum = S.sum()
    if S_sum < 1e-12:
        return 1.0

    p = S / S_sum                          # normalised singular values
    p = p.clamp(min=1e-12)
    H = -(p * p.log()).sum()               # Shannon entropy
    return float(torch.exp(H).item())


def dirichlet_energy(feat_matrix: torch.Tensor,
                     adj: torch.Tensor = None) -> float:
    """
    Compute graph Dirichlet Energy.

    If `adj` is not provided, we build a complete graph as worst-case proxy
    (all-pairs pairwise differences), which gives an upper-bound signal.

    Args:
        feat_matrix : [N, D]
        adj         : [N, N] binary or weighted adjacency (optional)

    Returns:
        Dirichlet energy (float, ≥ 0)
    """
    N = feat_matrix.shape[0]
    if N < 2:
        return 0.0

    feat_f32 = F.normalize(feat_matrix.float(), dim=-1)  # normalise rows

    if adj is not None and adj.numel() > 0:
        adj = adj.to(feat_f32.device).float()
        edge_mask = adj > 0
        num_edges = edge_mask.sum().item()
        if num_edges == 0:
            adj = None
        else:
            # Sparse version: iterate over edges
            rows, cols = edge_mask.nonzero(as_tuple=True)
            diff = feat_f32[rows] - feat_f32[cols]
            energy = (diff ** 2).sum(-1).mean().item()
            return energy

    # All-pairs approximation (complete graph)
    # ||F_i - F_j||² = ||F_i||² + ||F_j||² - 2 F_i·F_j
    # Since rows are normalised: = 2 - 2 F_i·F_j
    gram   = feat_f32 @ feat_f32.T           # [N, N]
    energy = (2 - 2 * gram)                  # [N, N]
    # Exclude diagonal
    mask   = ~torch.eye(N, dtype=torch.bool, device=gram.device)
    energy = energy[mask].mean().item()
    return float(energy)


def rank_collapse_score(feat_matrix: torch.Tensor) -> float:
    """
    Ratio of effective rank to min(N, D): 1 = full rank, 0 = fully collapsed.
    """
    N, D = feat_matrix.shape
    er   = effective_rank(feat_matrix)
    return er / min(N, D)


class FeatureLevelAudit:
    """
    Step 2: Feature-Level Audit

    For each model, accumulates node feature matrices across batches,
    then computes Effective Rank and Dirichlet Energy.
    """

    def __init__(self, device="cpu", max_nodes: int = 5000):
        self.device    = device
        self.max_nodes = max_nodes   # cap to avoid OOM on large test sets

    def run(self, models: dict, test_loader) -> dict:
        results = {}
        for model_name, model in models.items():
            results[model_name] = self._audit_model(model_name, model, test_loader)
        return results

    def _audit_model(self, name: str, model, test_loader) -> dict:
        centered_feats = []
        per_image_rank = []
        per_image_energy = []
        per_image_normalized_rank = []
        per_image_svr = []
        total     = 0
        error_count = 0
        last_error = None

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  [FeatureAudit] {name}", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                try:
                    feats = model.extract_node_features(batch)  # [N, D]
                except Exception as e:
                    error_count += 1
                    last_error = f"{type(e).__name__}: {e}"
                    if error_count <= 3:
                        print(f"\n  [FeatureAudit:{name}] error: {last_error}")
                    continue

                if feats is None or feats.numel() == 0:
                    continue

                feats_cpu = feats.cpu().float()
                centered = feats_cpu - feats_cpu.mean(dim=0, keepdim=True)
                adj = batch.get("graph_adj", None)
                adj_cpu = adj.cpu() if isinstance(adj, torch.Tensor) else None
                per_image_rank.append(effective_rank(centered))
                per_image_energy.append(dirichlet_energy(feats_cpu, adj_cpu))
                per_image_normalized_rank.append(rank_collapse_score(centered))
                per_image_svr.append(self._svd_ratio(centered))
                centered_feats.append(centered)

                total += feats.shape[0]
                if total >= self.max_nodes:
                    break

        if not centered_feats:
            return {
                "effective_rank": float("nan"),
                "dirichlet_energy": float("nan"),
                "rank_collapse_score": float("nan"),
                "num_nodes": 0,
                "status": "no_valid_features",
                "error_count": error_count,
                "last_error": last_error,
            }

        feat_matrix = torch.cat(centered_feats, dim=0)[:self.max_nodes]

        def mean(values):
            return float(np.mean(values)) if values else float("nan")

        def bootstrap_ci(values, trials=1000, seed=23):
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return [float("nan"), float("nan")]
            rng = np.random.default_rng(seed)
            boot = np.asarray([
                rng.choice(arr, size=arr.size, replace=True).mean()
                for _ in range(trials)
            ])
            return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

        er = mean(per_image_rank)
        de = mean(per_image_energy)
        rcs = mean(per_image_normalized_rank)

        return {
            "effective_rank":       round(er,  4),
            "dirichlet_energy":     round(de,  6),
            "rank_collapse_score":  round(rcs, 4),
            "normalized_effective_rank": round(rcs, 4),
            "global_centered_effective_rank": round(effective_rank(feat_matrix), 4),
            "num_nodes":            feat_matrix.shape[0],
            "num_graphs":           len(per_image_rank),
            "feature_dim":          feat_matrix.shape[1],
            "centering":            "per_image_feature_mean",
            "singular_value_normalization": "sigma_over_sum_sigma",
            "feature_mean_norm":    round(float(feat_matrix.norm(dim=-1).mean().item()), 4),
            "feature_std_norm":     round(float(feat_matrix.norm(dim=-1).std().item()), 4),
            # Additional collapse indicators
            "singular_value_ratio": round(mean(per_image_svr), 4),
            "bootstrap_95ci": {
                "effective_rank": bootstrap_ci(per_image_rank),
                "dirichlet_energy": bootstrap_ci(per_image_energy),
                "normalized_effective_rank": bootstrap_ci(per_image_normalized_rank),
            },
            "status": "ok",
            "error_count": error_count,
            "last_error": last_error,
        }

    @staticmethod
    def _svd_ratio(feat: torch.Tensor) -> float:
        """Ratio of top-1 singular value to sum of all: 1 = fully collapsed."""
        try:
            S = torch.linalg.svdvals(feat.float())
            return round(float((S[0] / S.sum()).item()), 4)
        except Exception:
            return float("nan")
