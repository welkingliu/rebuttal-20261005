"""Fit-only kernel bandwidth and scale; no evaluation inputs in fitting."""
import torch
from torch.nn import functional as F
from vq_math import residual_logits


def landmark_indices(y):
    generator = torch.Generator().manual_seed(17)
    chosen = []
    for label in range(1, 151):
        ids = torch.nonzero(y.cpu() == label).flatten()
        if len(ids):
            chosen.extend(ids[torch.randperm(len(ids), generator=generator)[:4]].tolist())
    if len(chosen) < 2:
        raise ValueError("At least two labeled landmarks required")
    return chosen


def fit_bandwidth(landmarks):
    distance = (1-landmarks @ landmarks.t()).clamp_min(0)
    mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    # Float32 dot products can put identical unit vectors slightly below one.
    values = distance[mask & (distance > 1e-5)]
    if not len(values):
        raise ValueError("Degenerate identical landmarks")
    return max(float(values.median()), 1e-3)


def kernel(x, landmarks, bandwidth):
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    return (-(1-x @ landmarks.t()).clamp_min(0)/bandwidth).exp()


def fit_models(x, native, y, device="cuda"):
    if x.shape != (len(y), 768) or native.shape != (len(y), 151):
        raise ValueError("Feature/ontology shape mismatch")
    if not all(torch.isfinite(v).all() for v in [x, native]):
        raise ValueError("Nonfinite fit inputs")
    if not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=2e-5):
        raise ValueError("Kernel requires normalized frozen features")
    if torch.any((y < -1) | (y > 150)):
        raise ValueError("Invalid training labels")
    chosen = landmark_indices(y)
    x, native, y = x.to(device), native.to(device), y.to(device)
    landmarks = x[chosen]; bandwidth = fit_bandwidth(landmarks)
    width = len(chosen)
    gram = torch.zeros(width, width, dtype=torch.float64, device=device)
    rhs = torch.zeros(width, 150, dtype=torch.float64, device=device)
    mass = 0.; activation_sum = 0.; activation_num = 0
    for start in range(0, len(y), 1024):
        by, bn = y[start:start+1024], native[start:start+1024]
        phi = kernel(x[start:start+1024], landmarks, bandwidth).double()
        positive = by > 0
        target = torch.zeros(len(by), 150, dtype=torch.float64, device=device)
        target[positive] = F.one_hot(by[positive]-1, 150).double()-bn[positive, 1:].softmax(-1).double()
        weight = torch.where(positive, phi.new_tensor(1.), phi.new_tensor(.1))
        gram += phi.t() @ (phi*weight[:, None])
        rhs += phi.t() @ (target*weight[:, None])
        mass += float(weight.sum())
        activation_sum += float(phi.sum()); activation_num += phi.numel()
    gram, rhs = (gram/mass).cpu(), (rhs/mass).cpu()
    rms = gram.diag().sqrt().clamp_min(1e-6)
    states = {}
    for name, scale in [("median_rms_kernel", rms), ("median_kernel", torch.ones_like(rms))]:
        matrix = gram/(scale[:, None]*scale[None, :])
        matrix = (matrix+matrix.t())*.5+.001*torch.eye(width, dtype=torch.float64)
        right = rhs/scale[:, None]
        coefficients = torch.linalg.solve(matrix, right)
        residual = float((matrix@coefficients-right).norm()/right.norm().clamp_min(1e-12))
        if not torch.isfinite(coefficients).all() or residual > 1e-6:
            raise RuntimeError("Unstable ridge solution")
        states[name] = dict(landmarks=landmarks.cpu(), bandwidth=bandwidth,
            scale=scale.float(), coefficients=coefficients.float(), landmark_row_indices=chosen,
            diagnostics=dict(bandwidth=bandwidth, column_rms_min=float(rms.min()),
                column_rms_median=float(rms.median()), column_rms_max=float(rms.max()),
                fit_mean_kernel_activation=activation_sum/activation_num,
                relative_solve_residual=residual, coefficients_norm=float(coefficients.norm()),
                fitting_proposals=len(y), fitting_positive=int((y > 0).sum())))
    return states


def predict_logits(x, native, state, strength):
    if strength == 0:
        return native.clone()
    phi = kernel(x, state["landmarks"].to(x.device), state["bandwidth"])
    phi = phi/state["scale"].to(x.device)
    residual = phi @ state["coefficients"].to(x.device)
    logp = native[:, 1:].log_softmax(-1)
    q = (logp.exp()+strength*residual).clamp_min(1e-8)
    q = q/q.sum(-1, keepdim=True)
    return residual_logits(native, q.log()-logp)
