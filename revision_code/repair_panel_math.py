"""Kernel evidence repair and visually constrained training-prior messages."""
import numpy as np
import torch
from torch.nn import functional as F
from vq_math import residual_logits


STRENGTHS = [0., .25, .5, 1.]


def design(x, landmarks=None):
    if landmarks is None: return torch.cat([x, x.new_ones(len(x), 1)], -1)
    return ((x @ landmarks.t()-1)/.07).exp()


def fit_ridge(x, native, y, kernel, device="cuda"):
    x, native, y = x.to(device), native.to(device), y.to(device)
    generator = torch.Generator(); generator.manual_seed(17)
    landmarks = None; chosen = []
    if kernel:
        for label in range(1, 151):
            ids = torch.nonzero(y == label).flatten().cpu()
            if len(ids): chosen.extend(ids[torch.randperm(len(ids), generator=generator)[:4]].tolist())
        if not chosen: raise ValueError("No labeled kernel landmarks")
        landmarks = x[chosen]
    width = len(landmarks) if kernel else 769
    gram = x.new_zeros(width, width); rhs = x.new_zeros(width, 150); mass = 0.
    for start in range(0, len(y), 1024):
        bx, bn, by = x[start:start+1024], native[start:start+1024], y[start:start+1024]
        phi = design(bx, landmarks); positive = by > 0
        target = torch.zeros_like(bn[:, 1:]); weight = torch.full_like(by, .1, dtype=torch.float32)
        target[positive] = F.one_hot(by[positive]-1, 150).float()-bn[positive, 1:].softmax(-1)
        weight[positive] = 1.
        gram += phi.t() @ (phi*weight[:, None]); rhs += phi.t() @ (target*weight[:, None]); mass += float(weight.sum())
    matrix = (gram/mass).double().cpu()+.001*torch.eye(width, dtype=torch.float64)
    matrix = (matrix+matrix.t())*.5; right = (rhs/mass).double().cpu()
    coefficients = torch.linalg.solve(matrix, right)
    relative = float((matrix@coefficients-right).norm()/right.norm().clamp_min(1e-12))
    if not torch.isfinite(coefficients).all() or relative > 1e-6: raise RuntimeError("Unstable ridge solve")
    return dict(landmarks=landmarks.detach().cpu() if kernel else None, coefficients=coefficients.float(),
        landmark_row_indices=chosen, relative_solve_residual=relative, fitting_proposals=len(y), fitting_positive=int((y > 0).sum()))


def ridge_logits(x, native, state, strength):
    if strength == 0: return native.clone()
    landmarks = state["landmarks"]
    if landmarks is not None: landmarks = landmarks.to(x.device)
    residual = design(x, landmarks) @ state["coefficients"].to(x.device)
    logp = native[:, 1:].log_softmax(-1)
    q = (logp.exp()+strength*residual).clamp_min(1e-8)
    q = q/q.sum(-1, keepdim=True)
    return residual_logits(native, q.log()-logp)


def make_statistics(raws):
    counts = np.zeros((51, 150, 150), dtype=np.float64)
    objects = np.ones(150, dtype=np.float64); images = 0
    for raw in raws:
        labels = raw["gt_labels"].numpy().astype(int)-1
        relations = raw["gt_relations"].numpy().astype(int)
        if np.any((labels < 0) | (labels >= 150)): raise ValueError("GT object ontology changed")
        np.add.at(objects, labels, 1)
        if len(relations):
            if np.any((relations[:, 2] < 1) | (relations[:, 2] > 50)): raise ValueError("Predicate ontology changed")
            np.add.at(counts, (relations[:, 2], labels[relations[:, 0]], labels[relations[:, 1]]), 1)
        images += 1
    return dict(counts=torch.from_numpy(counts).float(), marginal=torch.from_numpy(objects/objects.sum()).float(),
                fitting_images=images, gt_relations=int(counts.sum()))


def prior_logits(x, native, raw, state, expert, strength, visual):
    if strength == 0: return native.clone()
    p = native.softmax(-1); conditional = native[:, 1:].softmax(-1)
    labels = conditional.argmax(-1); score = p[:, 1:].max(-1)[0]
    pairs = raw["pairs"].to(x.device).long(); rel = raw["relation_logits"].to(x.device).float().softmax(-1)
    confidence, predicate = rel[:, 1:].max(-1); predicate += 1
    counts = state["counts"].to(x.device); prior = state["marginal"].to(x.device)
    if not len(pairs): return native.clone()
    nodes = torch.cat([pairs[:, 0], pairs[:, 1]])
    neighbors = torch.cat([pairs[:, 1], pairs[:, 0]])
    c = torch.cat([counts[predicate, :, labels[pairs[:, 1]]], counts[predicate, labels[pairs[:, 0]], :]])
    messages = (c+5*prior)/(c.sum(-1, keepdim=True)+5)
    weights = confidence.repeat(2)*score[neighbors]
    scores = weights.expand(len(x), -1).masked_fill(nodes[None, :] != torch.arange(len(x), device=x.device)[:, None], -float("inf"))
    order = np.argsort(-scores.detach().cpu().numpy(), axis=-1, kind="stable")[:, :min(3, len(nodes))]
    indices = torch.from_numpy(order.copy()).to(x.device)
    weights = scores.gather(1, indices)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    denom = weights.sum(-1, keepdim=True)
    msg = (messages[indices]*weights[:, :, None]).sum(1)/denom.clamp_min(1e-12)
    delta = torch.tanh((msg+1e-12).log()-(prior+1e-12).log()).clamp_min(0)
    delta = delta*(denom > 0)*(1-conditional.max(-1)[0][:, None])*strength
    if visual:
        q = F.linear(x, expert["weight"].to(x.device), expert["bias"].to(x.device))
        mask_native = torch.zeros_like(delta, dtype=torch.bool); mask_visual = torch.zeros_like(mask_native)
        mask_native.scatter_(1, conditional.topk(10, -1).indices, True)
        mask_visual.scatter_(1, q.topk(10, -1).indices, True)
        delta = delta*(mask_native & mask_visual)
    return residual_logits(native, delta)
