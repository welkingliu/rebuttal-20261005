"""Conditional visual residuals with foreground/background attribution."""
import torch
from torch import nn
from torch.nn import functional as F

from vq_math import residual_logits, choose_epoch

PRIMARY = "joint_visual"
TRAINED = [PRIMARY, "foreground_visual", "posterior_only"]
ARMS = ["native"] + TRAINED + ["joint_fg_projection", "joint_bg_projection"]
SPEC = dict(seed=17, batch_size=256, learning_rate=.001, weight_decay=.01,
            gradient_clip=1., background_weight=.25, kl_weight=1.,
            epochs=[0, 1, 2, 4, 8, 16], visual_width=1536, classes=151)


def design(visual, native, name):
    if name not in TRAINED or visual.shape != (len(native), 1536) or native.shape[1] != 151:
        raise ValueError("Invalid residual feature/ontology contract")
    # This block gives the head the native decision without GT or expert labels.
    posterior = F.normalize(native.detach().softmax(-1), dim=-1)
    view = torch.zeros_like(visual) if name == "posterior_only" else visual
    return torch.cat([view, posterior], -1)


def make_head(device="cpu"):
    head = nn.Linear(SPEC["visual_width"] + SPEC["classes"], SPEC["classes"]).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def apply_delta(native, delta, name):
    if native.shape != delta.shape or native.shape[1] != 151:
        raise ValueError("Invalid residual logit shape")
    if name == "foreground_visual":
        return residual_logits(native, delta[:, 1:])
    if name in [PRIMARY, "posterior_only"]:
        return native + delta
    raise ValueError("Unknown learned arm")


def project_channels(native, updated, channel):
    """Probability-factor attribution, not a separately fitted repair policy."""
    if channel == "foreground":
        return residual_logits(native, updated[:, 1:] - native[:, 1:])
    if channel == "background":
        # Keep original foreground conditional probabilities, use updated BG odds.
        shift = updated[:, 1:].logsumexp(-1, keepdim=True) - native[:, 1:].logsumexp(-1, keepdim=True)
        return torch.cat([updated[:, :1], native[:, 1:] + shift], -1)
    raise ValueError("Unknown attribution channel")


def objective(native, delta, labels, name):
    if torch.any((labels < -1) | (labels > 150)) or len(labels) != len(native):
        raise ValueError("Invalid training labels")
    updated = apply_delta(native, delta, name)
    if name == "foreground_visual":
        keep = labels > 0
        ce = F.cross_entropy(updated[keep, 1:], labels[keep]-1) if keep.any() else updated.sum()*0
    else:
        keep = labels >= 0
        weights = native.new_ones(151)
        weights[0] = SPEC["background_weight"]
        ce = F.cross_entropy(updated[keep], labels[keep], weight=weights) if keep.any() else updated.sum()*0
    kl = F.kl_div(updated.log_softmax(-1), native.detach().softmax(-1), reduction="batchmean")
    return ce + SPEC["kl_weight"]*kl, ce, kl


def fit(visual, native, labels, name, epochs, callback=None, device="cuda"):
    if name not in TRAINED or epochs < 0:
        raise ValueError("Invalid residual fit")
    torch.manual_seed(SPEC["seed"])
    head = make_head(device)
    native, labels = native.to(device), labels.to(device)
    x = design(visual.to(device), native, name)
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    generator = torch.Generator().manual_seed(SPEC["seed"])
    history = []
    if callback:
        callback(head, dict(epoch=0, loss=None, ce=None, kl=None))
    for epoch in range(1, epochs+1):
        sums = [0., 0., 0.]
        for ix in torch.randperm(len(labels), generator=generator).to(device).split(SPEC["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = objective(native[ix], head(x[ix]), labels[ix], name)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite residual loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(), SPEC["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite residual gradient")
            optimizer.step()
            for j, value in enumerate([loss, ce, kl]):
                sums[j] += float(value.detach())*len(ix)
        row = dict(epoch=epoch, **dict(zip(["loss", "ce", "kl"], [v/len(labels) for v in sums])))
        history.append(row)
        if callback:
            callback(head, row)
    state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError("Invalid residual weights")
    return state, history


def predict(visual, native, head, name):
    return apply_delta(native, head(design(visual, native, name)), name)
