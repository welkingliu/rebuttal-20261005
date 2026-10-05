"""Bounded evidence corrections that retain the native foreground decision."""
import torch
from torch import nn
from torch.nn import functional as F


SPEC = dict(seed=17, batch_size=256, learning_rate=.001, weight_decay=.01,
            gradient_clip=1., kl_weight=1., bound=1., epochs=[0, 1, 2, 4, 8, 16])
PRIMARY = "visual_residual"
ARMS = ["native", "early_expert", PRIMARY, "bias_residual"]


class ResidualHead(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.bias = nn.Parameter(torch.zeros(150))
        if visual: self.weight = nn.Parameter(torch.zeros(150, 768))

    def forward(self, x):
        z = F.linear(x, self.weight, self.bias) if self.visual else self.bias.expand(len(x), -1)
        return SPEC["bound"]*z.tanh()


def residual_logits(native, delta):
    if native.shape != (len(delta), 151) or delta.shape != (len(native), 150):
        raise ValueError("Ontology/shape mismatch")
    fg = native[:, 1:]
    shifted = fg+delta
    adjustment = shifted.logsumexp(-1, keepdim=True)-fg.logsumexp(-1, keepdim=True)
    return torch.cat([native[:, :1], shifted-adjustment], -1)


def objective(native, delta, target):
    if torch.any((target < -1) | (target > 150)): raise ValueError("Invalid targets")
    logits = native[:, 1:]+delta
    positive = target > 0
    ce = F.cross_entropy(logits[positive], target[positive]-1) if positive.any() else logits.sum()*0
    logq = F.log_softmax(logits, -1)
    teacher = native[:, 1:].detach().softmax(-1)
    kl = F.kl_div(logq, teacher, reduction="batchmean")
    return ce+SPEC["kl_weight"]*kl, ce, kl


def choose_epoch(rows):
    if not rows or rows[0]["epoch"] != 0 or len({r["epoch"] for r in rows}) != len(rows):
        raise ValueError("Unique candidates and epoch-zero baseline required")
    base = rows[0]; eligible = []
    for row in rows[1:]:
        if (row["object"] > base["object"]+1e-12
                and row["R50"] >= base["R50"]-.005-1e-12
                and row["mR50"] >= base["mR50"]-.005-1e-12):
            eligible.append(row)
    best = min(eligible, key=lambda r: (-r["object"], r["epoch"])) if eligible else base
    return dict(selected_epoch=best["epoch"], inner_eligible_epochs=[r["epoch"] for r in eligible],
        no_op=best["epoch"] == 0, selection_uses_inner_point_constraints_not_formal_gate=True)


def early_expert_epoch(history):
    if not history: raise ValueError("Missing V-P inner training history")
    return min(history, key=lambda r: (r["inner_weighted_nll"], r["epoch"]))["epoch"]


def fit(x, native, y, visual, epochs, callback=None, device="cuda"):
    torch.manual_seed(SPEC["seed"])
    model = ResidualHead(visual).to(device)
    x, native, y = x.to(device), native.to(device), y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    generator = torch.Generator(); generator.manual_seed(SPEC["seed"])
    history = []
    if callback: callback(model, dict(epoch=0, loss=None, ce=None, kl=None))
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(y), generator=generator).to(device)
        sums = [0., 0., 0.]; count = 0; model.train()
        for indices in order.split(SPEC["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = objective(native[indices], model(x[indices]), y[indices])
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
            loss.backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), SPEC["gradient_clip"])
            if not torch.isfinite(norm): raise RuntimeError("Non-finite gradient")
            optimizer.step()
            for j, value in enumerate([loss, ce, kl]): sums[j] += float(value.detach())*len(indices)
            count += len(indices)
        row = dict(epoch=epoch, **dict(zip(["loss", "ce", "kl"], [s/count for s in sums])))
        history.append(row)
        if callback: callback(model, row)
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()): raise RuntimeError("Invalid residual parameters")
    return state, history
