"""Proposal-expert fitting and label-free probability fusion."""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPEC = dict(seed=17, batch_size=256, learning_rate=.003, weight_decay=.01,
    gradient_clip=1., background_weight=.25, max_epochs=100, min_epochs=10,
    patience=10, min_delta=1e-6, alpha=.5, classes=151, width=768)
PRIMARY = "proposal_fg_fusion"
ARMS = ["native", "old_gt_fusion", PRIMARY, "proposal_full_fusion"]


def split_fold(ids, mapping, fold):
    if len(ids) != len(set(ids)) or set(mapping) != set(ids):
        raise ValueError("Invalid image grouping")
    training = [i for i in ids if mapping[i] != fold]
    held = [i for i in ids if mapping[i] == fold]
    ordered = sorted(training, key=lambda x: hashlib.sha256(("vp_inner_v1:%d:" % fold+x).encode()).digest())
    validation = ordered[:max(1, len(ordered)//10)]
    fit = [i for i in training if i not in set(validation)]
    if not fit or not held or set(training) & set(held):
        raise ValueError("Empty or overlapping split")
    return dict(training=training, fit=fit, inner_validation=validation, held=held)


def make_head(old):
    if old["weight"].shape != (150, 768) or old["bias"].shape != (150,):
        raise ValueError("Old visual expert ontology changed")
    head = nn.Linear(768, 151)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
        head.weight[1:].copy_(old["weight"]); head.bias[1:].copy_(old["bias"])
    return head


def weighted_nll(logits, target):
    if logits.shape != (len(target), 151) or torch.any((target < -1) | (target > 150)):
        raise ValueError("Invalid proposal target/logit contract")
    keep = target >= 0
    if not keep.any():
        raise ValueError("No supervised proposals")
    weights = logits.new_ones(151); weights[0] = SPEC["background_weight"]
    numerator = F.cross_entropy(logits[keep], target[keep], weight=weights, reduction="sum")
    denominator = weights[target[keep]].sum()
    return numerator/denominator, numerator, denominator


def fusion_logits(native, expert, preserve_background=True, alpha=.5):
    """Only predictions enter fusion; never matching labels or box IoUs to GT."""
    if not 0 <= alpha <= 1 or native.ndim != 2 or native.shape[1] != 151:
        raise ValueError("Invalid native logits/alpha")
    if expert.shape not in [(len(native), 150), (len(native), 151)]:
        raise ValueError("Invalid expert vocabulary")
    if not torch.isfinite(native).all() or not torch.isfinite(expert).all():
        raise ValueError("Non-finite probabilities")
    if (expert < 0).any() or not torch.allclose(expert.sum(-1), expert.new_ones(len(expert)), atol=1e-5):
        raise ValueError("Expert input is not a probability distribution")
    if alpha == 0:
        return native.clone()
    p = native.softmax(-1)
    if preserve_background:
        q = expert[:, -150:]; q = q/q.sum(-1, keepdim=True).clamp_min(1e-30)
        mass = p[:, 1:].sum(-1, keepdim=True)
        fused = torch.cat([p[:, :1], (1-alpha)*p[:, 1:]+alpha*mass*q], -1)
    else:
        if expert.shape[1] != 151: raise ValueError("Full fusion needs a background probability")
        fused = (1-alpha)*p+alpha*expert
    result = fused.clamp_min(1e-30).log()
    if not torch.allclose(result.softmax(-1), fused, atol=2e-7, rtol=1e-5):
        raise RuntimeError("Fusion score conversion drift")
    return result


def fit_head(x, y, old, *, validation=None, epochs=None, smoke=False, callback=None, device="cuda"):
    """Only supplied fitting rows enter gradients; validation selects epochs only."""
    torch.manual_seed(SPEC["seed"])
    head = make_head(old).to(device)
    keep = y >= 0; x, y = x[keep].to(device), y[keep].to(device)
    if not len(y): raise ValueError("Empty training data")
    opt = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    if validation is not None:
        vx, vy = validation; valid = vy >= 0
        vx, vy = vx[valid].to(device), vy[valid].to(device)
        if not len(vy): raise ValueError("Empty inner validation")
    elif epochs is None:
        raise ValueError("Refitting requires a preselected epoch count")
    limit = epochs if epochs is not None else (2 if smoke else SPEC["max_epochs"])
    minimum = 1 if smoke else SPEC["min_epochs"]
    generator = torch.Generator(); generator.manual_seed(SPEC["seed"])
    history = []; best, best_epoch, bad = float("inf"), None, 0
    for epoch in range(1, limit+1):
        order = torch.randperm(len(y), generator=generator).to(device)
        total, denom = 0., 0.; head.train()
        for batch in order.split(SPEC["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss, num, den = weighted_nll(head(x[batch]), y[batch])
            if not torch.isfinite(loss): raise RuntimeError("Non-finite classifier loss")
            loss.backward(); norm = nn.utils.clip_grad_norm_(head.parameters(), SPEC["gradient_clip"])
            if not torch.isfinite(norm): raise RuntimeError("Non-finite classifier gradient")
            opt.step(); total += float(num.detach()); denom += float(den.detach())
        row = dict(epoch=epoch, train_weighted_nll=total/denom)
        if validation is not None:
            vtotal, vden, correct, npos = 0., 0., 0, 0; head.eval()
            with torch.no_grad():
                for start in range(0, len(vy), SPEC["batch_size"]):
                    tx, ty = vx[start:start+SPEC["batch_size"]], vy[start:start+SPEC["batch_size"]]
                    logits = head(tx); _, num, den = weighted_nll(logits, ty)
                    vtotal += float(num); vden += float(den)
                    positive = ty > 0
                    correct += int(((logits[:, 1:].argmax(-1)+1 == ty) & positive).sum())
                    npos += int(positive.sum())
            value = vtotal/vden
            row.update(inner_weighted_nll=value, inner_fg_top1=correct/npos if npos else None)
            if epoch >= minimum:
                if value < best-SPEC["min_delta"]: best, best_epoch, bad = value, epoch, 0
                else: bad += 1
            row.update(best_epoch=best_epoch, patience_used=bad)
        history.append(row)
        if callback: callback(row, limit)
        if validation is not None and epoch >= minimum and bad >= SPEC["patience"]: break
    state = {k:v.detach().cpu() for k,v in head.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()): raise RuntimeError("Invalid fitted weights")
    return state, dict(history=history, selected_epochs=best_epoch if validation is not None else epochs,
        hit_epoch_budget=len(history) == limit, best_inner_loss=best if best_epoch is not None else None,
        fitting_proposals=len(y), fitting_foreground=int((y > 0).sum()), fitting_background=int((y == 0).sum()))
