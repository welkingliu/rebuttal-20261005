"""Object-balanced training weights and GT-free residual prediction interfaces."""
import numpy as np
import torch
from torch.nn import functional as F
from vx_common import SPEC, PRIMARY
from vw_math import relation_protection


def balanced_weights(target, assignment, gt_endpoints, diagnostic=False):
    target = np.asarray(target)
    assignment = np.asarray(assignment)
    if target.shape != assignment.shape or target.ndim != 1:
        raise ValueError("Unaligned supervision")
    positive = target > 0
    if np.any(assignment[positive] < 0):
        raise ValueError("Positive proposals require a matching GT instance")
    weight = np.zeros(len(target), dtype=np.float32)
    objects = np.unique(assignment[positive])
    for obj in objects:
        keep = positive & (assignment == obj)
        scale = SPEC["endpoint_weight"] if diagnostic and bool(gt_endpoints[obj]) else 1.
        weight[keep] = scale/keep.sum()
    bg = target == 0
    if bg.any():
        weight[bg] = SPEC["background_mass"]*max(1, len(objects))/bg.sum()
    if weight.sum() <= 0:
        raise ValueError("No supervised proposal")
    return torch.from_numpy(weight)


def objective(logits, meta, arm):
    valid = meta["target"] >= 0
    key = "diagnostic_weight" if arm == PRIMARY else "object_weight"
    weight = meta[key]
    ce = (F.cross_entropy(logits[valid], meta["target"][valid], reduction="none")*weight[valid]).sum()/weight.sum()
    kl = F.kl_div(logits.log_softmax(-1), meta["native"].detach().softmax(-1), reduction="batchmean")
    protect = relation_protection(logits, meta["native"], meta)
    loss = ce + SPEC["kl_weight"]*kl
    if arm == PRIMARY:
        loss = loss + SPEC["relation_weight"]*protect
    return dict(loss=loss, object_ce=ce, native_kl=kl, relation_protection=protect)


def paired_box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("Box row/coordinate mismatch")
    inter = np.maximum(0, np.minimum(a[:, 2:], b[:, 2:])-np.maximum(a[:, :2], b[:, :2])+1).prod(-1)
    area = np.maximum(0, a[:, 2:]-a[:, :2]+1).prod(-1)
    other = np.maximum(0, b[:, 2:]-b[:, :2]+1).prod(-1)
    return inter/np.maximum(area+other-inter, 1e-12)


def tensor_meta(meta, device):
    return {k:v.to(device) for k,v in meta.items() if torch.is_tensor(v)}
