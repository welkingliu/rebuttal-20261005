"""Endpoint-weighted identity learning with differentiable triplet-score protection."""
import torch
from torch import nn
from torch.nn import functional as F

PRIMARY = "endpoint_relation"
TRAINED = ["uniform_ce", "endpoint_ce", PRIMARY]
ARMS = ["native"] + TRAINED
SPEC = dict(seed=17, image_batch=1, accumulation_images=8, learning_rate=.0005,
            weight_decay=.01, gradient_clip=1., background_weight=.25,
            endpoint_weight=3., kl_weight=1., relation_weight=1.,
            epochs=[0, 1, 2, 4, 8], visual_width=1536, classes=151,
            primary=PRIMARY, relation_noninferiority_margin=.005)


def make_head(device="cpu"):
    head = nn.Linear(1687, 151).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def design(visual, native):
    if visual.shape != (len(native), 1536) or native.shape[1] != 151:
        raise ValueError("Wrong feature or ontology shape")
    return torch.cat([visual, F.normalize(native.detach().softmax(-1), dim=-1)], -1)


def predict(head, visual, native):
    # Prediction accepts no GT, matching assignment or training-only protection mask.
    return native + head(design(visual, native))


def pair_scores(logits, pairs, emitted_labels, predicate_logprob):
    logp = logits.log_softmax(-1)
    return (logp[pairs[:, 0], emitted_labels[pairs[:, 0]]]
            + logp[pairs[:, 1], emitted_labels[pairs[:, 1]]] + predicate_logprob)


def relation_protection(updated, native, meta):
    pairs, labels, rel = meta["pairs"], meta["emitted_labels"], meta["predicate_logprob"]
    current = pair_scores(updated, pairs, labels, rel)
    baseline = pair_scores(native.detach(), pairs, labels, rel).detach()
    good = meta["protected"]
    if len(good) != len(pairs):
        raise ValueError("Pair-protection alignment mismatch")
    if not good.any():
        return updated.sum() * 0
    score_loss = F.relu(baseline[good] - current[good]).mean()
    # Protect the native top-50 boundary, not a differentiable claim about NMS/R@K.
    if len(pairs) <= 50:
        return score_loss
    old_margin = baseline[good, None] - baseline[None, 50:]
    new_margin = current[good, None] - current[None, 50:]
    return score_loss + F.relu(old_margin - new_margin).mean()


def terms(updated, native, target, meta, arm):
    if arm not in TRAINED or ((target < -1) | (target > 150)).any():
        raise ValueError("Unknown objective or labels")
    valid = target >= 0
    weight = updated.new_ones(len(target))
    weight[target == 0] = SPEC["background_weight"]
    if arm != "uniform_ce":
        weight = weight * torch.where(meta["endpoint"], weight.new_tensor(SPEC["endpoint_weight"]), weight.new_tensor(1.))
    ce = (F.cross_entropy(updated[valid], target[valid], reduction="none") * weight[valid]).sum() / weight[valid].sum() if valid.any() else updated.sum() * 0
    kl = F.kl_div(updated.log_softmax(-1), native.detach().softmax(-1), reduction="batchmean")
    protect = relation_protection(updated, native, meta)
    loss = ce + SPEC["kl_weight"] * kl
    if arm == PRIMARY:
        loss = loss + SPEC["relation_weight"] * protect
    return dict(loss=loss, object_ce=ce, native_kl=kl, relation_protection=protect)


def to_device(record, device):
    return {k: v.to(device) for k, v in record.items() if torch.is_tensor(v)}


def fit(records, arm, epochs, callback=None, device="cuda"):
    if not records or epochs < 0:
        raise ValueError("Empty fit or invalid epochs")
    torch.manual_seed(SPEC["seed"])
    head = make_head(device)
    opt = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    # Compact image records keep memory bounded; no full SGG model or encoder is trained.
    data = [to_device(r, device) for r in records]
    generator = torch.Generator().manual_seed(SPEC["seed"])
    history = []
    if callback:
        callback(head, dict(epoch=0, loss=None))
    for epoch in range(1, epochs + 1):
        head.train()
        sums = {key: 0. for key in ["loss", "object_ce", "native_kl", "relation_protection"]}
        order = torch.randperm(len(data), generator=generator).tolist()
        for start in range(0, len(order), SPEC["accumulation_images"]):
            batch = order[start:start + SPEC["accumulation_images"]]
            opt.zero_grad(set_to_none=True)
            for index in batch:
                rec = data[index]
                values = terms(predict(head, rec["visual"], rec["native"]), rec["native"], rec["target"], rec, arm)
                if not all(torch.isfinite(value) for value in values.values()):
                    raise RuntimeError("Nonfinite objective")
                (values["loss"] / len(batch)).backward()
                for key in sums:
                    sums[key] += float(values[key].detach())
            norm = nn.utils.clip_grad_norm_(head.parameters(), SPEC["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            opt.step()
        row = dict(epoch=epoch, **{k: v / len(data) for k, v in sums.items()})
        history.append(row)
        if callback:
            callback(head, row)
    state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    return state, history


def gradient_audit(record, device="cpu"):
    rec = to_device(record, device)
    torch.manual_seed(SPEC["seed"])
    head = make_head(device)
    with torch.no_grad():
        head.weight.normal_(0, .02)
        # Lower emitted FG logits against BG to deliberately exercise protection.
        head.bias[1:] -= .5
    values = terms(predict(head, rec["visual"], rec["native"]), rec["native"], rec["target"], rec, PRIMARY)
    norms = {}
    for key in ["object_ce", "native_kl", "relation_protection"]:
        grads = torch.autograd.grad(values[key], tuple(head.parameters()), retain_graph=True)
        norms[key] = float(torch.sqrt(sum(g.square().sum() for g in grads)))
    active = torch.autograd.grad(values["loss"], tuple(head.parameters()), retain_graph=True)
    expected = torch.autograd.grad(values["object_ce"] + values["native_kl"] + values["relation_protection"], tuple(head.parameters()))
    error = max(float((a-b).abs().max()) for a, b in zip(active, expected))
    if not all(v > 0 and torch.isfinite(torch.tensor(v)) for v in norms.values()) or error > 1e-6:
        raise RuntimeError("A promised loss does not reach the trained head")
    return dict(loss_parameter_gradient_norms=norms, total_gradient_equivalence_error=error,
                audit_point="Deliberately nonzero synthetic head on real fit-image inputs; not an efficacy run",
                predicate_predictor_trained=False, nms_differentiable=False)
