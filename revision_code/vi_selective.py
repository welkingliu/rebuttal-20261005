"""Cross-fitted auxiliary probe and one training-only repair/damage selector."""
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, sha256
from independent_identity_expert import crop_tensor, save, summary, REPO, WEIGHT
from vi_protocol import OUT, CACHE, CKPT, SPEC, MANIFEST, read, verify


def budget(deadline):
    if time.time() >= deadline:
        raise TimeoutError("V-I budget exhausted")


def progress(folder, detail, n, total, start, **extra):
    elapsed = time.monotonic() - start
    row = dict(detail=detail, images=n, total=total, seconds=elapsed,
               eta_seconds=elapsed / max(n, 1) * (total - n), **extra)
    atomic_json(folder / "progress.json", row)
    print(__import__("json").dumps(row), flush=True)


def candidate(logits, expert):
    native = logits.softmax(-1)
    mass = native[:, 1:].sum(-1, keepdim=True)
    p = native[:, 1:] / mass.clamp_min(1e-12)
    q = expert
    changed = torch.cat([native[:, :1], mass * ((1 - SPEC["alpha"]) * p + SPEC["alpha"] * q)], -1)
    return native, p, q, changed


def router_features(p, q):
    a, b = p.topk(2, dim=-1).values, q.topk(2, dim=-1).values
    pl, ql = p.argmax(-1), q.argmax(-1)
    row = torch.arange(len(p), device=p.device)
    entropy = lambda x: -(x * x.clamp_min(1e-12).log()).sum(-1) / np.log(x.shape[-1])
    return torch.stack([a[:, 0], b[:, 0], a[:, 0] - a[:, 1], b[:, 0] - b[:, 1],
                        entropy(p), entropy(q), p[row, ql], q[row, pl],
                        (p - q).abs().sum(-1), (p * q).sum(-1)], -1)


def informative(p, changed, y):
    old = p.argmax(-1) + 1 == y
    new = changed[:, 1:].argmax(-1) + 1 == y
    return old ^ new, new.float()


def fuse(logits, expert, router):
    native, p, q, changed = candidate(logits, expert)
    x = (router_features(p, q) - router["mean"]) / router["scale"]
    confidence = (x @ router["weight"].T + router["bias"]).flatten().sigmoid()
    mask = ((p.argmax(-1) != q.argmax(-1)) &
            (p.argmax(-1) != changed[:, 1:].argmax(-1)) &
            (confidence >= SPEC["repair_probability_min"]))
    return torch.where(mask[:, None], changed, native), mask, confidence


def load_cached(split, deadline):
    ids = read(OUT / "splits.json")[split]
    xs, ys, bs, groups = [], [], [], []
    vh_hash = sha256(ROOT / "results/VH_independent_identity/protocol.json")
    vf_hash = read(MANIFEST)["vf_registration"]
    start = time.monotonic()
    for n, iid in enumerate(ids, 1):
        budget(deadline)
        f = torch.load(str(ROOT / "cache/VH_independent_identity" / split / (iid + ".pt")), map_location="cpu", weights_only=True)
        b = torch.load(str(ROOT / "cache/VF/sgcls" / split / (iid + ".pt")), map_location="cpu", weights_only=True)
        if (f["protocol_sha256"] != vh_hash or b["registration_sha256"] != vf_hash or
                not torch.equal(f["labels"], b["targets"]) or f["image_id"] != iid or b["image_id"] != iid):
            raise RuntimeError("Training cache alignment/provenance failed: " + iid)
        xs.append(f["features"]); ys.append(f["labels"]); bs.append(b["baseline"])
        groups.extend([iid] * len(f["labels"]))
        if n % 250 == 0:
            progress(OUT / "development", "load aligned " + split + " caches", n, len(ids), start)
    x, y, b = torch.cat(xs).cuda(), torch.cat(ys).cuda(), torch.cat(bs).cuda()
    if x.shape != (len(y), 768) or b.shape != (len(y), 151) or not ((y >= 1) & (y <= 150)).all():
        raise RuntimeError("Wrong VI feature/vocabulary shape")
    return x, y, b, groups


def fit_probe(x, y, name, deadline):
    torch.manual_seed(SPEC["seed"])
    head = nn.Linear(768, 150).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=SPEC["probe_lr"], weight_decay=SPEC["probe_weight_decay"])
    history, start = [], time.monotonic()
    for epoch in range(1, SPEC["probe_epochs"] + 1):
        budget(deadline)
        order = torch.randperm(len(y), generator=torch.Generator(device="cuda").manual_seed(17 + epoch), device="cuda")
        total = 0.
        for ix in order.split(SPEC["probe_batch"]):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[ix]), y[ix] - 1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite OOF probe loss")
            loss.backward(); opt.step(); total += float(loss.detach()) * len(ix)
        history.append(dict(epoch=epoch, train_nll=total / len(y)))
        if epoch % 10 == 0:
            progress(OUT / "development", "train auxiliary " + name, epoch, SPEC["probe_epochs"], start)
    head.eval()
    save(CKPT / (name + ".pth"), dict(head={k: v.cpu() for k, v in head.state_dict().items()},
         protocol_sha256=sha256(MANIFEST), history=history))
    return head


def image_rows(ids, groups, y, logits, probabilities, masks):
    result, cursor = {}, 0
    for iid in ids:
        n = 0
        while cursor + n < len(groups) and groups[cursor + n] == iid:
            n += 1
        if not n:
            raise RuntimeError("Noncontiguous or missing image records")
        ix = slice(cursor, cursor + n); cursor += n
        old_label = logits[ix, 1:].argmax(-1) + 1
        old = old_label == y[ix]
        for name, prob in probabilities.items():
            label = prob[ix, 1:].argmax(-1) + 1; correct = label == y[ix]
            result.setdefault(name, []).append(dict(image_id=iid, objects=n, correct=int(correct.sum()),
                baseline_correct=int(old.sum()), repaired=int((~old & correct).sum()), damaged=int((old & ~correct).sum()),
                eligible=int(masks[name][ix].sum()), label_flips=int((label != old_label).sum()),
                class_count=torch.bincount(y[ix], minlength=151)[1:].cpu().tolist(),
                class_correct=torch.bincount(y[ix][correct], minlength=151)[1:].cpu().tolist()))
    if cursor != len(groups):
        raise RuntimeError("Unconsumed image rows")
    return result


def develop(deadline):
    x, y, logits, groups = load_cached("train", deadline)
    folds = read(MANIFEST)["folds"]
    assigned = torch.tensor([folds[i] for i in groups], device="cuda")
    oof = torch.empty((len(y), 150), device="cuda"); seen = torch.zeros(len(y), dtype=torch.bool, device="cuda")
    fold_records = []
    for fold in range(SPEC["folds"]):
        hold = assigned == fold; fit = ~hold
        assert not bool((seen & hold).any())
        head = fit_probe(x[fit], y[fit], "fold%d" % fold, deadline)
        with torch.no_grad():
            oof[hold] = head(x[hold]).softmax(-1)
        seen |= hold
        fit_ids = sorted({i for i in groups if folds[i] != fold})
        hold_ids = sorted({i for i in groups if folds[i] == fold})
        if set(fit_ids) & set(hold_ids):
            raise RuntimeError("OOF image leakage")
        fold_records.append(dict(fold=fold, train_ids=fit_ids, prediction_ids=hold_ids,
                                 train_objects=int(fit.sum()), prediction_objects=int(hold.sum())))
    if not bool(seen.all()):
        raise RuntimeError("Incomplete OOF predictions")
    save(CACHE / "oof.pt", dict(expert=oof.cpu(), labels=y.cpu(), image_ids=groups, protocol_sha256=sha256(MANIFEST)))
    atomic_json(OUT / "crossfit.json", dict(folds=fold_records, protocol_sha256=sha256(MANIFEST),
        native_model_not_cross_fitted=True, auxiliary_predictions_out_of_fold=True))
    native, p, q, changed = candidate(logits, oof)
    useful, target = informative(p, changed, y)
    features = router_features(p, q)[useful]
    target = target[useful]
    if len(target) < 100 or min(int(target.sum()), int((1 - target).sum())) < 20:
        raise RuntimeError("Insufficient OOF repair/damage training support")
    mean, scale = features.mean(0), features.std(0).clamp_min(.001)
    normalized = (features - mean) / scale
    torch.manual_seed(17)
    selector = nn.Linear(normalized.shape[-1], 1).cuda()
    nn.init.zeros_(selector.weight); nn.init.zeros_(selector.bias)
    optim = torch.optim.Adam(selector.parameters(), lr=SPEC["router_lr"])
    history = []
    for epoch in range(1, SPEC["router_epochs"] + 1):
        budget(deadline); optim.zero_grad(set_to_none=True)
        ce = F.binary_cross_entropy_with_logits(selector(normalized).flatten(), target)
        loss = ce + SPEC["router_l2"] * selector.weight.square().sum() / 2
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite selector loss")
        loss.backward(); optim.step()
        history.append(dict(epoch=epoch, objective=float(loss.detach()), bce=float(ce.detach())))
    router = dict(mean=mean, scale=scale, weight=selector.weight.detach(), bias=selector.bias.detach())
    head = fit_probe(x, y, "full_train", deadline)
    payload = dict(head={k: v.cpu() for k, v in head.state_dict().items()},
                   router={k: v.cpu() for k, v in router.items()}, protocol_sha256=sha256(MANIFEST))
    save(CKPT / "selected.pth", payload)
    atomic_json(OUT / "selector_training.json", dict(objects=len(y), informative_objects=len(target),
        repairs=int(target.sum()), damages=int((1 - target).sum()), history=history,
        labels_from="native training split only; OOF auxiliary predictions", checkpoint_sha256=sha256(CKPT / "selected.pth")))
    dx, dy, base, group = load_cached("development", deadline)
    with torch.no_grad():
        expert = head(dx).softmax(-1)
        prob, mask, _ = fuse(base, expert, router)
        original, _, _, unselected = candidate(base, expert)
        exp = torch.cat([original[:, :1], original[:, 1:].sum(-1, keepdim=True) * expert], -1)
        conditions = dict(native=original, expert_only=exp, fixed_fusion_control=unselected, selective=prob)
        masks = {name: torch.ones(len(dy), dtype=torch.bool, device="cuda") for name in conditions}
        masks["native"][:] = False; masks["selective"] = mask
        rows = image_rows(read(OUT / "splits.json")["development"], group, dy, base, conditions, masks)
    results = {name: summary(value) for name, value in rows.items()}
    r = results["selective"]; req = SPEC["requirements"]
    checks = dict(identity_gain=r["delta"] >= req["identity_gain_min"],
        correction_precision=r["correction_precision"] is not None and r["correction_precision"] >= req["correction_precision_min"],
        macro_accuracy=r["macro_accuracy"] >= results["native"]["macro_accuracy"])
    atomic_json(OUT / "development/images.json", rows)
    atomic_json(OUT / "development/summary.json", dict(status="complete", development_eligible=all(checks.values()),
        checks=checks, results=results, protocol_sha256=verify(), checkpoint_sha256=sha256(CKPT / "selected.pth"),
        fresh_gate_evaluated=False, test_evaluated=False))
    print(__import__("json").dumps(dict(checks=checks, results=results)), flush=True)


def gate_features(task, deadline):
    from PIL import Image
    folder = OUT / task / "expert"
    export = read(OUT / task / "export/summary.json")
    if export["protocol_sha256"] != verify():
        raise RuntimeError("Gate export mismatch")
    vh = read(ROOT / "results/VH_independent_identity/protocol.json")
    if sha256(WEIGHT) != vh["encoder_sha256"] or any(sha256(REPO / name) != h for name, h in read(MANIFEST)["encoder_sources"].items()):
        raise RuntimeError("Frozen DINO source/weights changed")
    encoder = torch.hub.load(str(REPO), "dinov2_vitb14", source="local", pretrained=False)
    encoder.load_state_dict(torch.load(str(WEIGHT), map_location="cpu", weights_only=True), strict=True)
    encoder.cuda().eval(); encoder.requires_grad_(False)
    selected = torch.load(str(CKPT / "selected.pth"), map_location="cpu", weights_only=True)
    if selected["protocol_sha256"] != verify():
        raise RuntimeError("Selected head mismatch")
    head = nn.Linear(768, 150).cuda(); head.load_state_dict(selected["head"]); head.eval()
    router = {k: v.cuda() for k, v in selected["router"].items()}
    ids = read(OUT / "splits.json")["gate"]; start = time.monotonic()
    for n, iid in enumerate(ids, 1):
        budget(deadline)
        item = torch.load(str(CACHE / task / "export" / (iid + ".pt")), map_location="cpu", weights_only=True)
        if item["protocol_sha256"] != verify():
            raise RuntimeError("Native crop provenance mismatch")
        with Image.open(item["image"]) as source:
            image = source.convert("RGB")
        sx, sy = image.width / item["size"][0], image.height / item["size"][1]
        boxes = item["boxes"] * torch.tensor([sx, sy, sx, sy])
        with torch.no_grad():
            features = []
            for chunk in boxes.split(8):
                crops = torch.stack([crop_tensor(image, box.tolist()) for box in chunk]).cuda()
                features.append(F.normalize(encoder(crops).float(), dim=-1))
            expert = head(torch.cat(features)).softmax(-1)
            updated, mask, confidence = fuse(item["baseline"].cuda(), expert, router)
            save(CACHE / task / "updates" / (iid + ".pt"), dict(logits=updated.clamp_min(1e-30).log().cpu(),
                native_logits=item["baseline"], boxes=item["boxes"], size=item["size"],
                selected=mask.cpu(), repair_probability=confidence.cpu(), protocol_sha256=sha256(MANIFEST),
                checkpoint_sha256=sha256(CKPT / "selected.pth")))
        if n % 25 == 0:
            progress(folder, task + " native-proposal crop evidence", n, len(ids), start)
    atomic_json(folder / "summary.json", dict(status="complete", images=len(ids), protocol_sha256=verify(),
        checkpoint_sha256=sha256(CKPT / "selected.pth")))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["develop", "gate_features"])
    p.add_argument("--task", choices=["sgcls", "sgdet"], default="sgcls")
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args(); ensure_storage(); verify(); torch.set_num_threads(2)
    if args.stage == "develop":
        develop(args.deadline)
    else:
        gate_features(args.task, args.deadline)


if __name__ == "__main__":
    main()
