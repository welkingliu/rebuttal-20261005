"""One locked Mac-only visual-compatibility pilot; never claims a native gate pass."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPEC = dict(version="vj_visual_compatibility_v1", seed=17, device="cpu", cpu_threads=4,
    train_images=5000, development_images=500, folds=5, classes=150,
    feature="frozen normalized DINOv2-B GT-box crop CLS768; reused VH cache",
    expert="VI training-only5-fold OOF and fixed40-epoch full-train probe",
    new_evidence=["native class cosine", "proposed class cosine", "cosine difference", "class prototype cosine"],
    prototype="normalized class mean, exclude whole held-out image fold; full train only at inference",
    minimum_prototype_objects=5, alpha=.5, repair_probability_min=.6,
    selector="standardized linear logistic; nonzero utility repair versus damage; no reweighting",
    epochs=200, learning_rate=.01, l2=.01,
    control="same confidence-only selector, same support, data, loss, optimizer, threshold; not eligible for selection",
    screening=dict(identity_gain_min=.005, repair_precision_min=.6, macro_accuracy_not_lower=True),
    gate=dict(object_gain_min=.005, object_bootstrap_lower_min=0.,
        relation_noninferiority_margin=.005, bootstrap_samples=2000, lower_quantile=.05,
        metrics=["R@50", "mR@50"], minimum_positive_objects=1000, images=1000,
        requires_both_tasks=True),
    development="old500 images reused for one screen only, never independent confirmation or epoch selection",
    confirmation="reserved VF/VI1000, adapter-held-out only; native validation audits already saw them",
    native_gate="required later; actual SGCls/SGDet scores/NMS/boxes/ranking, same frozen repair on real proposals",
    stop="screen reject stops; pass awaits native two-task gate; no threshold search, extra seeds or test",
    wall_hours=2)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def seal(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("Immutable protocol changed")
    if not path.exists():
        write(path, value)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, str(tmp)); tmp.replace(path)


def progress(out, stage, done, total, start, **extra):
    elapsed = time.monotonic() - start
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
               stage=stage, completed=done, total=total, seconds=elapsed, **extra)
    write(out / "progress.json", row); print(json.dumps(row), flush=True)
    if elapsed >= SPEC["wall_hours"] * 3600:
        raise TimeoutError("Mac pilot budget reached")


def probabilities(logits, expert):
    native = logits.softmax(-1)
    mass = native[:, 1:].sum(-1, keepdim=True)
    p = native[:, 1:] / mass.clamp_min(1e-12)
    candidate = torch.cat([native[:, :1], mass * ((1 - SPEC["alpha"]) * p + SPEC["alpha"] * expert)], -1)
    return native, p, candidate


def confidence_features(p, q):
    a, b = p.topk(2, dim=-1).values, q.topk(2, dim=-1).values
    rows = torch.arange(len(p)); pl, ql = p.argmax(-1), q.argmax(-1)
    def entropy(z):
        return -(z * z.clamp_min(1e-12).log()).sum(-1) / np.log(z.shape[-1])
    return torch.stack([a[:, 0], b[:, 0], a[:, 0] - a[:, 1], b[:, 0] - b[:, 1],
        entropy(p), entropy(q), p[rows, ql], q[rows, pl],
        (p - q).abs().sum(-1), (p * q).sum(-1)], -1)


def prototypes(x, y):
    sums = torch.zeros((SPEC["classes"], x.shape[1]), dtype=x.dtype)
    sums.index_add_(0, y - 1, x)
    count = torch.bincount(y - 1, minlength=SPEC["classes"])
    return F.normalize(sums / count[:, None].clamp_min(1), dim=-1), count


def visual_features(x, p, candidate, centers, count):
    a, b = p.argmax(-1), candidate[:, 1:].argmax(-1)
    ca, cb = centers[a], centers[b]
    sa, sb = (x * ca).sum(-1), (x * cb).sum(-1)
    values = torch.stack([sa, sb, sb - sa, (ca * cb).sum(-1)], -1)
    enough = (count[a] >= SPEC["minimum_prototype_objects"]) & (count[b] >= SPEC["minimum_prototype_objects"])
    return values, enough


def group_folds(data, mapping):
    ids, offsets = data["image_ids"], data["offsets"]
    if len(ids) != len(set(ids)) or len(offsets) != len(ids) + 1 or offsets[0] != 0:
        raise ValueError("Invalid image grouping")
    if offsets[-1] != len(data["labels"]) or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Missing or unordered object groups")
    return torch.tensor([mapping[iid] for iid, a, b in zip(ids, offsets, offsets[1:]) for _ in range(b - a)])


def validate_bundle(bundle):
    train, dev = bundle["data"]["train"], bundle["data"]["development"]
    gate = bundle["provenance"]["reserved_gate_ids"]
    if (len(train["image_ids"]) != 5000 or len(dev["image_ids"]) != 500
            or len(set(gate)) != 1000 or set(train["image_ids"]) & set(dev["image_ids"] + gate)
            or set(dev["image_ids"]) & set(gate)):
        raise ValueError("Split isolation/size failed")
    for data in [train, dev]:
        x, y, b = data["features"], data["labels"], data["baseline"]
        if (x.shape != (len(y), 768) or b.shape != (len(y), 151)
                or not bool(((y >= 1) & (y <= 150)).all())
                or not torch.isfinite(x).all() or not torch.isfinite(b).all()
                or not torch.allclose(x.norm(dim=-1), torch.ones(len(y)), atol=1e-5)):
            raise ValueError("Wrong feature/logit/vocabulary contract")
    fold = group_folds(train, bundle["folds"])
    if sorted(fold.unique().tolist()) != list(range(5)):
        raise ValueError("Incomplete cross-fitting folds")
    oof = bundle["oof"]
    if (oof.shape != (len(train["labels"]), 150) or not torch.isfinite(oof).all()
            or (oof < 0).any() or not torch.allclose(oof.sum(-1), torch.ones(len(oof)), atol=1e-5)):
        raise ValueError("Invalid auxiliary probability cache")
    checks = []
    for f, state in enumerate(bundle["fold_heads"]):
        ix = fold == f
        actual = F.linear(train["features"][ix], state["weight"], state["bias"]).softmax(-1)
        error = float((actual - oof[ix]).abs().max())
        if not torch.allclose(actual, oof[ix], atol=2e-5, rtol=2e-5):
            raise ValueError("OOF probabilities differ from registered excluded-fold probe")
        checks.append(dict(fold=f, objects=int(ix.sum()), max_error=error))
    return fold, checks


def fit_router(x, y, out, name, start):
    mean, scale = x.mean(0), x.std(0).clamp_min(.001)
    x = (x - mean) / scale
    torch.manual_seed(SPEC["seed"])
    layer = nn.Linear(x.shape[1], 1)
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
    opt = torch.optim.Adam(layer.parameters(), lr=SPEC["learning_rate"])
    history = []
    for epoch in range(1, SPEC["epochs"] + 1):
        opt.zero_grad(set_to_none=True)
        ce = F.binary_cross_entropy_with_logits(layer(x).flatten(), y)
        loss = ce + SPEC["l2"] * layer.weight.square().sum() / 2
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite selector loss")
        loss.backward()
        grad = sum(float(p.grad.square().sum()) for p in layer.parameters()) ** .5
        opt.step()
        history.append(dict(epoch=epoch, bce=float(ce.detach()), objective=float(loss.detach()), gradient_norm=grad))
        if epoch % 25 == 0:
            progress(out, "train_" + name, epoch, SPEC["epochs"], start, loss=float(loss.detach()))
    write(out / (name + "_training.json"), dict(objects=len(y), repairs=int(y.sum()),
        damages=int((1 - y).sum()), features=x.shape[1], history=history))
    return dict(mean=mean, scale=scale, weight=layer.weight.detach(), bias=layer.bias.detach())


def route(logits, expert, features, router, supported):
    native, p, candidate = probabilities(logits, expert)
    confidence = F.linear((features - router["mean"]) / router["scale"], router["weight"], router["bias"]).flatten().sigmoid()
    selected = (supported & (p.argmax(-1) != expert.argmax(-1))
                & (p.argmax(-1) != candidate[:, 1:].argmax(-1))
                & (confidence >= SPEC["repair_probability_min"]))
    return torch.where(selected[:, None], candidate, native), selected, confidence


def records(data, probabilities_, selected):
    result = []
    for iid, a, b in zip(data["image_ids"], data["offsets"], data["offsets"][1:]):
        y = data["labels"][a:b]
        old_label = data["baseline"][a:b, 1:].argmax(-1) + 1
        label = probabilities_[a:b, 1:].argmax(-1) + 1
        old, new = old_label == y, label == y
        result.append(dict(image_id=iid, objects=b-a, baseline_correct=int(old.sum()), correct=int(new.sum()),
            repairs=int((~old & new).sum()), damages=int((old & ~new).sum()), selected=int(selected[a:b].sum()),
            flips=int((old_label != label).sum()),
            class_count=torch.bincount(y - 1, minlength=150).tolist(),
            class_correct=torch.bincount(y[new] - 1, minlength=150).tolist()))
    return result


def aggregate(rows):
    totals = {k: sum(x[k] for x in rows) for k in ["objects", "baseline_correct", "correct", "repairs", "damages", "selected", "flips"]}
    counts = np.sum([x["class_count"] for x in rows], axis=0)
    correct = np.sum([x["class_correct"] for x in rows], axis=0)
    supported = counts > 0
    n = totals["objects"]
    totals.update(top1=totals["correct"] / n, native_top1=totals["baseline_correct"] / n,
        delta=(totals["correct"] - totals["baseline_correct"]) / n,
        macro_accuracy=float((correct[supported] / counts[supported]).mean()),
        supported_classes=int(supported.sum()),
        repair_precision=totals["repairs"] / (totals["repairs"] + totals["damages"]) if totals["repairs"] + totals["damages"] else None)
    rng = np.random.default_rng(17029)
    values = np.array([[r["correct"] - r["baseline_correct"], r["objects"]] for r in rows])
    draws = []
    for _ in range(2000):
        sample = values[rng.integers(0, len(rows), len(rows))].sum(0)
        draws.append(sample[0] / sample[1])
    totals["descriptive_paired_image_ci95"] = np.quantile(draws, [.025, .975]).tolist()
    return totals


def screen_checks(result, native):
    req = SPEC["screening"]
    return dict(identity_gain=result["delta"] >= req["identity_gain_min"],
        repair_precision=result["repair_precision"] is not None and result["repair_precision"] >= req["repair_precision_min"],
        macro_accuracy=result["macro_accuracy"] >= native["macro_accuracy"])


def run(bundle_path, out):
    start = time.monotonic()
    sidecar = json.loads(bundle_path.with_suffix(".json").read_text())
    if digest(bundle_path) != sidecar["bundle_sha256"]:
        raise RuntimeError("Transferred bundle hash mismatch")
    protocol = dict(spec=SPEC, bundle_sha256=sidecar["bundle_sha256"],
        code_sha256=digest(Path(__file__)), plan_sha256=digest(Path(__file__).with_name("VJ_MAC_PLAN.md")),
        provenance=sidecar, torch_version=str(torch.__version__))
    seal(out / "protocol.json", protocol)
    if (out / "summary.json").exists() or (out / "selected.pth").exists():
        raise RuntimeError("Existing frozen/result files; do not silently rerun or overwrite")
    progress(out, "verify_inputs", 0, 5, start)
    bundle = torch.load(str(bundle_path), map_location="cpu", weights_only=True)
    folds, audit = validate_bundle(bundle)
    write(out / "input_audit.json", dict(folds=audit, image_disjoint_prototypes=True,
        gate_labels_or_features_loaded=False, all_parameters_cpu=True,
        source_order_hashes={k: d["ordered_input_digest"] for k, d in bundle["data"].items()}))
    train = bundle["data"]["train"]
    x, y, b, q = train["features"], train["labels"], train["baseline"], bundle["oof"]
    _, p, candidate = probabilities(b, q)
    basic = confidence_features(p, q)
    visual = torch.empty((len(y), 4)); supported = torch.zeros(len(y), dtype=torch.bool)
    for f in range(5):
        hold = folds == f
        centers, count = prototypes(x[~hold], y[~hold])
        visual[hold], supported[hold] = visual_features(x[hold], p[hold], candidate[hold], centers, count)
    informative = ((p.argmax(-1) + 1 == y) ^ (candidate[:, 1:].argmax(-1) + 1 == y)) & supported
    target = (candidate[:, 1:].argmax(-1) + 1 == y)[informative].float()
    if len(target) < 100 or min(int(target.sum()), int((1 - target).sum())) < 20:
        raise RuntimeError("Insufficient repair/damage training support")
    full = torch.cat([basic, visual], -1)
    router = fit_router(full[informative], target, out, "visual_verifier", start)
    control = fit_router(basic[informative], target, out, "confidence_control", start)
    centers, count = prototypes(x, y)
    frozen = dict(router=router, confidence_control=control, centers=centers, counts=count,
        expert_head=bundle["selected"]["head"], protocol_sha256=digest(out / "protocol.json"),
        trained_on="native train5000 only, OOF expert/prototypes; fixed epochs")
    save(out / "selected.pth", frozen)
    selected_hash = digest(out / "selected.pth")
    write(out / "frozen_before_development.json", dict(checkpoint_sha256=selected_hash,
        timestamp=datetime.now(timezone.utc).isoformat(), development_selection=False))
    progress(out, "development_screen_only", 0, 500, start, checkpoint_sha256=selected_hash)
    dev = bundle["data"]["development"]
    with torch.inference_mode():
        q = F.linear(dev["features"], frozen["expert_head"]["weight"], frozen["expert_head"]["bias"]).softmax(-1)
        native, p, candidate = probabilities(dev["baseline"], q)
        basic = confidence_features(p, q)
        vis, eligible = visual_features(dev["features"], p, candidate, centers, count)
        primary, mask, conf = route(dev["baseline"], q, torch.cat([basic, vis], -1), router, eligible)
        confidence, cmask, _ = route(dev["baseline"], q, basic, control, eligible)
        old, omask, old_conf = route(dev["baseline"], q, basic, bundle["selected"]["router"], torch.ones_like(eligible))
        conditions = dict(native=(native, torch.zeros_like(mask)), visual_verifier=(primary, mask),
            confidence_control=(confidence, cmask), saved_vi_reference=(old, omask),
            fixed_fusion_control=(candidate, torch.ones_like(mask)))
        rows = {name: records(dev, prob, m) for name, (prob, m) in conditions.items()}
        results = {name: aggregate(value) for name, value in rows.items()}
    if digest(out / "selected.pth") != selected_hash:
        raise RuntimeError("Frozen weights changed during development")
    write(out / "development_images.json", rows)
    checks = screen_checks(results["visual_verifier"], results["native"])
    passed = all(checks.values())
    conclusion = dict(status="awaiting_native_confirmation" if passed else "stopped_screen_rejected",
        development_eligible=passed, joint_gate_accepted=None if passed else False, checks=checks,
        results=results, highest_selection_probability=float(conf.max()),
        saved_vi_highest_probability=float(old_conf.max()),
        checkpoint_sha256=selected_hash, protocol_sha256=digest(out / "protocol.json"),
        fresh_confirmation_evaluated=False, SGCls_relation_gate_evaluated=False, SGDet_gate_evaluated=False,
        test_evaluated=False, extra_seeds_run=False, cuda_used=False,
        mechanism="Confidence-only control is descriptive; no switching primary candidate",
        next_step="Only a passed native SGCls AND SGDet joint gate permits seeds/test" if passed else "Stop; retain failure; no retuning or fresh-gate consumption",
        seconds=time.monotonic() - start)
    write(out / "summary.json", conclusion)
    progress(out, conclusion["status"], 500, 500, start,
             checks=checks, identity_delta=results["visual_verifier"]["delta"],
             native_gate_passed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(SPEC["cpu_threads"])
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.bundle.resolve(), out)
        except Exception as exc:
            write(out / "failure.json", dict(status="failed", error=type(exc).__name__ + ": " + str(exc)))
            raise


if __name__ == "__main__":
    main()
