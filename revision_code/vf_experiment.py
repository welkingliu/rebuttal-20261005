"""Cache, train and gate one decoupled object-readout repair without test tuning."""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, sha256
from native_runtime import infer
from repro_experiment import dataset, save_torch
from repro_protocol import immutable
from vb_native import targets, compare_predictions, OfficialMetrics
from vc_protocol import choose_ids, summarize_rows, paired_gate
from ve_experiment import setup as ve_setup, lookup, read, native_cache_check
from vf_native import RichReadout, ReadoutPatch, objective_terms
from vf_protocol import OUT, CACHE, SPEC, verify


def budget(args):
    if time.time() >= args.deadline:
        raise TimeoutError("V-F 12-hour budget reached; no automatic expansion")


def setup(task, out):
    model, cfg, transform, provenance, _, old_patch = ve_setup(task, out)
    old_patch.close()
    torch.manual_seed(SPEC["seed"])
    head = RichReadout().cuda()
    patch = ReadoutPatch(model, head)
    return model, cfg, transform, provenance, head, patch


def ckpt(task, mode):
    return ROOT / "checkpoints/VF" / task / (mode + "_seed17.pth")


def splits(task, cfg):
    previous = read(ROOT / "results/VE" / task / "splits.json")
    va, tr = lookup(dataset(cfg, "val")), lookup(dataset(cfg, "train"))
    excluded, hashes = set(), {}
    for old in SPEC["tasks"]:
        paths = [ROOT / "cache" / name / old / "protocol.json" for name in ["VB", "VC"]]
        paths.append(ROOT / "results/VE" / old / "splits.json")
        for path in paths:
            value = read(path); hashes[str(path)] = sha256(path)
            excluded.update(value["development"] + value["gate"])
    gate = choose_ids(set(va) - excluded, SPEC["gate_images"], "VF_adaptation_gate_20261001:")
    if len(gate) != 1000 or set(tr) & set(va) or set(gate) & excluded:
        raise RuntimeError("No complete fresh V-F adaptation gate")
    value = dict(train=previous["train"], development=previous["development"], gate=gate,
                 registration_sha256=verify(), previous_split_sha256=hashes)
    immutable(OUT / task / "splits.json", value)
    return value


def record(patch, gt, task, threshold):
    c = patch.capture
    value = dict(features=c["features"].cpu(), baseline=c["baseline"].cpu(), pairs=c["pairs"].cpu(),
                 relation_log_confidence=F.log_softmax(c["relation_logits"], -1)[:, 1:].max(-1)[0].cpu(),
                 targets=torch.from_numpy(targets(c, gt, task)), nms_threshold=threshold,
                 registration_sha256=verify())
    if task == "sgdet":
        value["boxes_per_cls"] = c["boxes_per_cls"].cpu()
    return value


def labels(logits, item, task):
    if task == "sgcls":
        return logits[:, 1:].argmax(1) + 1
    from pysgg.modeling.roi_heads.relation_head.utils_relation import obj_prediction_nms
    return obj_prediction_nms(item["boxes_per_cls"].to(logits.device), logits, item["nms_threshold"])


def report(out, detail, done, total, started, **extra):
    seconds = time.monotonic() - started
    row = dict(detail=detail, images=done, total=total, seconds=seconds,
               eta_seconds=seconds / max(done, 1) * (total - done), **extra)
    atomic_json(out / "progress.json", row); print(json.dumps(row), flush=True)


def prepare(args):
    out = OUT / args.task / "prepare"
    model, cfg, transform, provenance, head, patch = setup(args.task, out)
    selected = splits(args.task, cfg)
    threshold = model.roi_heads.relation.post_processor.later_nms_pred_thres
    smoke = []
    ds = dataset(cfg, "val"); mapping = lookup(ds)
    initial = {k: v.clone() for k, v in head.state_dict().items()}
    for iid in selected["development"][:3]:
        budget(args)
        i = mapping[iid]
        head.load_state_dict(initial); patch.enabled = False
        base, _ = infer(model, cfg, transform, ds, i)
        native_cache_check(base, args.task, iid)
        item = record(patch, ds.get_groundtruth(i, evaluation=True), args.task, threshold)
        if not torch.equal(labels(item["baseline"].cuda(), item, args.task).cpu(), base.get_field("pred_labels").cpu()):
            raise RuntimeError("Cached NMS does not match native postprocessing")
        patch.enabled = True
        zero, _ = infer(model, cfg, transform, ds, i)
        error = compare_predictions(base, zero)
        features, teacher, y = item["features"].cuda(), item["baseline"].cuda(), item["targets"].cuda()
        with torch.no_grad():
            head.output.weight.normal_(0, .005)
        updated = head(features, teacher)
        terms = objective_terms(updated, teacher, y, item["pairs"].cuda(), item["relation_log_confidence"].cuda())
        norms = {}
        for name, term in terms.items():
            gs = torch.autograd.grad(term, tuple(head.parameters()), retain_graph=True)
            norms[name] = sum(float(g.square().sum()) for g in gs) ** .5
            if not np.isfinite(norms[name]) or norms[name] <= 1e-12:
                raise RuntimeError("V-F missing active loss gradient: " + name)
        changed, _ = infer(model, cfg, transform, ds, i)
        score_change = float((changed.get_field("pred_scores") - base.get_field("pred_scores")).abs().max())
        if not patch.capture.get("route_checked") or score_change <= 1e-8:
            raise RuntimeError("V-F update did not change native object scores")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Native base was unfrozen")
        smoke.append(dict(image_id=iid, noop_error=error, gradient_norms=norms, emitted_score_delta=score_change))
    atomic_json(out / "smoke.json", dict(status="complete", checks=smoke, synthetic_head_discarded=True))
    head.load_state_dict(initial); patch.enabled = False
    started, done = time.monotonic(), 0
    for split, native_split in [("development", "val"), ("train", "train")]:
        ds = dataset(cfg, native_split); mapping = lookup(ds)
        for iid in selected[split]:
            budget(args)
            path = CACHE / args.task / split / (iid + ".pt")
            if path.exists():
                if torch.load(str(path), map_location="cpu")["registration_sha256"] != verify():
                    raise RuntimeError("Incompatible V-F cache")
            else:
                pred, _ = infer(model, cfg, transform, ds, mapping[iid])
                item = record(patch, ds.get_groundtruth(mapping[iid], evaluation=True), args.task, threshold)
                item["image_id"] = iid
                if not torch.equal(labels(item["baseline"].cuda(), item, args.task).cpu(), pred.get_field("pred_labels").cpu()):
                    raise RuntimeError("Cached/native identity mismatch")
                save_torch(path, item)
            done += 1
            if done % 100 == 0:
                report(out, "native feature cache: " + split, done, 5500, started)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", images=done, registration_sha256=verify(),
                 baseline_sha256=provenance["checkpoint_sha256"], split_sha256=sha256(OUT / args.task / "splits.json")))


def development(head, task, ids, args):
    correct = native_correct = count = 0
    nll = 0.
    with torch.no_grad():
        for iid in ids:
            budget(args)
            item = torch.load(str(CACHE / task / "development" / (iid + ".pt")), map_location="cpu")
            y, base = item["targets"].cuda(), item["baseline"].cuda()
            positive = y > 0
            logits = head(item["features"].cuda(), base)
            correct += int((labels(logits, item, task)[positive] == y[positive]).sum())
            native_correct += int((labels(base, item, task)[positive] == y[positive]).sum())
            count += int(positive.sum())
            nll += float(F.cross_entropy(logits[positive], y[positive], reduction="sum")) if positive.any() else 0.
    if count <= 0:
        raise RuntimeError("Empty development foreground support")
    return dict(post_nms_top1=correct / count, native_top1=native_correct / count,
                delta=(correct - native_correct) / count, foreground_nll=nll / count, objects=count)


def train(args):
    out = OUT / args.task / "training" / args.mode
    selected = read(OUT / args.task / "splits.json")
    if read(OUT / args.task / "prepare/summary.json")["status"] != "complete":
        raise RuntimeError("Cache/smoke incomplete")
    torch.manual_seed(SPEC["seed"])
    head = RichReadout().cuda()
    optim = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    history, best, stale = [], None, 0
    baseline = development(head, args.task, selected["development"], args)
    atomic_json(out / "initial_development.json", baseline)
    for epoch in range(1, SPEC["max_epochs"] + 1):
        head.train(); started = time.monotonic()
        order = np.random.default_rng(SPEC["seed"] + epoch).permutation(len(selected["train"]))
        loss_sums = {k: 0. for k in ["object_ce", "object_kl", "pair_score_kl"]}
        for start in range(0, len(order), SPEC["accumulate_images"]):
            budget(args); optim.zero_grad(set_to_none=True)
            batch = order[start:start + SPEC["accumulate_images"]]
            for index in batch:
                iid = selected["train"][int(index)]
                item = torch.load(str(CACHE / args.task / "train" / (iid + ".pt")), map_location="cpu")
                base = item["baseline"].cuda()
                logits = head(item["features"].cuda(), base)
                terms = objective_terms(logits, base, item["targets"].cuda(), item["pairs"].cuda(), item["relation_log_confidence"].cuda())
                loss = terms["object_ce"] if args.mode == "supervised" else sum(terms.values())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite V-F objective")
                (loss / len(batch)).backward()
                for key, value in terms.items():
                    loss_sums[key] += float(value.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SPEC["gradient_clip"]))
            if not np.isfinite(norm):
                raise RuntimeError("Nonfinite V-F gradient")
            optim.step()
            if (start + len(batch)) % 500 == 0:
                report(out, "cached " + args.mode + "; current epoch ETA only", start + len(batch), 5000, started, epoch=epoch)
        head.eval(); dev = development(head, args.task, selected["development"], args)
        history.append(dict(epoch=epoch, development=dev, mean_losses={k: v / len(order) for k, v in loss_sums.items()}))
        score = (dev["post_nms_top1"], -dev["foreground_nll"])
        if best is None or score > best:
            best, stale = score, 0
            save_torch(ckpt(args.task, args.mode), dict(head=head.state_dict(), epoch=epoch, development=dev,
                registration_sha256=verify(), split_sha256=sha256(OUT / args.task / "splits.json")))
        else:
            stale += 1
        save_torch(ckpt(args.task, args.mode).with_suffix(".last.pth"), dict(head=head.state_dict(), optimizer=optim.state_dict(),
                   epoch=epoch, history=history, registration_sha256=verify()))
        atomic_json(out / "history.json", history); print(json.dumps(history[-1]), flush=True)
        if epoch >= SPEC["min_epochs"] and stale >= SPEC["patience"]:
            break
    selected_head = torch.load(str(ckpt(args.task, args.mode)), map_location="cpu")
    atomic_json(out / "summary.json", dict(status="complete", history=history, selected_epoch=selected_head["epoch"],
        development=selected_head["development"], development_eligible=selected_head["development"]["delta"] >= .005,
        checkpoint_sha256=sha256(ckpt(args.task, args.mode)), registration_sha256=verify()))


def evaluate(args):
    out = OUT / args.task / "gate" / args.mode
    model, cfg, transform, provenance, head, patch = setup(args.task, out)
    selected = read(OUT / args.task / "splits.json")
    if args.mode != "native":
        payload = torch.load(str(ckpt(args.task, args.mode)), map_location="cpu")
        if payload["registration_sha256"] != verify() or payload["split_sha256"] != sha256(OUT / args.task / "splits.json"):
            raise RuntimeError("V-F gate checkpoint provenance mismatch")
        head.load_state_dict(payload["head"])
    head.eval(); patch.enabled = args.mode != "native"
    ds = dataset(cfg, "val"); mapping = lookup(ds)
    metric = OfficialMetrics(args.task); rows = []; started = time.monotonic()
    from export_pysgg_vg_task import convert_prediction
    for number, iid in enumerate(selected["gate"], 1):
        budget(args)
        pred, _ = infer(model, cfg, transform, ds, mapping[iid])
        if args.mode == "native":
            native_cache_check(pred, args.task, iid)
        gt = ds.get_groundtruth(mapping[iid], evaluation=True)
        row = metric.row(iid, pred, gt, patch.capture, targets(patch.capture, gt, args.task))
        row["protocol_sha256"] = verify()
        atomic_json(out / "images" / (iid + ".json"), row)
        from common import output_path
        path = output_path(out / "predictions" / (iid + ".npz"))
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **convert_prediction(pred.to("cpu"))); temp.replace(path)
        rows.append(row)
        if number % 50 == 0:
            report(out, "fresh adaptation gate: " + args.mode, number, 1000, started)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", registration_sha256=verify(), **summarize_rows(rows)))


def decision(args):
    selected = read(OUT / args.task / "splits.json")
    def load(mode):
        return [read(OUT / args.task / "gate" / mode / "images" / (iid + ".json")) for iid in selected["gate"]]
    base = load("native")
    results = {mode: paired_gate(base, load(mode)) for mode in SPEC["modes"]}
    atomic_json(OUT / args.task / "decision.json", dict(status="complete", accepted=results["score_protected"]["accepted"],
        primary="score_protected", results=results, registration_sha256=verify(),
        exploratory=True, test_evaluated=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "train", "evaluate", "decision"], required=True)
    p.add_argument("--task", choices=SPEC["tasks"], required=True)
    p.add_argument("--mode", choices=["native"] + SPEC["modes"], default="native")
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args(); ensure_storage(); verify(); budget(args); torch.set_num_threads(2)
    {"prepare": prepare, "train": train, "evaluate": evaluate, "decision": decision}[args.stage](args)


if __name__ == "__main__":
    main()
