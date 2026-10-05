"""V-E native smoke, bounded pilot training, and unchanged full-system gates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from native_runtime import image_id, infer
from repro_experiment import build, dataset, save_torch
from repro_protocol import immutable
from reproduction_pair_audit import ChunkUnion
from sgdet_identity import capture
from identity_intervention import fingerprint, tensor_leaves
from vb_native import compare_predictions, OfficialMetrics, targets
from vc_protocol import choose_ids, summarize_rows, paired_gate
from ve_protocol import OUT, SPEC, baseline, verify
from ve_native import SharedContextAdapter, ContextAdapterPatch, objective_terms, objective


def read(path):
    return json.loads(path.read_text())


def split_path(task):
    return OUT / task / "splits.json"


def checkpoint(task, mode, seed):
    return ROOT / "checkpoints/VE" / task / (mode + "_seed%d.pth" % seed)


def setup(task, out, seed=17):
    torch.manual_seed(seed)
    weights = ROOT / "results/R9_reproduction/sgdet/training/model_final.pth" if task == "sgdet" else None
    model, cfg, transform, provenance = build(task, out, weights=weights)
    if provenance["checkpoint_sha256"] != read(baseline(task) / "model_provenance.json")["checkpoint_sha256"]:
        raise RuntimeError("V-E base checkpoint changed")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if task == "sgdet":
        model.roi_heads.relation.samp_processor.max_proposal_pairs = 1000000
        model._ve_chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
    torch.manual_seed(seed)
    head = SharedContextAdapter().cuda()
    patch = ContextAdapterPatch(model, head)
    return model, cfg, transform, provenance, head, patch


def lookup(ds):
    return {image_id(ds, i): i for i in range(len(ds))}


def prepare_splits(task, cfg, provenance):
    tr, va = lookup(dataset(cfg, "train")), lookup(dataset(cfg, "val"))
    previous = read(ROOT / "cache/VB" / task / "protocol.json")
    train, dev = previous["train"], previous["development"]
    excluded, inputs = set(), {}
    for name in ["VB", "VC"]:
        for other in SPEC["tasks"]:
            path = ROOT / "cache" / name / other / "protocol.json"
            data = read(path)
            excluded.update(data["development"] + data["gate"])
            inputs[str(path)] = sha256(path)
    gate = choose_ids(set(va) - excluded, SPEC["gate_images"], "VE_adaptation_gate_20261001:")
    if len(train) != 5000 or len(dev) != 500 or len(gate) != 1000:
        raise RuntimeError("Incomplete V-E split")
    if not set(train) <= set(tr) or not set(dev + gate) <= set(va):
        raise RuntimeError("V-E split is not native train/validation")
    if set(tr) & set(va) or set(gate) & excluded or set(dev) & set(gate):
        raise RuntimeError("V-E adaptation split overlap")
    value = dict(registration_sha256=verify(), task=task, train=train, development=dev, gate=gate,
                 previous_split_sha256=inputs, checkpoint_sha256=provenance["checkpoint_sha256"],
                 note=SPEC["gate_scope"])
    immutable(split_path(task), value)
    return value


def native_cache_check(pred, task, iid):
    folder = (ROOT / "results/R14_pair_cap/sgdet/predictions/reference_all_pairs" if task == "sgdet"
              else ROOT / "results/R9_reproduction/sgcls/eval_corrected/val/predictions")
    path = folder / (iid + ".npz")
    from export_pysgg_vg_task import convert_prediction
    values = convert_prediction(pred.to("cpu"))
    errors = {}
    with np.load(path, allow_pickle=False) as z:
        for key, value in values.items():
            if value.shape != z[key].shape:
                raise RuntimeError("V-E native baseline shape mismatch")
            ok = np.array_equal(value, z[key]) if np.issubdtype(value.dtype, np.integer) else np.allclose(value, z[key], atol=1e-5, rtol=1e-5)
            if not ok:
                raise RuntimeError("V-E native baseline cache mismatch: " + key)
            errors[key] = float(np.max(np.abs(value - z[key]))) if value.size else 0.
    return errors


def captured_target(inputs, ds, index, task):
    proposal = inputs[0][0]
    captured = dict(proposal_boxes=proposal.bbox.detach(), size=proposal.size)
    return targets(captured, ds.get_groundtruth(index, evaluation=True), task)


def smoke(args):
    out = OUT / args.task / "smoke"
    model, cfg, transform, provenance, head, patch = setup(args.task, out)
    splits = prepare_splits(args.task, cfg, provenance)
    ds = dataset(cfg, "val")
    mapping = lookup(ds)
    records = []
    for iid in splits["development"][:3]:
        i = mapping[iid]
        with torch.no_grad():
            head.up.weight.zero_(); head.up.bias.zero_()
        patch.enabled = False
        base, _ = infer(model, cfg, transform, ds, i)
        cache_error = native_cache_check(base, args.task, iid)
        patch.enabled = True
        zero, _ = infer(model, cfg, transform, ds, i)
        noop = compare_predictions(base, zero)
        patch.enabled = False
        holder, _ = capture(model, cfg, transform, ds, i)
        inputs, teacher = holder["inputs"], holder["output"]
        frozen = fingerprint(list(tensor_leaves(inputs)))
        y = torch.as_tensor(captured_target(inputs, ds, i, args.task), device="cuda")
        # Consistency losses are zero at a zero update. A small synthetic
        # development-only perturbation tests their actual gradient routes.
        with torch.no_grad():
            head.up.weight.normal_(0, .01)
        patch.enabled = True
        student = model.roi_heads.relation.predictor(*inputs)
        terms = objective_terms(student, teacher, y, inputs[1][0])
        grads = {}
        for name, loss in terms.items():
            g = torch.autograd.grad(loss, tuple(head.parameters()), retain_graph=True, allow_unused=True)
            norm = sum(float(v.square().sum()) for v in g if v is not None) ** .5
            if not np.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError("V-E missing finite gradient route: " + name)
            grads[name] = norm
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError("Frozen native parameters unexpectedly trainable")
        if frozen != fingerprint(list(tensor_leaves(inputs))):
            raise RuntimeError("V-E replay mutated fixed visual inputs")
        updated, _ = infer(model, cfg, transform, ds, i)
        score_delta = float((updated.get_field("pred_scores") - base.get_field("pred_scores")).abs().max())
        if not patch.capture.get("final_route_checked") or score_delta <= 1e-8:
            raise RuntimeError("Shared adapter did not affect native emitted scores")
        records.append(dict(image_id=iid, noop_error=noop, cache_errors=cache_error,
                            nonzero_gradient_norms=grads, emitted_score_change=score_delta))
        print(json.dumps(records[-1]), flush=True)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", task=args.task, images=3,
                registration_sha256=verify(), split_sha256=sha256(split_path(args.task)), checks=records,
                native_parameters_frozen=True, synthetic_head_discarded=True))


def dev_nll(model, cfg, transform, patch, ds, mapping, ids):
    patch.enabled = True
    total, count = 0., 0
    for iid in ids:
        _, _ = infer(model, cfg, transform, ds, mapping[iid])
        y = torch.as_tensor(targets(patch.capture, ds.get_groundtruth(mapping[iid], evaluation=True),
                                   "sgcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX else "sgdet"), device="cuda")
        positive = y > 0
        if positive.any():
            total += float(F.cross_entropy(patch.capture["logits"][positive], y[positive], reduction="sum"))
            count += int(positive.sum())
    if count == 0:
        raise RuntimeError("No development foreground objects")
    return total / count


def train(args):
    out = OUT / args.task / "training" / (args.mode + "_seed%d" % args.seed)
    if (out / "summary.json").exists():
        row = read(out / "summary.json")
        if row.get("registration_sha256") != verify() or sha256(checkpoint(args.task, args.mode, args.seed)) != row["checkpoint_sha256"]:
            raise RuntimeError("Incompatible completed V-E training")
        return
    if args.seed != 17 and not read(OUT / "pilot_decision.json")["accepted"]:
        raise RuntimeError("V-E pilot did not permit extra seeds")
    if read(OUT / args.task / "smoke/summary.json").get("status") != "complete":
        raise RuntimeError("V-E native gradient/no-op smoke not completed")
    model, cfg, transform, provenance, head, patch = setup(args.task, out, args.seed)
    splits = read(split_path(args.task))
    train_ds, val_ds = dataset(cfg, "train"), dataset(cfg, "val")
    tr, va = lookup(train_ds), lookup(val_ds)
    optim = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    reg, split_hash = verify(), sha256(split_path(args.task))
    epoch, position, best, stale, history, total_loss, loss_images = 1, 0, float("inf"), 0, [], 0., 0
    last = ROOT / "checkpoints/VE" / args.task / (args.mode + "_seed%d.last.pth" % args.seed)
    if last.exists():
        saved = torch.load(str(last), map_location="cpu")
        if saved["registration_sha256"] != reg or saved["split_sha256"] != split_hash:
            raise RuntimeError("V-E resume provenance mismatch")
        head.load_state_dict(saved["head"]); optim.load_state_dict(saved["optimizer"])
        epoch, position, best, stale = saved["epoch"], saved["next_image"], saved["best"], saved["stale"]
        history, total_loss, loss_images = saved["history"], saved["total_loss"], saved["loss_images"]

    def persist(ep, pos):
        save_torch(last, dict(head=head.state_dict(), optimizer=optim.state_dict(), epoch=ep, next_image=pos,
                             best=best, stale=stale, history=history, total_loss=total_loss, loss_images=loss_images,
                             registration_sha256=reg, split_sha256=split_hash))

    started, computed = time.monotonic(), 0
    n = len(splits["train"])
    while epoch <= SPEC["max_epochs"]:
        if history and history[-1]["epoch"] >= SPEC["min_epochs"] and stale >= SPEC["patience"]:
            break
        order = np.random.default_rng(args.seed + epoch).permutation(n)
        head.train()
        for start in range(position, n, SPEC["accumulate_images"]):
            optim.zero_grad(set_to_none=True)
            stop = min(start + SPEC["accumulate_images"], n)
            for offset in range(start, stop):
                iid = splits["train"][int(order[offset])]
                index = tr[iid]
                patch.enabled = False
                holder, _ = capture(model, cfg, transform, train_ds, index)
                inputs, teacher = holder["inputs"], holder["output"]
                y = torch.as_tensor(captured_target(inputs, train_ds, index, args.task), device="cuda")
                patch.enabled = True
                student = model.roi_heads.relation.predictor(*inputs)
                terms = objective_terms(student, teacher, y, inputs[1][0], SPEC["background_weight"])
                loss = objective(terms, args.mode)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite V-E objective")
                (loss / (stop - start)).backward()
                total_loss += float(loss.detach()); loss_images += 1; computed += 1
                del holder, inputs, teacher, student, terms, loss
            grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SPEC["gradient_clip"]))
            if not np.isfinite(grad):
                raise RuntimeError("Nonfinite V-E gradient")
            optim.step()
            if stop % 100 == 0 or stop == n:
                persist(epoch, stop)
                elapsed = time.monotonic() - started
                progress = dict(epoch=epoch, images=stop, total=n, seconds=elapsed,
                                eta_seconds=elapsed / max(computed, 1) * (n - stop),
                                detail="%s %s seed%d epoch %d/%d; ETA for this training epoch only" %
                                (args.task, args.mode, args.seed, epoch, SPEC["max_epochs"]),
                                mean_objective=total_loss / loss_images, gradient_norm=grad)
                atomic_json(out / "progress.json", progress); print(json.dumps(progress), flush=True)
        head.eval()
        p = read(out / "progress.json")
        p.update(detail="development NLL checkpoint selection; 500 images", images=0, total=500, eta_seconds=None)
        atomic_json(out / "progress.json", p)
        val = dev_nll(model, cfg, transform, patch, val_ds, va, splits["development"])
        history.append(dict(epoch=epoch, development_foreground_nll=val, objective=total_loss / loss_images))
        if val < best - 1e-6:
            best, stale = val, 0
            save_torch(checkpoint(args.task, args.mode, args.seed),
                       dict(head=head.state_dict(), task=args.task, mode=args.mode, seed=args.seed, epoch=epoch,
                            registration_sha256=reg, split_sha256=split_hash, baseline_sha256=provenance["checkpoint_sha256"]))
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        epoch += 1; position = 0; total_loss = 0.; loss_images = 0
        persist(epoch, 0)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", registration_sha256=reg, history=history,
                best_development_nll=best, checkpoint=str(checkpoint(args.task, args.mode, args.seed)),
                checkpoint_sha256=sha256(checkpoint(args.task, args.mode, args.seed)), seconds=time.monotonic() - started,
                relation_loss_used=args.mode == "relation_aware", interpretation=SPEC["interpretation"]))


def evaluate(args):
    if args.split == "test" and not read(OUT / "pilot_decision.json")["accepted"]:
        raise RuntimeError("V-E test access blocked")
    condition = args.mode + "_seed%d" % args.seed
    out = OUT / args.task / args.split / condition
    model, cfg, transform, provenance, head, patch = setup(args.task, out, args.seed)
    splits = read(split_path(args.task))
    ds = dataset(cfg, "test" if args.split == "test" else "val")
    mapping = lookup(ds)
    ids = list(mapping) if args.split == "test" else splits["gate"]
    if (args.split == "test" and len(ids) != 26446) or set(ids) & set(splits["train"] + splits["development"]):
        raise RuntimeError("V-E evaluation split invalid")
    state_hash = None
    if args.mode != "native":
        path = checkpoint(args.task, args.mode, args.seed)
        saved = torch.load(str(path), map_location="cpu")
        if saved["registration_sha256"] != verify() or saved["split_sha256"] != sha256(split_path(args.task)) or saved["baseline_sha256"] != provenance["checkpoint_sha256"]:
            raise RuntimeError("V-E checkpoint provenance mismatch")
        head.load_state_dict(saved["head"]); state_hash = sha256(path)
    head.eval(); patch.enabled = args.mode != "native"
    immutable(out / "protocol.json", dict(registration_sha256=verify(), image_ids=ids, mode=args.mode, seed=args.seed,
                                          baseline_sha256=provenance["checkpoint_sha256"], adapter_sha256=state_hash))
    digest = sha256(out / "protocol.json")
    metric = OfficialMetrics(args.task)
    rows, start, computed = [], time.monotonic(), 0
    from export_pysgg_vg_task import convert_prediction
    for number, iid in enumerate(ids):
        path = out / "images" / (iid + ".json")
        cache = out / "predictions" / (iid + ".npz")
        if path.exists() and cache.exists():
            row = read(path)
            if row["protocol_sha256"] != digest:
                raise RuntimeError("Stale V-E evaluation row")
        else:
            i = mapping[iid]
            pred, _ = infer(model, cfg, transform, ds, i)
            gt = ds.get_groundtruth(i, evaluation=True)
            target = targets(patch.capture, gt, args.task)
            row = metric.row(iid, pred, gt, patch.capture, target)
            row["protocol_sha256"] = digest
            temp = output_path(cache.with_suffix(".tmp"))
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **convert_prediction(pred.to("cpu")))
            temp.replace(cache); atomic_json(path, row); computed += 1
        rows.append(row)
        if (number + 1) % 25 == 0 or number + 1 == len(ids):
            elapsed = time.monotonic() - start
            p = dict(images=number + 1, total=len(ids), seconds=elapsed,
                     eta_seconds=elapsed / max(computed, 1) * (len(ids) - number - 1),
                     detail="native downstream context, NMS and ranking rerun")
            atomic_json(out / "progress.json", p); print(json.dumps(p), flush=True)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", registration_sha256=verify(),
                protocol_sha256=digest, task=args.task, mode=args.mode, seed=args.seed, **summarize_rows(rows)))


def decision():
    checks = {}
    for task in SPEC["tasks"]:
        ids = read(split_path(task))["gate"]
        def rows(mode):
            folder = OUT / task / "gate" / (mode + "_seed17")
            if read(folder / "summary.json")["status"] != "complete":
                raise RuntimeError("Incomplete V-E gate")
            return [read(folder / "images" / (iid + ".json")) for iid in ids]
        base = rows("native")
        checks[task] = {m: paired_gate(base, rows(m)) for m in SPEC["modes"]}
    accepted = all(checks[t]["relation_aware"]["accepted"] for t in SPEC["tasks"])
    atomic_json(OUT / "pilot_decision.json", dict(status="complete", accepted=accepted, tasks=checks,
                registration_sha256=verify(), candidate="relation_aware_seed17",
                action="confirm seeds23/31 then all-seed tests" if accepted else "stop; no extra seeds, test or coefficient search",
                interpretation="Gate passing is not proof of superiority to the equally sized supervised control"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "train", "evaluate", "decision"], required=True)
    p.add_argument("--task", choices=SPEC["tasks"], default="sgcls")
    p.add_argument("--mode", choices=["native"] + SPEC["modes"], default="relation_aware")
    p.add_argument("--seed", type=int, choices=SPEC["seeds"], default=17)
    p.add_argument("--split", choices=["gate", "test"], default="gate")
    args = p.parse_args()
    ensure_storage(); verify(); torch.set_num_threads(4)
    if args.stage == "decision":
        decision()
    else:
        {"smoke": smoke, "train": train, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
