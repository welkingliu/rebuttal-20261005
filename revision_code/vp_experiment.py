"""V-P: train a DINO classifier on detected proposals, with nested image splits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources
from repro_experiment import save_torch
from vp_math import SPEC, PRIMARY, ARMS, split_fold, make_head, fit_head, fusion_logits


OUT = ROOT/"results/VP_proposal_expert"
CACHE = ROOT/"cache/VP_proposal_expert"
WEIGHTS = ROOT/"checkpoints/VP_proposal_expert"
VN = ROOT/"results/VN_train_proposal_router/train3000"
INPUT = ROOT/"cache/VN_train_proposal_router/train3000"
VO = ROOT/"results/VO_grouped_train_validation/train3000"
VO_CACHE = ROOT/"cache/VO_grouped_train_validation/train3000"
SOURCE_NAMES = ["vp_experiment.py", "vp_math.py", "vp_queue.py", "VP_PROPOSAL_EXPERT_PLAN.txt",
    "vn_train_proposals.py", "vb_native.py", "vc_protocol.py", "vo_math.py", "repro_experiment.py",
    "vk_gate_native.py", "vk_gate_protocol.py", "common.py"]


def registration(smoke):
    from vn_train_proposals import verify
    prior, prior_sha = verify(VN)
    previous = read(VO/"protocol.json")
    if previous["prior_protocol_sha256"] != prior_sha or previous["image_ids"] != prior["training_ids"]:
        raise RuntimeError("Training source isolation changed")
    for name, digest in previous["sources"].items():
        if sha256(Path(__file__).with_name(name)) != digest: raise RuntimeError("V-O source drift: "+name)
    ids = prior["training_ids"][:15] if smoke else prior["training_ids"]
    folds = {iid: (0 if i < 3 else 1) for i, iid in enumerate(ids)} if smoke else previous["folds"]
    if set(ids) & set(prior["evaluation_ids"]+prior["excluded_expert_images"]):
        raise RuntimeError("Image split overlap")
    protocol = dict(version="vp_proposal_domain_classifier_v1", smoke=smoke, spec=SPEC,
        image_ids=ids, folds=folds, primary=PRIMARY, arms=ARMS,
        vn_protocol_sha256=prior_sha, vo_protocol_sha256=sha256(VO/"protocol.json"),
        expert_sha256=prior["expert_sha256"], native_checkpoint_sha256=prior["native_checkpoint_sha256"],
        sources=sources(SOURCE_NAMES), plan=Path(__file__).with_name("VP_PROPOSAL_EXPERT_PLAN.txt").read_text(),
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False, independent_confirmation=False)
    lane = "smoke" if smoke else "train3000"
    return protocol, lock(OUT/lane/"protocol.json", protocol), lane


def progress(path, stage, count, total, start, **extra):
    elapsed = time.monotonic()-start
    if isinstance(extra.get("detail"), dict): extra["detail"] = json.dumps(extra["detail"], sort_keys=True)
    value = dict(status="running", pid=os.getpid(), stage=stage, images=count, total=total,
        seconds=elapsed, eta_seconds=elapsed/count*(total-count) if count else None, **extra)
    atomic_json(path/"progress.json", value); print(json.dumps(value), flush=True)


def finish(path, value):
    atomic_json(path/"summary.json", value)
    atomic_json(path/"progress.json", dict(status="complete", images=value["images"], total=value["images"]))
    print(json.dumps({k:v for k,v in value.items() if k not in ["arms", "comparisons"]}), flush=True)


def prepare(protocol, reg, lane):
    from vn_train_proposals import checked_load
    stage = OUT/lane/"prepare"; start = time.monotonic()
    path = CACHE/lane/"data.pt"
    if (stage/"summary.json").exists():
        s = read(stage/"summary.json")
        if s["protocol_sha256"] != reg or s["bundle_sha256"] != sha256(path): raise RuntimeError("Pack changed")
        return
    xx, yy, offsets, hashes = [], [], [0], {}
    for n, iid in enumerate(protocol["image_ids"], 1):
        paths = {k:INPUT/k/(iid+".pt") for k in ["raw", "features"]}
        reference = read(VO/"replay/images"/(iid+".json"))["inputs"]
        digests = {k:sha256(v) for k,v in paths.items()}
        if any(digests[k] != reference[k] for k in paths): raise RuntimeError("V-N input bytes changed")
        raw, feature = [checked_load(paths[k], protocol["vn_protocol_sha256"]) for k in ["raw", "features"]]
        if feature["raw_sha256"] != digests["raw"] or not torch.equal(raw["crop_boxes"], feature["crop_boxes"]):
            raise RuntimeError("Proposal feature alignment changed")
        x, y = feature["features"], raw["target"]
        if x.shape != (len(y), 768) or not torch.isfinite(x).all() or torch.any((y < -1) | (y > 150)):
            raise RuntimeError("Feature/ontology contract failed")
        if not torch.allclose(x.norm(dim=-1), torch.ones(len(x)), atol=2e-5): raise RuntimeError("Feature normalization differs")
        xx.append(x); yy.append(y); offsets.append(offsets[-1]+len(y)); hashes[iid] = digests
        if n % 100 == 0 or n == len(protocol["image_ids"]): progress(stage, "pack_detected_training_crops", n, len(protocol["image_ids"]), start)
    labels = torch.cat(yy).long()
    bundle = dict(protocol_sha256=reg, image_ids=protocol["image_ids"], offsets=offsets,
                  features=torch.cat(xx).float(), labels=labels)
    save_torch(path, bundle); lock(stage/"input_hashes.json", hashes)
    finish(stage, dict(status="complete", protocol_sha256=reg, images=len(xx), proposals=len(labels),
        positive=int((labels > 0).sum()), background=int((labels == 0).sum()), ambiguous=int((labels < 0).sum()),
        object_support=torch.bincount(labels[labels > 0], minlength=151)[1:].tolist(),
        bundle_sha256=sha256(path), elapsed_seconds=time.monotonic()-start))


def rows_for(bundle, ids):
    mapping = {iid:i for i,iid in enumerate(bundle["image_ids"])}
    return torch.cat([torch.arange(bundle["offsets"][mapping[i]], bundle["offsets"][mapping[i]+1]) for i in ids])


def load_bundle(reg, lane):
    path = CACHE/lane/"data.pt"; s = read(OUT/lane/"prepare/summary.json")
    if s["protocol_sha256"] != reg or s["bundle_sha256"] != sha256(path): raise RuntimeError("Bundle mismatch")
    bundle = torch.load(str(path), map_location="cpu")
    if bundle["protocol_sha256"] != reg: raise RuntimeError("Unregistered data")
    return bundle, s["bundle_sha256"]


def train_fold(protocol, reg, lane, fold, old):
    stage = OUT/lane/("fold%d" % fold); start = time.monotonic()
    split = split_fold(protocol["image_ids"], protocol["folds"], fold)
    lock(stage/"split.json", dict(protocol_sha256=reg, **split))
    bundle, bundle_sha = load_bundle(reg, lane)
    path = WEIGHTS/lane/("fold%d.pth" % fold)
    if (stage/"training_summary.json").exists():
        s = read(stage/"training_summary.json")
        if s["protocol_sha256"] != reg or s["checkpoint_sha256"] != sha256(path) or s["bundle_sha256"] != bundle_sha:
            raise RuntimeError("Fitted head provenance mismatch")
        return torch.load(str(path), map_location="cpu"), split, bundle
    fit, inner, all_train = [rows_for(bundle, split[k]) for k in ["fit", "inner_validation", "training"]]
    x, y = bundle["features"], bundle["labels"]
    selection_path = stage/"epoch_selection.json"
    if selection_path.exists():
        selection = read(selection_path)
        if selection["protocol_sha256"] != reg or selection["bundle_sha256"] != bundle_sha: raise RuntimeError("Selection provenance drift")
    else:
        _, history = fit_head(x[fit], y[fit], old, validation=(x[inner], y[inner]), smoke=protocol["smoke"],
            callback=lambda row,total: progress(stage, "inner_epoch_selection", row["epoch"], total, start, detail=row))
        selection = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, **history)
        atomic_json(selection_path, selection)
    epochs = selection["selected_epochs"]
    if not isinstance(epochs, int) or epochs < 1: raise RuntimeError("No selected epoch")
    refit_start = time.monotonic()
    state, refit = fit_head(x[all_train], y[all_train], old, epochs=epochs,
        callback=lambda row,total: progress(stage, "refit_all_outer_training", row["epoch"], total, refit_start, detail=row))
    change = float((state["weight"][1:]-old["weight"]).abs().max())
    if not change > 0: raise RuntimeError("Classifier weights never changed")
    payload = dict(protocol_sha256=reg, fold=fold, state_dict=state, fit_ids=split["training"],
        held_ids=split["held"], epochs=epochs, bundle_sha256=bundle_sha)
    save_torch(path, payload)
    valid = y[all_train] >= 0
    summary = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, images=len(split["training"]), held_images=len(split["held"]),
        selected_epochs=epochs, checkpoint_sha256=sha256(path), foreground_weight_max_change=change,
        foreground_class_support=torch.bincount(y[all_train][valid], minlength=151)[1:].tolist(),
        inner_selection_budget_hit=selection["hit_epoch_budget"], refit=refit, elapsed_seconds=time.monotonic()-start)
    atomic_json(stage/"training_summary.json", summary)
    return payload, split, bundle


def evaluate_fold(protocol, reg, lane, fold, payload, split, bundle, old):
    from vn_train_proposals import checked_load, make_post, replay, ground_truth
    from vb_native import OfficialMetrics, targets, compare_predictions
    from vc_protocol import KS, summarize_rows
    stage = OUT/lane/("fold%d" % fold); start = time.monotonic()
    head = make_head(old).cuda().eval(); head.load_state_dict(payload["state_dict"], strict=True)
    head.requires_grad_(False)
    old_gpu = {k:v.cuda() for k,v in old.items()}
    _, post = make_post(stage/"evaluation")
    metric = OfficialMetrics("sgdet"); rows = []; mapping = {v:i for i,v in enumerate(bundle["image_ids"])}
    ckpt_sha = sha256(WEIGHTS/lane/("fold%d.pth" % fold))
    for n, iid in enumerate(split["held"], 1):
        path = stage/"images"/(iid+".json")
        if path.exists():
            row = read(path)
            if row["protocol_sha256"] != reg or row["checkpoint_sha256"] != ckpt_sha: raise RuntimeError("Prediction resume mismatch")
        else:
            raw_path = INPUT/"raw"/(iid+".pt")
            meta = read(VO/"replay/images"/(iid+".json"))
            if sha256(raw_path) != meta["inputs"]["raw"]: raise RuntimeError("Replay input changed")
            raw = checked_load(raw_path, protocol["vn_protocol_sha256"])
            index = mapping[iid]; a,b = bundle["offsets"][index:index+2]
            x = bundle["features"][a:b].cuda(); baseline = raw["native_logits"].cuda()
            with torch.no_grad():
                qold = F.linear(x, old_gpu["weight"], old_gpu["bias"]).softmax(-1)
                qnew = head(x).softmax(-1)
                logits = dict(native=baseline, old_gt_fusion=fusion_logits(baseline, qold),
                    proposal_fg_fusion=fusion_logits(baseline, qnew),
                    proposal_full_fusion=fusion_logits(baseline, qnew, False))
                predictions = {k:replay(post, raw, v) for k,v in logits.items()}
                if n == 1:
                    no_op = replay(post, raw, fusion_logits(baseline, qnew, alpha=0))
                    if compare_predictions(predictions["native"], no_op) != 0: raise RuntimeError("No-op replay not exact")
                # Predictions are committed before GT is attached for scoring.
                gt = ground_truth(raw); target = targets(raw, gt, "sgdet")
                if not np.array_equal(target, bundle["labels"][a:b].numpy()): raise RuntimeError("Proposal target alignment changed")
                scored = {k:metric.row(iid, predictions[k], gt, dict(logits=logits[k]), target) for k in ARMS}
                prior_path = VO_CACHE/(iid+".pt")
                if sha256(prior_path) != meta["cache_sha256"]: raise RuntimeError("Reference baseline changed")
                prior = torch.load(str(prior_path), map_location="cpu")["baseline"]
                for key in ["recalls", "class_recalls", "positive_objects", "positive_correct", "post_nms_correct"]:
                    if prior[key] != scored["native"][key]: raise RuntimeError("Native baseline metric parity failed: "+key)
                positive = torch.from_numpy(target > 0).cuda(); truth = torch.from_numpy(target).cuda()
                row = dict(image_id=iid, fold=fold, protocol_sha256=reg, checkpoint_sha256=ckpt_sha, metrics=scored,
                    old_expert_fg_correct=int(((qold.argmax(-1)+1 == truth) & positive).sum()),
                    new_expert_fg_correct=int(((qnew[:, 1:].argmax(-1)+1 == truth) & positive).sum()),
                    proposal_count=len(target), positive_objects=int(positive.sum()),
                    pre_nms_label_changes={k:int((logits[k][:, 1:].argmax(-1) != baseline[:, 1:].argmax(-1)).sum()) for k in ARMS[1:]})
                atomic_json(path, row)
                for key in KS: metric.result["sgdet_recall"][key].clear()
        rows.append(row)
        if n % 25 == 0 or n == len(split["held"]): progress(stage, "heldout_native_sgdet_replay", n, len(split["held"]), start)
    finish(stage, dict(status="complete", protocol_sha256=reg, fold=fold, images=len(rows),
        checkpoint_sha256=ckpt_sha, arms={name:summarize_rows([r["metrics"][name] for r in rows]) for name in ARMS},
        primary=PRIMARY, validation_accessed=False, test_accessed=False, formal_gate_accepted=False,
        independent_confirmation=False, elapsed_seconds=time.monotonic()-start))


def summarize(protocol, reg, lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]: raise ValueError("Smoke does not estimate scientific efficacy")
    start = time.monotonic(); records = []
    for fold in range(5):
        stage = OUT/lane/("fold%d" % fold)
        completed = read(stage/"summary.json")
        ckpt_sha = sha256(WEIGHTS/lane/("fold%d.pth" % fold))
        if completed["protocol_sha256"] != reg or completed["checkpoint_sha256"] != ckpt_sha:
            raise RuntimeError("Incomplete fold or changed checkpoint")
        split = read(stage/"split.json")
        for iid in split["held"]:
            row = read(stage/"images"/(iid+".json"))
            if row["protocol_sha256"] != reg or row["fold"] != fold or row["checkpoint_sha256"] != ckpt_sha:
                raise RuntimeError("Fold row changed")
            records.append(row)
    by_id = {r["image_id"]:r for r in records}
    if len(records) != 3000 or len(by_id) != 3000 or set(by_id) != set(protocol["image_ids"]): raise RuntimeError("Incomplete/disordered OOF coverage")
    records = [by_id[i] for i in protocol["image_ids"]]
    aggregate = {name:summarize_rows([r["metrics"][name] for r in records]) for name in ARMS}
    folds = np.asarray([r["fold"] for r in records]); comparisons = {}
    for reference, method in [("native", m) for m in ARMS[1:]]+[("old_gt_fusion", PRIMARY)]:
        base = [r["metrics"][reference] for r in records]; trial = [r["metrics"][method] for r in records]
        pos = np.asarray([r["positive_objects"] for r in base])
        if not np.array_equal(pos, [r["positive_objects"] for r in trial]): raise RuntimeError("Object denominator drift")
        bc = np.asarray([r["class_recalls"][4] for r in base], dtype=float)
        tc = np.asarray([r["class_recalls"][4] for r in trial], dtype=float)
        if not np.array_equal(np.isfinite(bc), np.isfinite(tc)): raise RuntimeError("Predicate support drift")
        gain = np.asarray([u["post_nms_correct"]-b["post_nms_correct"] for b,u in zip(base,trial)])
        dr = np.asarray([u["recalls"][4]-b["recalls"][4] for b,u in zip(base,trial)])
        stats = policy_summary(np.arange(3000), np.arange(3000), gain, dr, np.nan_to_num(tc-bc), np.isfinite(bc), pos, folds)
        stats["updated_images"] = stats.pop("selected_images")
        stats["scope"] = "All proposals fused in every image; not the V-O one-action policy"
        comparisons[method+"_versus_"+reference] = stats
    check = comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    total = sum(r["positive_objects"] for r in records)
    convergence = [{k:v for k,v in read(OUT/lane/("fold%d" % fold)/"training_summary.json").items()
        if k not in ["refit", "foreground_class_support"]} for fold in range(5)]
    finish(OUT/lane, dict(status="complete", protocol_sha256=reg, images=3000, primary=PRIMARY,
        arms=aggregate, comparisons=comparisons, convergence=convergence,
        old_expert_fg_top1=sum(r["old_expert_fg_correct"] for r in records)/total,
        new_expert_fg_top1=sum(r["new_expert_fg_correct"] for r in records)/total,
        primary_training_screen_satisfied=all(check.values()), formal_gate_accepted=False,
        independent_confirmation=False, validation_accessed=False, test_accessed=False,
        native_sgg_trained_on_these_images=True, training_domain_previously_used_for_diagnostics=True,
        elapsed_seconds=time.monotonic()-start,
        next_step="Stop. Independent two-task confirmation requires separate registration and exposure audit; no automatic test, seeds, or control promotion."))


def main():
    from vk_gate_protocol import CHECKPOINT
    from vk_gate_native import configure_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "fold", "summarize"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); ensure_storage(); torch.set_num_threads(2); configure_backend()
    protocol, reg, lane = registration(args.smoke)
    name = "fold%d" % args.fold if args.stage == "fold" and args.fold is not None else args.stage
    guard = output_path(OUT/lane/(name+".lock")).open("w"); fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage == "prepare": prepare(protocol, reg, lane)
    elif args.stage == "fold":
        if args.fold is None or (args.smoke and args.fold != 0): raise ValueError("Invalid fold")
        if (OUT/lane/name/"summary.json").exists():
            completed = read(OUT/lane/name/"summary.json")
            if (completed["protocol_sha256"] != reg
                    or completed["checkpoint_sha256"] != sha256(WEIGHTS/lane/(name+".pth"))):
                raise RuntimeError("Completed result changed")
            print("Already complete: "+name, flush=True); return
        old = torch.load(str(CHECKPOINT), map_location="cpu")["expert_head"]
        payload, split, bundle = train_fold(protocol, reg, lane, args.fold, old)
        evaluate_fold(protocol, reg, lane, args.fold, payload, split, bundle, old)
    else: summarize(protocol, reg, lane)


if __name__ == "__main__": main()
