"""V-Q train-only residual correction with native post-NMS model selection."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources
from repro_experiment import save_torch
from vp_experiment import (registration as vp_registration, load_bundle as vp_bundle,
    INPUT, VO, VO_CACHE, progress, finish, rows_for, SOURCE_NAMES as VP_SOURCES)
from vp_math import split_fold, fit_head, make_head, fusion_logits
from vq_math import SPEC, PRIMARY, ARMS, ResidualHead, residual_logits, fit, choose_epoch, early_expert_epoch


OUT = ROOT/"results/VQ_bounded_residual"
CACHE = ROOT/"cache/VQ_bounded_residual"
WEIGHTS = ROOT/"checkpoints/VQ_bounded_residual"
SOURCE_NAMES = sorted(set(VP_SOURCES+["vq_experiment.py", "vq_math.py", "repair_panel_queue.py", "VQ_RESIDUAL_PLAN.txt"]))


def registration(smoke):
    prior, prior_sha, lane = vp_registration(smoke)
    protocol = dict(version="vq_native_bounded_residual_v1", smoke=smoke, spec=SPEC,
        image_ids=prior["image_ids"], folds=prior["folds"], primary=PRIMARY, arms=ARMS,
        vp_protocol_sha256=prior_sha, vn_protocol_sha256=prior["vn_protocol_sha256"],
        native_checkpoint_sha256=prior["native_checkpoint_sha256"], sources=sources(SOURCE_NAMES),
        plan=Path(__file__).with_name("VQ_RESIDUAL_PLAN.txt").read_text(),
        validation_accessed=False, test_accessed=False, independent_confirmation=False,
        formal_gate_accepted=False, designed_after_prior_training_results=True)
    return protocol, lock(OUT/lane/"protocol.json", protocol), lane


def prepare(protocol, reg, lane):
    from vn_train_proposals import checked_load
    stage = OUT/lane/"prepare"; start = time.monotonic(); path = CACHE/lane/"data.pt"
    if (stage/"summary.json").exists():
        info = read(stage/"summary.json")
        if info["protocol_sha256"] != reg or info["bundle_sha256"] != sha256(path): raise RuntimeError("Data pack changed")
        return
    bundle, prior_sha = vp_bundle(protocol["vp_protocol_sha256"], lane)
    if bundle["image_ids"] != protocol["image_ids"]: raise RuntimeError("Image ordering changed")
    logits = []
    for i, iid in enumerate(protocol["image_ids"]):
        raw_path = INPUT/"raw"/(iid+".pt")
        if sha256(raw_path) != read(VO/"replay/images"/(iid+".json"))["inputs"]["raw"]:
            raise RuntimeError("Native input drift")
        raw = checked_load(raw_path, protocol["vn_protocol_sha256"])
        a,b = bundle["offsets"][i:i+2]
        if not torch.equal(bundle["labels"][a:b], raw["target"]): raise RuntimeError("Target alignment changed")
        native = raw["native_logits"].float()
        if native.shape != (b-a, 151) or not torch.isfinite(native).all(): raise RuntimeError("Invalid logits")
        logits.append(native)
        if (i+1) % 100 == 0 or i+1 == len(protocol["image_ids"]):
            progress(stage, "pack_frozen_native_logits", i+1, len(protocol["image_ids"]), start)
    bundle.update(protocol_sha256=reg, vp_bundle_sha256=prior_sha, native_logits=torch.cat(logits))
    save_torch(path, bundle)
    finish(stage, dict(status="complete", protocol_sha256=reg, images=len(protocol["image_ids"]),
        proposals=len(bundle["labels"]), positive_objects=int((bundle["labels"] > 0).sum()),
        bundle_sha256=sha256(path), vp_bundle_sha256=prior_sha, elapsed_seconds=time.monotonic()-start))


def load_bundle(reg, lane):
    path = CACHE/lane/"data.pt"; info = read(OUT/lane/"prepare/summary.json")
    if info["protocol_sha256"] != reg or info["bundle_sha256"] != sha256(path): raise RuntimeError("Pack mismatch")
    bundle = torch.load(str(path), map_location="cpu")
    if bundle["protocol_sha256"] != reg: raise RuntimeError("Unregistered bundle")
    return bundle, info["bundle_sha256"]


def image_inputs(protocol, bundle, ids):
    from vn_train_proposals import checked_load, ground_truth
    from vb_native import targets
    mapping = {iid:i for i,iid in enumerate(bundle["image_ids"])}
    records = []
    for iid in ids:
        meta = read(VO/"replay/images"/(iid+".json")); path = INPUT/"raw"/(iid+".pt")
        if sha256(path) != meta["inputs"]["raw"]: raise RuntimeError("Raw cache changed")
        raw = checked_load(path, protocol["vn_protocol_sha256"])
        i = mapping[iid]; a,b = bundle["offsets"][i:i+2]
        if not torch.equal(bundle["native_logits"][a:b], raw["native_logits"].float()):
            raise RuntimeError("Native logit ordering changed")
        gt = ground_truth(raw); target = targets(raw, gt, "sgdet")
        if not np.array_equal(target, bundle["labels"][a:b].numpy()): raise RuntimeError("Fixed denominator changed")
        prior_path = VO_CACHE/(iid+".pt")
        if sha256(prior_path) != meta["cache_sha256"]: raise RuntimeError("Native baseline cache drift")
        previous = torch.load(str(prior_path), map_location="cpu")["baseline"]
        records.append(dict(iid=iid, raw=raw, x=bundle["features"][a:b], gt=gt, target=target, baseline=previous))
    return records


def assert_baseline(actual, expected):
    for k in ["recalls", "class_recalls", "positive_objects", "positive_correct", "post_nms_correct"]:
        if actual[k] != expected[k]: raise RuntimeError("Native baseline parity failed: "+k)


def score_images(records, transform, post, metric, stage, label, native=False, context=False):
    from vn_train_proposals import replay
    from vc_protocol import KS
    from vb_native import compare_predictions
    start = time.monotonic(); scored = []
    for i, rec in enumerate(records):
        raw = rec["raw"]; base = raw["native_logits"].float().cuda()
        with torch.no_grad():
            prediction_context = {k:raw[k] for k in ["pairs", "relation_logits"]}
            logits = transform(rec["x"].cuda(), base, prediction_context) if context else transform(rec["x"].cuda(), base)
            if not torch.isfinite(logits).all(): raise RuntimeError("Invalid repair logits")
            pred = replay(post, raw, logits)
            if native and i == 0 and compare_predictions(pred, replay(post, raw, base)) != 0:
                raise RuntimeError("Residual initialization is not an exact no-op")
            row = metric.row(rec["iid"], pred, rec["gt"], dict(logits=logits), rec["target"])
            if native: assert_baseline(row, rec["baseline"])
            scored.append(row)
            for k in KS: metric.result["sgdet_recall"][k].clear()
        if (i+1) % 100 == 0 or i+1 == len(records):
            progress(stage, label, i+1, len(records), start)
    return scored


def inner_select(protocol, reg, lane, fold, bundle, bundle_sha, split, visual, post, metric):
    from vc_protocol import summarize_rows
    stage = OUT/lane/("fold%d" % fold); name = PRIMARY if visual else "bias_residual"
    path = stage/("selection_"+name+".json")
    if path.exists():
        saved = read(path)
        if saved["protocol_sha256"] != reg or saved["bundle_sha256"] != bundle_sha: raise RuntimeError("Selection drift")
        return saved
    ix = rows_for(bundle, split["fit"])
    records = image_inputs(protocol, bundle, split["inner_validation"])
    candidates = [0, 1, 2] if protocol["smoke"] else SPEC["epochs"]
    summaries = []; start = time.monotonic()

    def callback(model, row):
        epoch = row["epoch"]
        if epoch in candidates:
            model.eval()
            values = score_images(records, lambda x,n: residual_logits(n, model(x)), post, metric,
                stage, "inner_%s_epoch%d" % (name, epoch), native=epoch == 0)
            summary = summarize_rows(values)
            summaries.append(dict(epoch=epoch, object=summary["post_nms_object_top1"],
                R50=summary["R"]["50"], mR50=summary["mR"]["50"], metrics=summary))
            atomic_json(stage/("candidate_progress_"+name+".json"), dict(protocol_sha256=reg, candidates=summaries))
        progress(stage, "fit_inner_"+name, epoch, max(candidates), start, detail=row)

    state, history = fit(bundle["features"][ix], bundle["native_logits"][ix], bundle["labels"][ix],
        visual, max(candidates), callback=callback)
    max_change = max(float(t.abs().max()) for t in state.values())
    if not max_change > 0: raise RuntimeError("No residual training gradients reached parameters")
    saved = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, candidates=summaries,
        history=history, trained_parameter_max_change=max_change, **choose_epoch(summaries))
    atomic_json(path, saved)
    return saved


def train_fold(protocol, reg, lane, fold, post, metric):
    from vk_gate_protocol import CHECKPOINT
    stage = OUT/lane/("fold%d" % fold); start = time.monotonic()
    split = split_fold(protocol["image_ids"], protocol["folds"], fold)
    lock(stage/"split.json", dict(protocol_sha256=reg, **split))
    bundle, bundle_sha = load_bundle(reg, lane); path = WEIGHTS/lane/("fold%d.pth" % fold)
    if (stage/"training_summary.json").exists():
        info = read(stage/"training_summary.json")
        if (info["protocol_sha256"] != reg or info["checkpoint_sha256"] != sha256(path)
                or info["bundle_sha256"] != bundle_sha): raise RuntimeError("Fitted checkpoint changed")
        return torch.load(str(path), map_location="cpu"), split, bundle, info
    selections = {name:inner_select(protocol, reg, lane, fold, bundle, bundle_sha, split, name == PRIMARY, post, metric)
        for name in [PRIMARY, "bias_residual"]}
    rows = rows_for(bundle, split["training"]); x,y,n = [bundle[k][rows] for k in ["features", "labels", "native_logits"]]
    states = {}; epochs = {}; histories = {}
    for name, selection in selections.items():
        epochs[name] = selection["selected_epoch"]; refit_start = time.monotonic()
        states[name], histories[name] = fit(x, n, y, name == PRIMARY, epochs[name],
            callback=lambda model,row: progress(stage, "refit_"+name, row["epoch"], epochs[name], refit_start, detail=row))
    old_selection_path = ROOT/"results/VP_proposal_expert"/lane/("fold%d" % fold)/"epoch_selection.json"
    old_selection = read(old_selection_path)
    if old_selection["protocol_sha256"] != protocol["vp_protocol_sha256"]: raise RuntimeError("V-P history registration differs")
    epochs["early_expert"] = early_expert_epoch(old_selection["history"])
    old = torch.load(str(CHECKPOINT), map_location="cpu")["expert_head"]; refit_start = time.monotonic()
    states["early_expert"], histories["early_expert"] = fit_head(x, y, old, epochs=epochs["early_expert"],
        callback=lambda row,total: progress(stage, "refit_early_expert", row["epoch"], total, refit_start, detail=row))
    payload = dict(protocol_sha256=reg, fold=fold, states=states, epochs=epochs,
        fit_ids=split["training"], held_ids=split["held"], bundle_sha256=bundle_sha)
    save_torch(path, payload)
    info = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, checkpoint_sha256=sha256(path),
        images=len(split["training"]), held_images=len(split["held"]), selected_epochs=epochs,
        early_control_source_sha256=sha256(old_selection_path),
        selection_parameter_changes={k:v["trained_parameter_max_change"] for k,v in selections.items()},
        histories=histories, elapsed_seconds=time.monotonic()-start)
    atomic_json(stage/"training_summary.json", info)
    return payload, split, bundle, info


def evaluate_fold(protocol, reg, lane, fold, payload, split, bundle, info, post, metric):
    from vc_protocol import summarize_rows
    from vk_gate_protocol import CHECKPOINT
    stage = OUT/lane/("fold%d" % fold); start = time.monotonic()
    records = image_inputs(protocol, bundle, split["held"]); arms = {}
    old = torch.load(str(CHECKPOINT), map_location="cpu")["expert_head"]
    for name in ARMS:
        if name == "native": transform = lambda x,n: n
        elif name == "early_expert":
            head = make_head(old).cuda().eval(); head.load_state_dict(payload["states"][name], strict=True)
            transform = lambda x,n: fusion_logits(n, head(x).softmax(-1))
        else:
            head = ResidualHead(name == PRIMARY).cuda().eval(); head.load_state_dict(payload["states"][name], strict=True)
            transform = lambda x,n: residual_logits(n, head(x))
        is_noop = name == "native" or payload["epochs"].get(name) == 0
        arms[name] = score_images(records, transform, post, metric, stage, "heldout_"+name, native=is_noop)
    for i, iid in enumerate(split["held"]):
        atomic_json(stage/"images"/(iid+".json"), dict(protocol_sha256=reg, fold=fold,
            checkpoint_sha256=info["checkpoint_sha256"], image_id=iid, metrics={k:arms[k][i] for k in ARMS}))
    finish(stage, dict(status="complete", protocol_sha256=reg, checkpoint_sha256=info["checkpoint_sha256"],
        fold=fold, images=len(records), primary=PRIMARY, selected_epochs=payload["epochs"],
        arms={k:summarize_rows(arms[k]) for k in ARMS},
        smoke_gradients_exercised=all(v > 0 for v in info["selection_parameter_changes"].values()),
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False,
        independent_confirmation=False, elapsed_seconds=time.monotonic()-start))


def summarize(protocol, reg, lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]: raise ValueError("Smoke is not an efficacy estimate")
    start = time.monotonic(); records = []; epoch_choices = []
    for fold in range(5):
        stage = OUT/lane/("fold%d" % fold); complete = read(stage/"summary.json")
        digest = sha256(WEIGHTS/lane/("fold%d.pth" % fold))
        if complete["protocol_sha256"] != reg or complete["checkpoint_sha256"] != digest: raise RuntimeError("Fold changed")
        epoch_choices.append(dict(fold=fold, selected_epochs=complete["selected_epochs"]))
        split = read(stage/"split.json")
        for iid in split["held"]:
            row = read(stage/"images"/(iid+".json"))
            if row["protocol_sha256"] != reg or row["checkpoint_sha256"] != digest or row["fold"] != fold:
                raise RuntimeError("Prediction provenance drift")
            records.append(row)
    mapping = {r["image_id"]:r for r in records}
    if len(records) != 3000 or len(mapping) != 3000 or set(mapping) != set(protocol["image_ids"]): raise RuntimeError("OOF coverage incomplete")
    records = [mapping[i] for i in protocol["image_ids"]]; folds = np.asarray([r["fold"] for r in records])
    comparisons = {}
    for reference, name in [("native", m) for m in ARMS[1:]]+[("bias_residual", PRIMARY)]:
        base = [r["metrics"][reference] for r in records]; trial = [r["metrics"][name] for r in records]
        pos = np.asarray([r["positive_objects"] for r in base]); support = np.asarray([r["class_recalls"][4] for r in base], dtype=float)
        other = np.asarray([r["class_recalls"][4] for r in trial], dtype=float)
        if not np.array_equal(pos, [r["positive_objects"] for r in trial]) or not np.array_equal(np.isfinite(support), np.isfinite(other)):
            raise RuntimeError("Metric support drift")
        gain = np.asarray([u["post_nms_correct"]-b["post_nms_correct"] for b,u in zip(base,trial)])
        dr = np.asarray([u["recalls"][4]-b["recalls"][4] for b,u in zip(base,trial)])
        stats = policy_summary(np.arange(3000), np.arange(3000), gain, dr, np.nan_to_num(other-support), np.isfinite(support), pos, folds)
        stats["evaluated_images"] = stats.pop("selected_images")
        stats["scope"] = "All held-out images including any epoch-zero/no-op fold; not action selection"
        comparisons[name+"_versus_"+reference] = stats
    checks = comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(OUT/lane, dict(status="complete", protocol_sha256=reg, primary=PRIMARY, images=3000,
        arms={name:summarize_rows([r["metrics"][name] for r in records]) for name in ARMS}, comparisons=comparisons,
        selected_epochs=epoch_choices, primary_training_screen_satisfied=all(checks.values()),
        formal_gate_accepted=False, independent_confirmation=False, validation_accessed=False, test_accessed=False,
        native_sgg_trained_on_these_images=True, designed_after_prior_training_results=True,
        elapsed_seconds=time.monotonic()-start, next_step="Stop after exploratory train-only screen; no automatic control promotion, seeds or formal gate."))


def main():
    from vk_gate_native import configure_backend
    from vn_train_proposals import make_post
    from vb_native import OfficialMetrics
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
        completed = OUT/lane/name/"summary.json"
        if completed.exists():
            record = read(completed)
            if record["protocol_sha256"] != reg or record["checkpoint_sha256"] != sha256(WEIGHTS/lane/(name+".pth")):
                raise RuntimeError("Completed fold changed")
            print("Already complete: "+name, flush=True); return
        _, post = make_post(OUT/lane/name/"runtime"); metric = OfficialMetrics("sgdet")
        payload, split, bundle, info = train_fold(protocol, reg, lane, args.fold, post, metric)
        evaluate_fold(protocol, reg, lane, args.fold, payload, split, bundle, info, post, metric)
    else: summarize(protocol, reg, lane)


if __name__ == "__main__": main()
