"""Train-only kernel and relational-prior repair branches, without gate access."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, ensure_storage, output_path, atomic_json, sha256
from evidence_completion import read, sources, lock
from repro_experiment import save_torch
from vp_experiment import rows_for, progress, finish, INPUT, VO
from vp_math import split_fold
from vq_experiment import (registration as vq_registration, load_bundle, image_inputs,
    score_images, SOURCE_NAMES as VQ_SOURCES)
from vq_math import choose_epoch
from repair_panel_math import STRENGTHS, fit_ridge, ridge_logits, make_statistics, prior_logits


BRANCHES = dict(VR=("VR_kernel_residual", "kernel_residual", "linear_residual"),
                VS=("VS_visual_prior", "visual_prior", "relation_prior"))
SOURCE_NAMES = sorted(set(VQ_SOURCES+["repair_panel_experiment.py", "repair_panel_math.py",
    "VR_VS_PLAN.txt", "EXPERIMENT_V_LITERATURE_20261003.txt"]))


def registration(branch, smoke):
    prior, prior_sha, lane = vq_registration(smoke)
    name, primary, control = BRANCHES[branch]; out = ROOT/"results"/name/lane
    protocol = dict(version="repair_panel_v1", branch=branch, smoke=smoke, image_ids=prior["image_ids"],
        folds=prior["folds"], vq_protocol_sha256=prior_sha, vn_protocol_sha256=prior["vn_protocol_sha256"],
        arms=["native", primary, control], primary=primary, control=control, strengths=STRENGTHS,
        sources=sources(SOURCE_NAMES), plan=Path(__file__).with_name("VR_VS_PLAN.txt").read_text(),
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False, independent_confirmation=False)
    return protocol, lock(out/"protocol.json", protocol), lane, out


def fitting_raws(protocol, ids):
    from vn_train_proposals import checked_load
    for iid in ids:
        path = INPUT/"raw"/(iid+".pt")
        if sha256(path) != read(VO/"replay/images"/(iid+".json"))["inputs"]["raw"]: raise RuntimeError("Training annotation cache changed")
        yield checked_load(path, protocol["vn_protocol_sha256"])


def fitted_models(protocol, bundle, ids):
    primary, control = protocol["primary"], protocol["control"]
    if protocol["branch"] == "VS":
        state = make_statistics(fitting_raws(protocol, ids))
        return {primary:state, control:state}
    ix = rows_for(bundle, ids)
    return {name:fit_ridge(bundle["features"][ix], bundle["native_logits"][ix], bundle["labels"][ix], name == primary)
        for name in [primary, control]}


def transform_for(protocol, name, state, strength, expert):
    if name == "native": return lambda x,n,raw: n
    if protocol["branch"] == "VR": return lambda x,n,raw: ridge_logits(x, n, state, strength)
    return lambda x,n,raw: prior_logits(x, n, raw, state, expert, strength, name == protocol["primary"])


def run_fold(protocol, reg, lane, out, fold):
    from vn_train_proposals import make_post
    from vb_native import OfficialMetrics
    from vc_protocol import summarize_rows
    from vk_gate_protocol import CHECKPOINT
    stage = out/("fold%d" % fold); start = time.monotonic()
    weight_path = ROOT/"checkpoints"/BRANCHES[protocol["branch"]][0]/lane/("fold%d.pth" % fold)
    if (stage/"summary.json").exists():
        info = read(stage/"summary.json")
        if info["protocol_sha256"] != reg or info["checkpoint_sha256"] != sha256(weight_path): raise RuntimeError("Completed branch fold changed")
        print("Already complete: "+str(stage), flush=True); return
    bundle, bundle_sha = load_bundle(protocol["vq_protocol_sha256"], lane)
    split = split_fold(protocol["image_ids"], protocol["folds"], fold)
    lock(stage/"split.json", dict(protocol_sha256=reg, **split))
    expert = torch.load(str(CHECKPOINT), map_location="cpu")["expert_head"]
    _, post = make_post(stage/"runtime"); metric = OfficialMetrics("sgdet")
    training_path = stage/"training_summary.json"
    if training_path.exists():
        info = read(training_path)
        if (info["protocol_sha256"] != reg or info["bundle_sha256"] != bundle_sha
                or info["checkpoint_sha256"] != sha256(weight_path)): raise RuntimeError("Resume checkpoint drift")
        payload = torch.load(str(weight_path), map_location="cpu")
    else:
        progress(stage, "fit_inner_"+protocol["branch"], 0, len(split["fit"]), start)
        models = fitted_models(protocol, bundle, split["fit"])
        inner = image_inputs(protocol, bundle, split["inner_validation"]); choices = {}
        for name in protocol["arms"][1:]:
            candidates = []
            for index, strength in enumerate(STRENGTHS):
                scored = score_images(inner, transform_for(protocol, name, models[name], strength, expert), post, metric,
                    stage, "inner_%s_strength%.2f" % (name, strength), native=strength == 0, context=True)
                agg = summarize_rows(scored)
                candidates.append(dict(epoch=index, strength=strength, object=agg["post_nms_object_top1"],
                    R50=agg["R"]["50"], mR50=agg["mR"]["50"], metrics=agg))
            selected = choose_epoch(candidates); index = selected.pop("selected_epoch")
            choices[name] = dict(strength=STRENGTHS[index], candidate_index=index, candidates=candidates, **selected)
        del models
        progress(stage, "refit_all_outer_training_"+protocol["branch"], 0, len(split["training"]), start)
        models = fitted_models(protocol, bundle, split["training"])
        payload = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, states=models, choices=choices,
            fit_ids=split["training"], held_ids=split["held"], fold=fold)
        save_torch(weight_path, payload)
        info = dict(protocol_sha256=reg, bundle_sha256=bundle_sha, checkpoint_sha256=sha256(weight_path),
            choices=choices, fitting_images=len(split["training"]), held_images=len(split["held"]))
        atomic_json(training_path, info)
    outer = image_inputs(protocol, bundle, split["held"]); arms = {}
    for name in protocol["arms"]:
        strength = payload["choices"][name]["strength"] if name != "native" else 0
        state = payload["states"].get(name)
        arms[name] = score_images(outer, transform_for(protocol, name, state, strength, expert), post, metric,
            stage, "outer_"+name, native=strength == 0, context=True)
    for i, iid in enumerate(split["held"]):
        atomic_json(stage/"images"/(iid+".json"), dict(protocol_sha256=reg, image_id=iid, fold=fold,
            checkpoint_sha256=info["checkpoint_sha256"], metrics={k:arms[k][i] for k in protocol["arms"]}))
    finish(stage, dict(status="complete", protocol_sha256=reg, checkpoint_sha256=info["checkpoint_sha256"], fold=fold,
        images=len(outer), primary=protocol["primary"], selected_strengths={k:v["strength"] for k,v in payload["choices"].items()},
        arms={k:summarize_rows(v) for k,v in arms.items()}, validation_accessed=False, test_accessed=False,
        formal_gate_accepted=False, independent_confirmation=False, elapsed_seconds=time.monotonic()-start))


def summarize(protocol, reg, lane, out):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]: raise ValueError("Smoke does not estimate efficacy")
    start = time.monotonic(); records = []; choices = []
    for fold in range(5):
        stage = out/("fold%d" % fold); completed = read(stage/"summary.json")
        digest = sha256(ROOT/"checkpoints"/BRANCHES[protocol["branch"]][0]/lane/("fold%d.pth" % fold))
        if completed["protocol_sha256"] != reg or completed["checkpoint_sha256"] != digest: raise RuntimeError("Completed fold changed")
        choices.append(dict(fold=fold, selected_strengths=completed["selected_strengths"]))
        for iid in read(stage/"split.json")["held"]:
            row = read(stage/"images"/(iid+".json"))
            if row["protocol_sha256"] != reg or row["checkpoint_sha256"] != digest or row["fold"] != fold: raise RuntimeError("Prediction provenance mismatch")
            records.append(row)
    mapping = {r["image_id"]:r for r in records}
    if len(records) != 3000 or len(mapping) != 3000 or set(mapping) != set(protocol["image_ids"]): raise RuntimeError("Incomplete outer folds")
    records = [mapping[i] for i in protocol["image_ids"]]; folds = np.asarray([r["fold"] for r in records]); comparisons = {}
    for reference, name in [("native", m) for m in protocol["arms"][1:]]+[(protocol["control"], protocol["primary"])]:
        base = [r["metrics"][reference] for r in records]; trial = [r["metrics"][name] for r in records]
        pos = np.asarray([r["positive_objects"] for r in base]); support = np.asarray([r["class_recalls"][4] for r in base], dtype=float)
        other = np.asarray([r["class_recalls"][4] for r in trial], dtype=float)
        if not np.array_equal(pos, [r["positive_objects"] for r in trial]) or not np.array_equal(np.isfinite(support), np.isfinite(other)):
            raise RuntimeError("Support drift")
        gain = np.asarray([u["post_nms_correct"]-b["post_nms_correct"] for b,u in zip(base,trial)])
        dr = np.asarray([u["recalls"][4]-b["recalls"][4] for b,u in zip(base,trial)])
        stats = policy_summary(np.arange(3000), np.arange(3000), gain, dr, np.nan_to_num(other-support), np.isfinite(support), pos, folds)
        stats["evaluated_images"] = stats.pop("selected_images")
        stats["scope"] = "All held-out images including no-op folds; not action-level selections"
        comparisons[name+"_versus_"+reference] = stats
    check = comparisons[protocol["primary"]+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(out, dict(status="complete", protocol_sha256=reg, images=3000, primary=protocol["primary"],
        arms={k:summarize_rows([r["metrics"][k] for r in records]) for k in protocol["arms"]}, comparisons=comparisons,
        choices=choices, primary_training_screen_satisfied=all(check.values()), formal_gate_accepted=False,
        independent_confirmation=False, validation_accessed=False, test_accessed=False,
        native_sgg_trained_on_these_images=True, designed_after_prior_training_results=True,
        elapsed_seconds=time.monotonic()-start, next_step="Stop; no control promotion or automatic formal gate."))


def main():
    from vk_gate_native import configure_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--stage", choices=["fold", "summarize"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); ensure_storage(); torch.set_num_threads(2); configure_backend()
    protocol, reg, lane, out = registration(args.branch, args.smoke)
    name = "fold%d" % args.fold if args.fold is not None else args.stage
    guard = output_path(out/(name+".lock")).open("w"); fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage == "fold":
        if args.fold is None or (args.smoke and args.fold != 0): raise ValueError("Invalid fold")
        run_fold(protocol, reg, lane, out, args.fold)
    else: summarize(protocol, reg, lane, out)


if __name__ == "__main__": main()
