"""V-T nested training-only kernel-scale experiment using audited SGDet replay."""
import argparse
import fcntl
from pathlib import Path
import time

import torch
from common import ROOT, ensure_storage, output_path, atomic_json, sha256
from evidence_completion import read, sources, lock
from repro_experiment import save_torch
from vp_experiment import rows_for, progress, finish
from vp_math import split_fold
from vq_experiment import registration as parent_registration, load_bundle, image_inputs, score_images
from repair_panel_experiment import SOURCE_NAMES as PARENT_SOURCES
from repair_panel_math import STRENGTHS
from vq_math import choose_epoch
from vt_kernel_math import fit_models, predict_logits

NAME = "VT_kernel_scale"
PRIMARY, CONTROL = "median_rms_kernel", "median_kernel"
OUT = ROOT/"results"/NAME
SOURCE_NAMES = sorted(set(PARENT_SOURCES+["vt_experiment.py", "vt_kernel_math.py", "vt_queue.py", "VT_SCALE_PLAN.txt"]))


def registration(smoke):
    prior, digest, lane = parent_registration(smoke)
    protocol = dict(version="vt_fit_only_kernel_scale_v1", branch="VT", smoke=smoke,
        image_ids=prior["image_ids"], folds=prior["folds"], vq_protocol_sha256=digest,
        vn_protocol_sha256=prior["vn_protocol_sha256"], sources=sources(SOURCE_NAMES),
        arms=["native", PRIMARY, CONTROL], primary=PRIMARY, control=CONTROL,
        strengths=STRENGTHS, plan=Path(__file__).with_name("VT_SCALE_PLAN.txt").read_text(),
        native_checkpoint_sha256=prior["native_checkpoint_sha256"],
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False,
        independent_confirmation=False, designed_after_panel_results=True)
    return protocol, lock(OUT/lane/"protocol.json", protocol), lane


def fit_subset(bundle, ids):
    ix = rows_for(bundle, ids)
    return fit_models(*[bundle[k][ix] for k in ["features", "native_logits", "labels"]])


def run_fold(protocol, reg, lane, fold):
    from vn_train_proposals import make_post
    from vb_native import OfficialMetrics
    from vc_protocol import summarize_rows
    stage = OUT/lane/("fold%d" % fold); start = time.monotonic()
    weight = ROOT/"checkpoints"/NAME/lane/("fold%d.pth" % fold)
    if (stage/"summary.json").exists():
        info = read(stage/"summary.json")
        if info["protocol_sha256"] != reg or info["checkpoint_sha256"] != sha256(weight):
            raise RuntimeError("Completed fold changed")
        return
    bundle, digest = load_bundle(protocol["vq_protocol_sha256"], lane)
    split = split_fold(protocol["image_ids"], protocol["folds"], fold)
    lock(stage/"split.json", dict(protocol_sha256=reg, **split))
    _, post = make_post(stage/"runtime"); metric = OfficialMetrics("sgdet")
    info_path = stage/"training_summary.json"
    if info_path.exists():
        info = read(info_path)
        if info["protocol_sha256"] != reg or info["bundle_sha256"] != digest or info["checkpoint_sha256"] != sha256(weight):
            raise RuntimeError("Fitted checkpoint drift")
        payload = torch.load(str(weight), map_location="cpu")
    else:
        progress(stage, "fit_inner_VT", 0, len(split["fit"]), start)
        models = fit_subset(bundle, split["fit"])
        inner_diagnostics = {k:v["diagnostics"] for k,v in models.items()}
        inner = image_inputs(protocol, bundle, split["inner_validation"]); choices = {}
        for name in [PRIMARY, CONTROL]:
            candidates = []
            for index, strength in enumerate(STRENGTHS):
                rows = score_images(inner, lambda x,n: predict_logits(x,n,models[name],strength), post,
                    metric, stage, "inner_%s_strength%.2f" % (name,strength), native=strength == 0)
                agg = summarize_rows(rows)
                candidates.append(dict(epoch=index, strength=strength, object=agg["post_nms_object_top1"],
                    R50=agg["R"]["50"], mR50=agg["mR"]["50"], metrics=agg))
            selection = choose_epoch(candidates); index = selection.pop("selected_epoch")
            choices[name] = dict(strength=STRENGTHS[index], candidates=candidates, **selection)
            atomic_json(stage/("selection_"+name+".json"), dict(protocol_sha256=reg, **choices[name]))
        del models
        progress(stage, "refit_outer_training_VT", 0, len(split["training"]), start)
        models = fit_subset(bundle, split["training"])
        payload = dict(protocol_sha256=reg, bundle_sha256=digest, states=models, choices=choices,
            fit_ids=split["training"], held_ids=split["held"], fold=fold)
        save_torch(weight, payload)
        info = dict(protocol_sha256=reg, bundle_sha256=digest, checkpoint_sha256=sha256(weight),
            choices=choices, fitting_images=len(split["training"]), held_images=len(split["held"]),
            inner_diagnostics=inner_diagnostics, refit_diagnostics={k:v["diagnostics"] for k,v in models.items()})
        atomic_json(info_path, info)
    outer = image_inputs(protocol, bundle, split["held"]); arms = {}
    for name in protocol["arms"]:
        strength = payload["choices"][name]["strength"] if name != "native" else 0
        transform = (lambda x,n:n) if name == "native" else (lambda x,n:predict_logits(x,n,payload["states"][name],strength))
        arms[name] = score_images(outer, transform, post, metric, stage, "outer_"+name, native=strength == 0)
    for i,iid in enumerate(split["held"]):
        atomic_json(stage/"images"/(iid+".json"), dict(protocol_sha256=reg, image_id=iid, fold=fold,
            checkpoint_sha256=info["checkpoint_sha256"], metrics={k:arms[k][i] for k in protocol["arms"]}))
    finish(stage, dict(status="complete", protocol_sha256=reg, checkpoint_sha256=info["checkpoint_sha256"],
        fold=fold, images=len(outer), primary=PRIMARY,
        selected_strengths={k:v["strength"] for k,v in payload["choices"].items()},
        arms={k:summarize_rows(v) for k,v in arms.items()}, validation_accessed=False, test_accessed=False,
        formal_gate_accepted=False, independent_confirmation=False, elapsed_seconds=time.monotonic()-start))


def main():
    from vk_gate_native import configure_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fold", "summarize"], required=True)
    parser.add_argument("--fold", choices=range(5), type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); ensure_storage(); torch.set_num_threads(2); configure_backend()
    protocol, reg, lane = registration(args.smoke)
    name = "fold%d" % args.fold if args.fold is not None else args.stage
    guard = output_path(OUT/lane/(name+".lock")).open("w")
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage == "fold":
        if args.fold is None or (args.smoke and args.fold != 0): raise ValueError("Invalid fold")
        run_fold(protocol, reg, lane, args.fold)
    else:
        # Reuse the unchanged paired-image evaluator with an additional result location.
        from repair_panel_experiment import BRANCHES, summarize
        BRANCHES["VT"] = (NAME, PRIMARY, CONTROL)
        summarize(protocol, reg, lane, OUT/lane)


if __name__ == "__main__": main()
