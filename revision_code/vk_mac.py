"""Finite calibration-source experiment; cached MacCPU only, no native gate claims."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import vj_mac as base


SPEC = dict(version="vk_heldout_router_supervision_v1", seed=17,
    folds=5, fold_salt="VK_calibration_images_20261002:", calibration_images=500,
    training=copy.deepcopy({k: base.SPEC[k] for k in ["epochs", "learning_rate", "l2", "alpha", "repair_probability_min", "minimum_prototype_objects"]}),
    primary="VJ14 inputs; router fit on500 calibration images outside native gradient training; image5fold OOF",
    controls=["calibration confidence10 OOF", "frozen train-fitted VJ", "unselected fusion"],
    selection="none: fixed final200 epochs; no threshold/feature/primary substitution",
    screening=copy.deepcopy(base.SPEC["screening"]),
    gate=copy.deepcopy(base.SPEC["gate"]),
    limitation="500 calibration images reused from prior model exploration; OOF screening is NOT independent confirmation",
    requires_fresh_adapter_gate=True, server_gpu_used=False, local_cpu_threads=4,
    stop="screen fail terminates; pass awaits native SGCls/SGDet with same frozen weights, original joint criteria")


def fold_map(ids):
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate calibration image IDs")
    order = sorted(ids, key=lambda x: hashlib.sha256((SPEC["fold_salt"] + x).encode()).digest())
    return {iid: n % SPEC["folds"] for n, iid in enumerate(order)}


def informative_targets(p, candidate, labels):
    old = p.argmax(-1) + 1 == labels
    new = candidate[:, 1:].argmax(-1) + 1 == labels
    return old ^ new, new.float()


def router_fit(features, targets, informative, eligible, fit_mask, out, name, start):
    fit = informative & eligible & fit_mask
    y = targets[fit]
    if len(y) < 100 or min(int(y.sum()), int((1-y).sum())) < 20:
        raise RuntimeError("Insufficient calibration fold support")
    return base.fit_router(features[fit], y, out, name, start)


def old_distribution_diagnosis(bundle):
    result = {}
    for split in ["train", "development"]:
        d = bundle["data"][split]; head = bundle["selected"]["head"]
        q = bundle["oof"] if split == "train" else F.linear(d["features"], head["weight"], head["bias"]).softmax(-1)
        _, p, candidate = base.probabilities(d["baseline"], q)
        useful, target = informative_targets(p, candidate, d["labels"])
        _, selected, pred = base.route(d["baseline"], q, base.confidence_features(p, q),
            bundle["selected"]["router"], torch.ones(len(q), dtype=torch.bool))
        o = p.argmax(-1) + 1 == d["labels"]
        repair = useful & (target > 0); damage = useful & (target == 0)
        result[split] = dict(images=len(d["image_ids"]), objects=len(q), native_accuracy=float(o.double().mean()),
            expert_accuracy=float((q.argmax(-1)+1==d["labels"]).double().mean()), repairs=int(repair.sum()),
            damages=int(damage.sum()), conditional_repair_fraction=float(repair.sum().double()/useful.sum()),
            predicted_conditional_repair_mean=float(pred[useful].double().mean()), selected=int(selected.sum()),
            conditional_brier=float((pred[useful]-target[useful]).square().double().mean()))
    result["interpretation"] = "Posthoc diagnosis on known data; different class mix and auxiliary training size confound a causal attribution"
    return result


def calibration_quality(confidence, target, informative):
    p, y = confidence[informative].double(), target[informative].double()
    return dict(informative_objects=len(p), observed_repair_fraction=float(y.mean()),
        predicted_repair_fraction=float(p.mean()), brier=float((p-y).square().mean()),
        conditional_nll=float(F.binary_cross_entropy(p.clamp(1e-8,1-1e-8),y)))


def run(bundle_path, previous, out):
    start = time.monotonic()
    sidecar = json.loads(bundle_path.with_suffix(".json").read_text())
    if base.digest(bundle_path) != sidecar["bundle_sha256"]:
        raise RuntimeError("Transferred input hash mismatch")
    prior = json.loads((previous/"summary.json").read_text())
    prior_spec = json.loads((previous/"protocol.json").read_text())
    if base.digest(previous/"selected.pth") != prior["checkpoint_sha256"]:
        raise RuntimeError("VJ weights changed")
    if (base.digest(Path(base.__file__)) != prior_spec["code_sha256"]
            or base.digest(previous/"protocol.json") != prior["protocol_sha256"]):
        raise RuntimeError("Prior frozen implementation changed")
    b = torch.load(str(bundle_path), map_location="cpu", weights_only=True)
    dev, train = b["data"]["development"], b["data"]["train"]
    mapping = fold_map(dev["image_ids"])
    protocol = dict(spec=SPEC, bundle_sha256=sidecar["bundle_sha256"], provenance=sidecar,
        code_sha256=base.digest(Path(__file__)), helper_sha256=base.digest(Path(base.__file__)),
        plan_sha256=base.digest(Path(__file__).with_name("VK_MAC_PLAN.md")),
        previous_checkpoint_sha256=prior["checkpoint_sha256"], calibration_folds=mapping,
        torch_version=str(torch.__version__))
    base.seal(out/"protocol.json", protocol)
    if (out/"summary.json").exists() or (out/"selected.pth").exists():
        raise RuntimeError("Preserve existing fixed-run output")
    base.progress(out, "verify_inputs", 0, 5, start)
    _, audit = base.validate_bundle(b)
    base.write(out/"input_audit.json", dict(OOF_auxiliary_checks=audit,
        gate_data_loaded=False, router_calibration_ids_excluded_from_original_training=True))
    base.write(out/"mechanism_diagnosis.json", old_distribution_diagnosis(b))
    centers, counts = base.prototypes(train["features"], train["labels"])
    head = b["selected"]["head"]
    q = F.linear(dev["features"], head["weight"], head["bias"]).softmax(-1)
    native, p, candidate = base.probabilities(dev["baseline"], q)
    basic = base.confidence_features(p, q)
    vis, eligible = base.visual_features(dev["features"], p, candidate, centers, counts)
    features = torch.cat([basic, vis], -1)
    useful, target = informative_targets(p, candidate, dev["labels"])
    assigned = base.group_folds(dev, mapping)
    pred = native.clone(); control = native.clone()
    chosen = torch.zeros(len(q), dtype=torch.bool); control_chosen = chosen.clone()
    probabilities = torch.zeros(len(q)); control_probabilities = probabilities.clone()
    seen = torch.zeros(len(q), dtype=torch.bool); folds = []
    for f in range(5):
        hold = assigned == f; fit = ~hold
        a = router_fit(features, target, useful, eligible, fit, out, "fold%d_primary"%f, start)
        c = router_fit(basic, target, useful, eligible, fit, out, "fold%d_confidence_control"%f, start)
        base.save(out/("fold%d.pth"%f), dict(primary=a, confidence_control=c, protocol_sha256=base.digest(out/"protocol.json")))
        with torch.inference_mode():
            pred[hold], chosen[hold], probabilities[hold] = base.route(dev["baseline"][hold], q[hold], features[hold], a, eligible[hold])
            control[hold], control_chosen[hold], control_probabilities[hold] = base.route(dev["baseline"][hold], q[hold], basic[hold], c, eligible[hold])
        fit_ids = sorted(i for i in mapping if mapping[i] != f)
        hold_ids = sorted(i for i in mapping if mapping[i] == f)
        if set(fit_ids) & set(hold_ids) or (seen & hold).any():
            raise RuntimeError("Calibration fold leakage")
        seen |= hold
        folds.append(dict(fold=f, fit_image_ids=fit_ids, prediction_image_ids=hold_ids,
            fit_informative_objects=int((fit & useful & eligible).sum()), predicted_objects=int(hold.sum())))
    if not seen.all():
        raise RuntimeError("Incomplete fold predictions")
    # This full-pool artifact is frozen before any aggregate OOF decision is inspected.
    full = router_fit(features, target, useful, eligible, torch.ones_like(eligible), out, "full_primary", start)
    base.save(out/"selected.pth", dict(router=full, centers=centers, counts=counts,
        expert_head=head, protocol_sha256=base.digest(out/"protocol.json"),
        calibration_image_ids=dev["image_ids"], fold_map=mapping))
    checkpoint_sha = base.digest(out/"selected.pth")
    base.write(out/"frozen_before_screen.json", dict(checkpoint_sha256=checkpoint_sha,
        timestamp=datetime.now(timezone.utc).isoformat(), full_fit_not_evaluated_on_training_pool=True))
    base.write(out/"crossfit.json", dict(folds=folds, same_image_excluded_from_mean_scale_and_weights=True,
        prior_method_selection_not_cross_fitted=True, independent_confirmation=False))
    old = torch.load(str(previous/"selected.pth"), map_location="cpu", weights_only=True)
    with torch.inference_mode():
        reference, ref_selected, ref_confidence = base.route(dev["baseline"], q, features, old["router"], eligible)
    arms = dict(native=(native,torch.zeros_like(chosen)), primary_oof=(pred,chosen),
        confidence_control_oof=(control,control_chosen), frozen_vj_reference=(reference,ref_selected),
        fixed_fusion_control=(candidate,torch.ones_like(chosen)))
    rows = {name:base.records(dev,prob,mask) for name,(prob,mask) in arms.items()}
    results = {name:base.aggregate(value) for name,value in rows.items()}
    if results["frozen_vj_reference"] != prior["results"]["visual_verifier"]:
        raise RuntimeError("VJ replay drift")
    base.write(out/"calibration_oof_images.json", rows)
    base.write(out/"calibration_quality.json", dict(primary_oof=calibration_quality(probabilities,target,useful),
        confidence_control_oof=calibration_quality(control_probabilities,target,useful),
        frozen_vj=calibration_quality(ref_confidence,target,useful),
        scope="Conditional on informative utility; calibration statistics are not identity or SGG success"))
    base.save(out/"oof_predictions.pt", dict(probabilities=pred, selected=chosen, repair_probability=probabilities,
        image_ids=dev["image_ids"], offsets=dev["offsets"], protocol_sha256=base.digest(out/"protocol.json")))
    checks = base.screen_checks(results["primary_oof"], results["native"])
    passed = all(checks.values())
    summary = dict(status="awaiting_native_confirmation" if passed else "stopped_screen_rejected",
        screening_eligible=passed, checks=checks, results=results,
        full_checkpoint_sha256=checkpoint_sha, protocol_sha256=base.digest(out/"protocol.json"),
        heldout_confirmation_evaluated=False, SGCls_relation_gate_evaluated=False, SGDet_gate_evaluated=False,
        native_joint_gate_accepted=None, extra_seeds_run=False, test_evaluated=False,
        reuse_warning=SPEC["limitation"], cuda_used=False,
        next_step="Request reserved native two-task gate, not extra seeds/test" if passed else "Stop this candidate; no automatic threshold/candidate expansion",
        seconds=time.monotonic()-start)
    if base.digest(out/"selected.pth") != checkpoint_sha:
        raise RuntimeError("Frozen full-pool model changed")
    base.write(out/"summary.json",summary)
    base.progress(out,summary["status"],500,500,start,checks=checks,
                  identity_delta=results["primary_oof"]["delta"],native_gate_passed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(SPEC["local_cpu_threads"])
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args.bundle.resolve(),args.previous.resolve(),out)
        except Exception as exc:
            base.write(out/"failure.json",dict(status="failed",error=type(exc).__name__+": "+str(exc)))
            raise


if __name__=="__main__":
    main()
