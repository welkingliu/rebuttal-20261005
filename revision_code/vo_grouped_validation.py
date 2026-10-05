"""V-O: five-fold utility diagnostics on V-N training images only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources
from vo_math import (image_folds, protection_targets, choose_top, macro_delta,
                     policy_summary, ranking_metrics)


BASE = ROOT / "results/VO_grouped_train_validation"
CACHE = ROOT / "cache/VO_grouped_train_validation"
PRIOR = ROOT / "results/VN_train_proposal_router/train3000"
INPUT = ROOT / "cache/VN_train_proposal_router/train3000"
SOURCES = ["vo_grouped_validation.py", "vo_math.py", "VO_GROUPED_VALIDATION_PLAN.txt",
           "vn_train_proposals.py", "vl_router.py", "vl_policy.py", "vm_localization_router.py",
           "vl_sgdet_feasibility.py", "vb_native.py", "vc_protocol.py", "vk_gate_math.py"]


def progress(out, stage, start, count, total):
    elapsed = time.monotonic()-start
    row = dict(status="running", pid=os.getpid(), stage=stage, images=count, total=total,
               seconds=elapsed, eta_seconds=elapsed/count*(total-count) if count else None)
    atomic_json(out/"progress.json", row)
    print(json.dumps(row), flush=True)


def registration(out, smoke):
    from vn_train_proposals import verify
    prior, prior_sha = verify(PRIOR)
    ids = prior["training_ids"][:3] if smoke else prior["training_ids"]
    if set(ids) & (set(prior["evaluation_ids"]) | set(prior["excluded_expert_images"])):
        raise RuntimeError("Training image isolation failed")
    folds = image_folds(ids)
    protocol = dict(version="vo_image_grouped_training_diagnostics_v1", smoke=smoke,
        prior_protocol_sha256=prior_sha, image_ids=ids, folds=folds,
        data_role="Native VG TRAIN only; NO validation/test records accessed",
        native_model_exposure="Frozen SGG fitted VG train; auxiliary DINO classifier excluded these images",
        formal_gate_accepted=False, independent_confirmation=False, no_hyperparameter_search=True,
        no_threshold_search=True, no_automatic_expansion=True,
        policy_plan=Path(__file__).with_name("VO_GROUPED_VALIDATION_PLAN.txt").read_text(),
        sources=sources(SOURCES))
    return protocol, lock(out/"protocol.json", protocol)


def finish(out, value):
    atomic_json(out/"summary.json", value)
    atomic_json(out/"progress.json", dict(status="complete", images=value["images"], total=value["images"]))
    print(json.dumps(value), flush=True)


def replay_outcomes(out, cache, protocol, reg):
    from torch.nn import functional as F
    from vn_train_proposals import checked_load, make_post, replay, ground_truth
    from repro_experiment import save_torch
    from vb_native import OfficialMetrics, targets
    from vk_gate_protocol import CHECKPOINT
    from vk_gate_native import configure_backend
    from vl_sgdet_feasibility import all_candidates
    from vl_router import action_features, FEATURE_NAMES
    from vm_localization_router import localization_features
    configure_backend()
    stage = out/"replay"; start = time.monotonic()
    _, post = make_post(stage); metric = OfficialMetrics("sgdet")
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    rows = []; prior_sha = protocol["prior_protocol_sha256"]
    progress(stage, "train_single_action_replay", start, 0, len(protocol["image_ids"]))
    for n, iid in enumerate(protocol["image_ids"], 1):
        target_path, meta_path = cache/(iid+".pt"), stage/"images"/(iid+".json")
        paths = {name: INPUT/name/(iid+".pt") for name in ["raw", "features", "outcomes"]}
        hashes = {name: sha256(path) for name, path in paths.items()}
        if meta_path.exists() and target_path.exists():
            meta = read(meta_path)
            if meta["protocol_sha256"] != reg or meta["inputs"] != hashes or meta["cache_sha256"] != sha256(target_path):
                raise RuntimeError("V-O resume/input drift")
        else:
            raw, feature, old = [checked_load(paths[key], prior_sha) for key in ["raw", "features", "outcomes"]]
            export_meta = read(PRIOR/"export/images"/(iid+".json"))
            if hashes["raw"] != export_meta["raw_sha256"] or hashes["raw"] != feature["raw_sha256"]:
                raise RuntimeError("Raw/feature provenance mismatch")
            if not torch.equal(raw["crop_boxes"], feature["crop_boxes"]):
                raise RuntimeError("Feature alignment changed")
            gt = ground_truth(raw); y = targets(raw, gt, "sgdet")
            if not np.array_equal(y, raw["target"].numpy()): raise RuntimeError("Target alignment changed")
            with torch.no_grad():
                baseline = raw["native_logits"].cuda(); base = replay(post, raw, baseline)
                b = metric.row(iid, base, gt, dict(logits=baseline), y)
                bc = np.asarray(b["class_recalls"][4], dtype=float); support = np.isfinite(bc)
                candidate, eligible, _, legacy = all_candidates(feature["features"], raw["native_logits"], state)
                indices = eligible.nonzero(as_tuple=False).flatten().tolist()
                if indices != old["indices"]: raise RuntimeError("Candidate set changed")
                expert = F.linear(feature["features"], state["expert_head"]["weight"], state["expert_head"]["bias"]).softmax(-1)
                xx, gain, dr, dc = [], [], [], []
                for index in indices:
                    logits = baseline.clone(); logits[index] = candidate[index].cuda()
                    prediction = replay(post, raw, logits)
                    xx.append(action_features(index, legacy, raw["native_logits"], feature["features"], expert, state["centers"], base, prediction))
                    trial = metric.row(iid, prediction, gt, dict(logits=logits), y)
                    tc = np.asarray(trial["class_recalls"][4], dtype=float)
                    if not np.array_equal(np.isfinite(tc), support): raise RuntimeError("Predicate support changed")
                    if trial["positive_objects"] != b["positive_objects"]: raise RuntimeError("Identity denominator changed")
                    gain.append(trial["post_nms_correct"]-b["post_nms_correct"])
                    dr.append(trial["recalls"][4]-b["recalls"][4])
                    dc.append(np.nan_to_num(tc-bc))
                x = np.asarray(xx, dtype=float).reshape(-1, len(FEATURE_NAMES))
                np.testing.assert_allclose(x, old["x"].numpy(), rtol=1e-6, atol=1e-7)
                gain, dr, dc = np.asarray(gain), np.asarray(dr), np.asarray(dc).reshape(-1, 50)
                labels = protection_targets(gain, dr, dc, support)
                if not np.array_equal(labels["strict"], old["safe"].numpy()) or not np.array_equal(labels["effect"], old["effect"].numpy()):
                    raise RuntimeError("Sealed V-N label parity failed")
                local_x = localization_features(feature["features"], raw["native_logits"], raw["proposal_boxes"], raw["size"])
                record = dict(image_id=iid, protocol_sha256=reg, x=x, indices=np.asarray(indices, dtype=int),
                    gain=gain.astype(float), delta_r=dr, delta_class=dc, support=support, labels=labels,
                    localization_x=local_x.astype(np.float32), target=y, baseline=b)
                save_torch(target_path, record)
                for key in [1, 5, 10, 20, 50, 100]: metric.result["sgdet_recall"][key].clear()
            meta = dict(image_id=iid, protocol_sha256=reg, inputs=hashes, cache_sha256=sha256(target_path),
                actions=len(indices), positives=b["positive_objects"], strict=int(labels["strict"].sum()),
                image_guard=int(labels["image_guard"].sum()), identity_only=int(labels["identity_only"].sum()),
                parity="V-N candidate indices, feature vectors, strict/effective labels all match")
            atomic_json(meta_path, meta)
        rows.append(meta)
        if n % 25 == 0 or n == len(protocol["image_ids"]):
            progress(stage, "train_single_action_replay", start, n, len(protocol["image_ids"]))
    finish(stage, dict(status="complete", protocol_sha256=reg, images=len(rows),
        counts={k: sum(r[k] for r in rows) for k in ["actions", "positives", "strict", "image_guard", "identity_only"]},
        elapsed_seconds=time.monotonic()-start, validation_accessed=False, test_accessed=False))


def cross_validate(out, cache, protocol, reg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from vl_router import fit_logistic, predict_logistic
    from vm_localization_router import fit_localization
    from repro_experiment import save_torch
    if protocol["smoke"]: raise ValueError("Three-image smoke cannot estimate efficacy")
    if read(out/"replay/summary.json")["protocol_sha256"] != reg: raise RuntimeError("Missing replay")
    stage = out/"crossval"; start = time.monotonic(); ids = protocol["image_ids"]
    records = []
    for n, iid in enumerate(ids, 1):
        path = cache/(iid+".pt"); meta = read(out/"replay/images"/(iid+".json"))
        if sha256(path) != meta["cache_sha256"]: raise RuntimeError("Outcome cache changed")
        row = torch.load(str(path), map_location="cpu")
        if row["protocol_sha256"] != reg or row["image_id"] != iid: raise RuntimeError("Row provenance changed")
        records.append(row)
        if n % 500 == 0: progress(stage, "load_train_only_rows", start, n, len(ids))
    groups = np.concatenate([np.full(len(r["indices"]), i, dtype=int) for i, r in enumerate(records)])
    x, indices, gain, dr, dc = [np.concatenate([r[k] for r in records]) for k in ["x", "indices", "gain", "delta_r", "delta_class"]]
    labels = {k: np.concatenate([r["labels"][k] for r in records]) for k in records[0]["labels"]}
    support = np.stack([r["support"] for r in records])
    pos = np.asarray([r["baseline"]["positive_objects"] for r in records])
    folds = np.asarray([protocol["folds"][iid] for iid in ids]); action_folds = folds[groups]
    names = ["strict", "image_guard", "identity_only"]
    scores = {name: np.zeros(len(x)) for name in names+["ridge_gain", "hgb_gain"]}
    safety = {name: np.zeros(len(x)) for name in names}
    pm = np.zeros(len(x)); mean_prediction = np.zeros(len(x)); fold_audits = []
    for fold in range(5):
        fold_dir = stage/("fold%d" % fold)
        fit_images = np.flatnonzero(folds != fold); held_images = np.flatnonzero(folds == fold)
        fit, held = action_folds != fold, action_folds == fold
        if set(groups[fit]) & set(groups[held]): raise RuntimeError("Image leakage")
        progress(stage, "crossfit_fold%d_of5" % (fold+1), start, fold, 5)
        effect_model = fit_logistic(x[fit], labels["effect"][fit])
        pe = predict_logistic(effect_model, x[held]); models = {"effect": effect_model}
        for name in names:
            use = fit & labels["effect"]
            models[name] = fit_logistic(x[use], labels[name][use])
            ps = predict_logistic(models[name], x[held])
            safety[name][held] = ps; scores[name][held] = pe*(2*ps-1)
        local_x = np.concatenate([records[i]["localization_x"][records[i]["target"] >= 0] for i in fit_images]).astype(float)
        local_y = np.concatenate([records[i]["target"][records[i]["target"] >= 0] > 0 for i in fit_images]).astype(int)
        local_model = fit_localization(local_x, local_y); del local_x, local_y
        local_truth, local_pred = [], []
        for image in held_images:
            r = records[image]; probability = predict_logistic(local_model, r["localization_x"])
            pm[groups == image] = probability[r["indices"]]
            keep = r["target"] >= 0
            local_truth.extend((r["target"][keep] > 0).tolist()); local_pred.extend(probability[keep].tolist())
        regressors = {
            "ridge_gain": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "hgb_gain": HistGradientBoostingRegressor(max_iter=100, max_depth=3, max_leaf_nodes=7,
                min_samples_leaf=100, learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=17)}
        for name, model in regressors.items():
            model.fit(x[fit], gain[fit]); scores[name][held] = model.predict(x[held])
        mean_prediction[held] = gain[fit].mean()
        lock(fold_dir/"logistic_models.json", dict(protocol_sha256=reg, router=models, localization=local_model))
        audit = dict(fold=fold, fit_ids=[ids[i] for i in fit_images], held_ids=[ids[i] for i in held_images],
            fit_actions=int(fit.sum()), held_actions=int(held.sum()),
            fit_labels={k: int(v[fit].sum()) for k, v in labels.items()}, held_labels={k: int(v[held].sum()) for k, v in labels.items()},
            localization_auc=float(roc_auc_score(local_truth, local_pred)),
            localization_ap=float(average_precision_score(local_truth, local_pred)),
            mean_gain_predictor=float(gain[fit].mean()), scalers_fit_on_training_fold_only=True)
        atomic_json(fold_dir/"audit.json", audit); fold_audits.append(audit)
        save_torch(fold_dir/"regressors.pt", dict(protocol_sha256=reg, models=regressors))
    # Commit predictions before evaluating policy outcomes. No GT is passed to choose_top.
    save_torch(stage/"oof_predictions.pt", dict(protocol_sha256=reg, groups=groups, indices=indices,
        scores=scores, conditional_probabilities=safety, localization_probability=pm,
        fold_training_mean_prediction=mean_prediction, action_folds=action_folds))
    policies = {}; selections = {}
    for name in names+["ridge_gain", "hgb_gain"]:
        for use_loc in [False, True]:
            label = name+("_localization" if use_loc else "_no_localization")
            eligible = scores[name] > 0
            if name in safety: eligible &= safety[name] >= .6
            if use_loc: eligible &= pm >= .5
            priority = scores[name]*pm if use_loc else scores[name]
            selections[label] = choose_top(groups, indices, priority, eligible, len(ids))
    for name in names:
        selections["privileged_oracle_"+name] = choose_top(groups, indices, gain, labels[name], len(ids))
    for number, (name, chosen) in enumerate(selections.items(), 1):
        progress(stage, "policy_statistics:"+name, start, number-1, len(selections))
        summary = policy_summary(chosen, groups, gain, dr, dc, support, pos, folds)
        selected_rows = chosen[chosen >= 0]
        summary.update(oracle=name.startswith("privileged_oracle"),
            selected_actions_satisfying_strict=int(labels["strict"][selected_rows].sum()),
            selected_actions_satisfying_image_guard=int(labels["image_guard"][selected_rows].sum()))
        policies[name] = summary; atomic_json(stage/"policies"/(name+".json"), summary)
    rank = {name: ranking_metrics(gain, value, mean_prediction if name.endswith("_gain") else None) for name, value in scores.items()}
    for name, value in scores.items():
        rank[name]["by_fold"] = [dict(fold=k, **ranking_metrics(gain[action_folds == k], value[action_folds == k],
            mean_prediction[action_folds == k] if name.endswith("_gain") else None)) for k in range(5)]
        rank[name]["on_identity_effective_actions"] = ranking_metrics(gain[gain != 0], value[gain != 0])
    categories = dict(actions=len(x), identity_gains=int((gain > 0).sum()), identity_losses=int((gain < 0).sum()),
        identity_neutral=int((gain == 0).sum()), strict_safe=int(labels["strict"].sum()),
        image_guard_safe=int(labels["image_guard"].sum()),
        gains_rejected_only_by_per_predicate_rule=int((labels["image_guard"] & ~labels["strict"]).sum()),
        gains_rejected_by_image_guard=int((labels["identity_only"] & ~labels["image_guard"]).sum()))
    eligibility = {name: dict(conditional_ge_06=int((safety[name] >= .6).sum()),
        localization_ge_05=int((pm >= .5).sum()), both=int(((safety[name] >= .6) & (pm >= .5)).sum()),
        target_positives=int(labels[name].sum()), target_positives_passing_conditional=int((labels[name] & (safety[name] >= .6)).sum()),
        target_positives_passing_both=int((labels[name] & (safety[name] >= .6) & (pm >= .5)).sum())) for name in names}
    per_image = []
    for i, iid in enumerate(ids):
        per_image.append(dict(image_id=iid, fold=int(folds[i]), positive_objects=int(pos[i]),
            policies={name: (None if chosen[i] < 0 else dict(proposal=int(indices[chosen[i]]),
                gain=int(gain[chosen[i]]), R50_delta=float(dr[chosen[i]]))) for name, chosen in selections.items()}))
    atomic_json(stage/"per_image.json", dict(protocol_sha256=reg, records=per_image))
    baseline_correct = sum(r["baseline"]["post_nms_correct"] for r in records)
    baseline_r = np.mean([r["baseline"]["recalls"][4] for r in records])
    baseline_mr = macro_delta(np.stack([np.nan_to_num(np.asarray(r["baseline"]["class_recalls"][4], dtype=float)) for r in records]), support)
    finish(stage, dict(status="complete", protocol_sha256=reg, images=len(ids),
        baseline=dict(post_nms_object_top1=baseline_correct/int(pos.sum()), R50=float(baseline_r), mR50=float(baseline_mr)),
        counts=categories, eligibility=eligibility, predictability=rank, policies=policies,
        folds=[{k:v for k,v in row.items() if not k.endswith("_ids")} for row in fold_audits],
        native_model_fitted_on_these_training_images=True, validation_accessed=False, test_accessed=False,
        formal_gate_accepted=False, independent_confirmation=False, elapsed_seconds=time.monotonic()-start,
        conclusion_policy="Diagnostic results only; no automatic deployment, threshold search or independent-gate claim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["replay", "crossval"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); ensure_storage(); torch.set_num_threads(2)
    lane = "smoke" if args.smoke else "train3000"
    out, cache = BASE/lane, CACHE/lane
    guard = output_path(out/"execution.lock").open("w"); fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol, reg = registration(out, args.smoke)
    summary = out/args.stage/"summary.json"
    if summary.exists():
        if read(summary)["protocol_sha256"] != reg: raise RuntimeError("Completed result mismatch")
        print("Already complete: "+str(summary), flush=True); return
    if args.stage == "replay": replay_outcomes(out, cache, protocol, reg)
    else: cross_validate(out, cache, protocol, reg)


if __name__ == "__main__": main()
