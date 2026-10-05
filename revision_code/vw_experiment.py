"""V-W nested image fitting, output-route audits and four-scope identity reporting."""
import argparse
import fcntl
import time
import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import read, lock
from repro_experiment import save_torch
from vp_experiment import progress, finish
from vp_math import split_fold
from vq_math import choose_epoch
from vq_experiment import assert_baseline
from vv_experiment import records_for
from vu_experiment import bundle_for
from vw_math import SPEC, PRIMARY, TRAINED, ARMS, make_head, predict, fit, gradient_audit
from vw_protocol import OUT, WEIGHTS, load, register
from audit_identity_metric_history import paired_counts


def pair_protection(pred, gt):
    from vc_protocol import iou_pixel
    labels = pred.get_field("pred_labels").cpu().numpy()
    pairs = pred.get_field("rel_pair_idxs")[:100].cpu().numpy()
    probs = pred.get_field("pred_rel_scores")[:100].cpu()
    relation_labels = probs[:, 1:].argmax(-1).numpy() + 1
    relation_score = probs[:, 1:].max(-1).values.clamp_min(1e-30).log()
    g = gt.resize(pred.size)
    truth = g.get_field("labels").numpy()
    rels = g.get_field("relation_tuple").numpy()
    valid = (iou_pixel(pred.bbox.cpu().numpy(), g.bbox.numpy()) >= .5) & (labels[:, None] == truth[None, :])
    protected = np.zeros(len(pairs), dtype=bool)
    for j, (s, o) in enumerate(pairs[:50]):
        protected[j] = np.any(valid[s, rels[:, 0]] & valid[o, rels[:, 1]] & (relation_labels[j] == rels[:, 2]))
    endpoint = np.zeros(len(labels), dtype=bool)
    endpoint[pairs[:50].reshape(-1)] = True
    return dict(pairs=torch.from_numpy(pairs).long(), emitted_labels=torch.from_numpy(labels).long(),
                predicate_logprob=relation_score, protected=torch.from_numpy(protected), endpoint=torch.from_numpy(endpoint))


def augment(records, post, metric, stage):
    from vn_train_proposals import replay
    from vc_protocol import iou_pixel, KS
    from r19_proposal_sensitivity import unique_support
    start = time.monotonic()
    compact = {}
    with torch.no_grad():
        for i, rec in enumerate(records, 1):
            raw = rec["raw"]
            native = raw["native_logits"].float()
            pred = replay(post, raw, native.cuda())
            row = metric.row(rec["iid"], pred, rec["gt"], dict(logits=native), rec["target"])
            assert_baseline(row, rec["baseline"])
            meta = pair_protection(pred, rec["gt"])
            gt = rec["gt"].resize(raw["size"])
            p, g = unique_support(iou_pixel(raw["proposal_boxes"].numpy(), gt.bbox.numpy()))
            rec["unique_proposal"] = p
            rec["unique_labels"] = gt.get_field("labels").numpy()[g]
            rec["gt_objects"] = len(gt)
            rec["native_endpoint"] = meta["endpoint"].numpy()
            compact[rec["iid"]] = dict(meta, visual=rec["visual"].float(), native=native,
                                       target=torch.from_numpy(rec["target"]).long())
            for k in KS:
                metric.result["sgdet_recall"][k].clear()
            if i % 100 == 0 or i == len(records):
                progress(stage, "verify_native_and_prepare_fixed_scopes", i, len(records), start)
    return compact


def calibration_bins(confidence, correct):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct)
    index = np.minimum((confidence * 15).astype(int), 14)
    return np.stack([np.bincount(index, minlength=15),
                     np.bincount(index, weights=confidence, minlength=15),
                     np.bincount(index, weights=correct, minlength=15)], axis=1).tolist()


def scopes(rec, logits, prediction):
    y = rec["target"]
    positive = y > 0
    pre = logits[:, 1:].argmax(-1).cpu().numpy() + 1
    post = prediction.get_field("pred_labels").cpu().numpy()
    if len(pre) != len(y) or len(post) != len(y):
        raise RuntimeError("Fixed proposal support changed")
    result = {}
    for name, valid in [("pre_nms", positive), ("post_nms", positive),
                        ("native_top50_endpoint", positive & rec["native_endpoint"]),
                        ("not_native_top50_endpoint", positive & ~rec["native_endpoint"])]:
        label = pre if name == "pre_nms" else post
        result[name] = dict(objects=int(valid.sum()), correct=int((label[valid] == y[valid]).sum()))
    index, target = rec["unique_proposal"], rec["unique_labels"]
    for name, label in [("unique_pre", pre), ("unique_post", post)]:
        result[name] = dict(objects=len(index), correct=int((label[index] == target).sum()))
    conditional = logits[:, 1:].softmax(-1).max(-1).values.cpu().numpy()
    result.update(gt_objects=rec["gt_objects"], unique_matched=len(index),
        fg_conditional_bins=calibration_bins(conditional[positive], pre[positive] == y[positive]),
        emitted_bins=calibration_bins(prediction.get_field("pred_scores").cpu().numpy()[positive], post[positive] == y[positive]))
    return result


def scope_summary(rows):
    output = {}
    for name in ["pre_nms", "post_nms", "unique_pre", "unique_post", "native_top50_endpoint", "not_native_top50_endpoint"]:
        n, c = sum(r[name]["objects"] for r in rows), sum(r[name]["correct"] for r in rows)
        output[name] = dict(objects=n, correct=c, accuracy=c/n if n else None)
    for key in ["fg_conditional_bins", "emitted_bins"]:
        bins = np.asarray([r[key] for r in rows]).sum(0)
        n = bins[:, 0].sum()
        output[key.replace("_bins", "_ece")] = float(np.abs(bins[:, 1]-bins[:, 2]).sum()/n) if n else None
    output["gt_objects"] = sum(r["gt_objects"] for r in rows)
    output["unique_matched"] = sum(r["unique_matched"] for r in rows)
    return output


def score(records, head, post, metric, stage, label, no_op=False):
    from vn_train_proposals import replay
    from vc_protocol import KS
    rows, extra = [], []
    start = time.monotonic()
    if head is not None:
        head.eval()
    with torch.no_grad():
        for i, rec in enumerate(records, 1):
            native = rec["raw"]["native_logits"].float().cuda()
            updated = native if head is None else predict(head, rec["visual"].cuda(), native)
            if no_op and not torch.equal(updated, native):
                raise RuntimeError("Nonexact no-op logits")
            if not torch.isfinite(updated).all():
                raise RuntimeError("Nonfinite predictions")
            prediction = replay(post, rec["raw"], updated)
            row = metric.row(rec["iid"], prediction, rec["gt"], dict(logits=updated), rec["target"])
            if no_op:
                assert_baseline(row, rec["baseline"])
            rows.append(row)
            extra.append(scopes(rec, updated, prediction))
            for k in KS:
                metric.result["sgdet_recall"][k].clear()
            if i % 100 == 0 or i == len(records):
                progress(stage, label, i, len(records), start)
    return rows, extra


def route_audit(records, compact, split, post):
    from vn_train_proposals import replay
    from vb_native import compare_predictions
    eligible = [r for r in records if r["iid"] in set(split["fit"]) and compact[r["iid"]]["protected"].any()]
    if not eligible:
        raise RuntimeError("No fit-image protected triplets for gradient audit")
    rec = eligible[0]
    report = gradient_audit(compact[rec["iid"]], "cuda")
    native = rec["raw"]["native_logits"].float().cuda()
    head = make_head("cuda")
    with torch.no_grad():
        base = replay(post, rec["raw"], native)
        zero = replay(post, rec["raw"], predict(head, rec["visual"].cuda(), native))
        report["no_op_prediction_max_error"] = compare_predictions(base, zero)
        # Deliberately large head update demonstrates the actual output route.
        chosen = (int(native[0, 1:].argmax()) + 20) % 150 + 1
        head.bias[chosen] = 20.
        changed = replay(post, rec["raw"], predict(head, rec["visual"].cuda(), native))
        report["nonzero_output_score_change"] = float((base.get_field("pred_scores") - changed.get_field("pred_scores")).abs().max())
        report["nonzero_output_label_changes"] = int((base.get_field("pred_labels") != changed.get_field("pred_labels")).sum())
    if report["nonzero_output_score_change"] <= 1e-6 and report["nonzero_output_label_changes"] == 0:
        raise RuntimeError("Updated head cannot reach emitted output")
    report.update(image_id=rec["iid"], fit_image_only=True, all_checks_passed=True)
    return report


def train(protocol, reg, lane, fold, records, compact, split, post, metric):
    from vc_protocol import summarize_rows
    stage = OUT / lane / ("fold%d" % fold)
    path = WEIGHTS / lane / ("fold%d.pth" % fold)
    summary_path = stage / "training_summary.json"
    if summary_path.exists():
        info = read(summary_path)
        if info["protocol_sha256"] != reg or info["checkpoint_sha256"] != sha256(path):
            raise RuntimeError("Training resume drift")
        return torch.load(str(path), map_location="cpu"), info
    audit = route_audit(records, compact, split, post)
    atomic_json(stage / "route_gradient_audit.json", dict(protocol_sha256=reg, **audit))
    mapping = {r["iid"]: r for r in records}
    validation = [mapping[i] for i in split["inner_validation"]]
    states, histories, epochs, changes = {}, {}, {}, {}
    for name in TRAINED:
        candidates = []
        start = time.monotonic()
        limit = 2 if protocol["smoke"] else max(SPEC["epochs"])

        def callback(head, row):
            if row["epoch"] in SPEC["epochs"]:
                values, extra = score(validation, head, post, metric, stage,
                    "inner_%s_epoch%d" % (name, row["epoch"]), no_op=row["epoch"] == 0)
                m = summarize_rows(values)
                candidates.append(dict(epoch=row["epoch"], object=m["post_nms_object_top1"],
                    R50=m["R"]["50"], mR50=m["mR"]["50"], metrics=m, scopes=scope_summary(extra)))
                atomic_json(stage / ("candidate_progress_"+name+".json"), dict(protocol_sha256=reg, candidates=candidates))
            progress(stage, "fit_inner_"+name, row["epoch"], limit, start, detail=row)

        state, history = fit([compact[i] for i in split["fit"]], name, limit, callback)
        changes[name] = max(float(v.abs().max()) for v in state.values())
        if changes[name] <= 0:
            raise RuntimeError("Fit did not update head")
        selected = choose_epoch(candidates)
        epochs[name] = selected["selected_epoch"]
        selection = dict(protocol_sha256=reg, history=history, candidates=candidates, **selected)
        atomic_json(stage / ("selection_"+name+".json"), selection)
        refit_start = time.monotonic()
        states[name], refit_history = fit([compact[i] for i in split["training"]], name, epochs[name],
            callback=lambda head, row: progress(stage, "refit_"+name, row["epoch"], epochs[name], refit_start, detail=row))
        histories[name] = dict(inner=history, refit=refit_history)
    payload = dict(protocol_sha256=reg, bundle_sha256=protocol["bundle_sha256"], states=states, epochs=epochs,
                   fit_ids=split["training"], held_ids=split["held"], fold=fold)
    save_torch(path, payload)
    info = dict(protocol_sha256=reg, checkpoint_sha256=sha256(path), epochs=epochs, histories=histories,
                selection_parameter_changes=changes, route_gradient_audit=audit,
                fit_images=len(split["training"]), held_images=len(split["held"]))
    atomic_json(summary_path, info)
    return payload, info


def evaluate(protocol, reg, lane, fold, records, split, payload, info, post, metric):
    from vc_protocol import summarize_rows
    stage = OUT / lane / ("fold%d" % fold)
    mapping = {r["iid"]: r for r in records}
    held = [mapping[i] for i in split["held"]]
    arms, extra = {}, {}
    start = time.monotonic()
    for name in ARMS:
        head = None if name == "native" else make_head("cuda")
        if head is not None:
            head.load_state_dict(payload["states"][name], strict=True)
        arms[name], extra[name] = score(held, head, post, metric, stage, "outer_"+name,
                                      no_op=name == "native" or payload["epochs"].get(name) == 0)
    for i, iid in enumerate(split["held"]):
        atomic_json(stage / "images" / (iid+".json"), dict(protocol_sha256=reg, fold=fold,
            checkpoint_sha256=info["checkpoint_sha256"], image_id=iid,
            metrics={k: arms[k][i] for k in ARMS}, scopes={k: extra[k][i] for k in ARMS}))
    finish(stage, dict(status="complete", protocol_sha256=reg, checkpoint_sha256=info["checkpoint_sha256"],
        fold=fold, images=len(held), primary=PRIMARY, epochs=payload["epochs"],
        arms={k: summarize_rows(v) for k, v in arms.items()}, scopes={k: scope_summary(v) for k, v in extra.items()},
        route_gradient_checks_passed=info["route_gradient_audit"]["all_checks_passed"],
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False, independent_confirmation=False,
        elapsed_seconds=time.monotonic()-start))


def summarize(protocol, reg, lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]:
        raise ValueError("Smoke is not efficacy evidence")
    rows, choices = [], []
    for fold in range(5):
        stage = OUT / lane / ("fold%d" % fold)
        done = read(stage / "summary.json")
        digest = sha256(WEIGHTS / lane / ("fold%d.pth" % fold))
        if done["protocol_sha256"] != reg or done["checkpoint_sha256"] != digest or not done["route_gradient_checks_passed"]:
            raise RuntimeError("Invalid completed fold")
        choices.append(dict(fold=fold, epochs=done["epochs"]))
        for iid in read(stage / "split.json")["held"]:
            row = read(stage / "images" / (iid+".json"))
            if row["protocol_sha256"] != reg or row["checkpoint_sha256"] != digest or row["image_id"] != iid or row["fold"] != fold:
                raise RuntimeError("Invalid per-image provenance")
            rows.append(row)
    mapping = {r["image_id"]: r for r in rows}
    if len(rows) != 3000 or set(mapping) != set(protocol["image_ids"]):
        raise RuntimeError("Incomplete or duplicate outer predictions")
    rows = [mapping[i] for i in protocol["image_ids"]]
    folds = np.array([r["fold"] for r in rows])
    comparisons = {}
    scope_comparisons = {}
    for reference, arm in [("native", a) for a in TRAINED] + [(a, PRIMARY) for a in TRAINED if a != PRIMARY]:
        base, changed = [[r["metrics"][k] for r in rows] for k in [reference, arm]]
        pos = np.array([r["positive_objects"] for r in base])
        support = np.array([r["class_recalls"][4] for r in base], dtype=float)
        other = np.array([r["class_recalls"][4] for r in changed], dtype=float)
        if pos.tolist() != [r["positive_objects"] for r in changed] or not np.array_equal(np.isfinite(support), np.isfinite(other)):
            raise RuntimeError("Unpaired supports")
        gain = np.array([c["post_nms_correct"]-b["post_nms_correct"] for b, c in zip(base, changed)])
        delta_r = np.array([c["recalls"][4]-b["recalls"][4] for b, c in zip(base, changed)])
        name = arm+"_versus_"+reference
        stats = policy_summary(np.arange(3000), np.arange(3000), gain, delta_r, np.nan_to_num(other-support), np.isfinite(support), pos, folds)
        stats["evaluated_images"] = stats.pop("selected_images")
        comparisons[name] = stats
        scope_comparisons[name] = {}
        for scope in ["pre_nms", "post_nms", "unique_pre", "unique_post", "native_top50_endpoint", "not_native_top50_endpoint"]:
            paired = [[dict(image_id=r["image_id"], **r["scopes"][k][scope]) for r in rows] for k in [reference, arm]]
            scope_comparisons[name][scope] = paired_counts(paired[0], paired[1], "objects", "correct", draws=2000)
    checks = comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(OUT / lane, dict(status="complete", protocol_sha256=reg, images=3000, primary=PRIMARY, choices=choices,
        arms={k: summarize_rows([r["metrics"][k] for r in rows]) for k in ARMS},
        scopes={k: scope_summary([r["scopes"][k] for r in rows]) for k in ARMS},
        comparisons=comparisons, scope_comparisons=scope_comparisons,
        primary_training_screen_satisfied=all(checks.values()), formal_gate_accepted=False, independent_confirmation=False,
        native_sgg_trained_on_these_images=True, reused_train_images=True, validation_accessed=False, test_accessed=False,
        next_step="Stop this pilot; if primary passes, freeze it and audit confirmation exposure before any new registration. No automatic gate, seeds or control promotion."))


def main():
    from vk_gate_native import configure_backend
    from vn_train_proposals import make_post
    from vb_native import OfficialMetrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["register", "fold", "summarize"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(2)
    configure_backend()
    if args.stage == "register":
        _, digest, lane = register(args.smoke)
        print("[registered]", lane, digest, flush=True)
        return
    protocol, reg, lane = load(args.smoke)
    name = "fold%d" % args.fold if args.fold is not None else args.stage
    stage = OUT / lane / name
    guard = output_path(OUT / lane / (name+".lock")).open("w")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == "summarize":
        summarize(protocol, reg, lane)
        return
    if args.fold is None or (args.smoke and args.fold != 0):
        raise ValueError("Invalid fold")
    if (stage / "summary.json").exists():
        done = read(stage / "summary.json")
        if done["protocol_sha256"] != reg or done["checkpoint_sha256"] != sha256(WEIGHTS / lane / (name+".pth")):
            raise RuntimeError("Completed fold drift")
        return
    progress(stage, "load_verified_visual_bundle", 0, 1, time.monotonic())
    bundle, digest = bundle_for(protocol["vu_protocol_sha256"], lane)
    if digest != protocol["bundle_sha256"] or bundle["image_ids"] != protocol["image_ids"]:
        raise RuntimeError("Bundle changed")
    split = split_fold(protocol["image_ids"], protocol["folds"], args.fold)
    lock(stage / "split.json", dict(protocol_sha256=reg, **split))
    # Record all dependencies with absolute paths before native setup changes cwd.
    _, post = make_post(stage / "runtime")
    metric = OfficialMetrics("sgdet")
    records = records_for(protocol, bundle, protocol["image_ids"])
    compact = augment(records, post, metric, stage)
    del bundle
    payload, info = train(protocol, reg, lane, args.fold, records, compact, split, post, metric)
    evaluate(protocol, reg, lane, args.fold, records, split, payload, info, post, metric)


if __name__ == "__main__":
    main()
