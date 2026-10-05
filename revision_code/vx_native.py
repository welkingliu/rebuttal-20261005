"""V-X original-runtime postprocessing, geometry audit and unchanged metrics."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from common import ensure_storage, sha256, atomic_json
from evidence_completion import read
from vx_common import OUT, CACHE, PRIMARY, ARMS, load, register, save, progress, finish, checked_done
from vx_math import balanced_weights, paired_box_iou


def prepare(p, reg, lane):
    from vn_train_proposals import make_post, replay, ground_truth
    from vb_native import OfficialMetrics, targets, compare_predictions
    from vc_protocol import iou_pixel, KS
    from vw_experiment import pair_protection
    from vq_experiment import assert_baseline
    from vp_experiment import VO, VO_CACHE
    from r19_proposal_sensitivity import unique_support
    stage = OUT/lane/"prepare"
    if checked_done(stage, reg):
        return
    start = time.monotonic()
    _, post = make_post(stage/"runtime")
    metric = OfficialMetrics("sgdet")
    manifest, audit, all_iou = {}, {}, []
    route = None
    for number, entry in enumerate(p["inputs"], 1):
        iid = entry["image_id"]
        if sha256(entry["raw"]) != entry["raw_sha256"]:
            raise RuntimeError("Raw cache changed")
        raw = torch.load(entry["raw"], map_location="cpu")
        if raw["protocol_sha256"] != p["vn_protocol_sha256"] or raw["image_id"] != iid:
            raise RuntimeError("Input registration changed")
        gt = ground_truth(raw)
        target = targets(raw, gt, "sgdet")
        if not np.array_equal(target, raw["target"].numpy()):
            raise RuntimeError("Original identity denominator changed")
        with torch.no_grad():
            native = raw["native_logits"].float()
            pred = replay(post, raw, native.cuda())
            row = metric.row(iid, pred, gt, dict(logits=native), target)
            meta_ref = read(VO/"replay/images"/(iid+".json"))
            prior = VO_CACHE/(iid+".pt")
            if sha256(prior) != meta_ref["cache_sha256"]:
                raise RuntimeError("Prior baseline changed")
            assert_baseline(row, torch.load(str(prior), map_location="cpu")["baseline"])
            if route is None:
                zero = replay(post, raw, native.cuda()+torch.zeros_like(native.cuda()))
                error = compare_predictions(pred, zero)
                delta = native.cuda().clone()
                changed_class = (int(native[0, 1:].argmax())+20) % 150+1
                delta[:, changed_class] += 20.
                changed = replay(post, raw, delta)
                score_diff = float((changed.get_field("pred_scores")-pred.get_field("pred_scores")).abs().max())
                label_diff = int((changed.get_field("pred_labels") != pred.get_field("pred_labels")).sum())
                if score_diff <= 1e-6 and label_diff == 0:
                    raise RuntimeError("Updated logits did not reach final outputs")
                route = dict(zero_prediction_max_error=error, synthetic_label_changes=label_diff,
                             synthetic_score_max_change=score_diff)
            resized = gt.resize(raw["size"])
            overlap = iou_pixel(raw["proposal_boxes"].numpy(), resized.bbox.numpy())
            assigned = overlap.argmax(-1)
            if not np.array_equal(resized.get_field("labels").numpy()[assigned[target>0]], target[target>0]):
                raise RuntimeError("Positive target matching differs")
            gt_endpoint = np.zeros(len(gt), dtype=bool)
            gt_endpoint[gt.get_field("relation_tuple")[:, :2].reshape(-1).numpy()] = True
            unique, indices = unique_support(overlap)
            meta = pair_protection(pred, gt)
            meta.update(native=native, target=torch.from_numpy(target).long(),
                        object_weight=balanced_weights(target, assigned, gt_endpoint),
                        diagnostic_weight=balanced_weights(target, assigned, gt_endpoint, True),
                        assignment=torch.from_numpy(assigned),
                        unique_proposal=torch.from_numpy(unique),
                        unique_labels=resized.get_field("labels")[indices],
                        gt_objects=len(gt), image_id=iid, protocol_sha256=reg,
                        raw_sha256=entry["raw_sha256"])
            path = CACHE/lane/"meta"/(iid+".pt")
            save(path, meta)
            manifest[iid] = dict(meta_sha256=sha256(path), baseline=row)
            n = len(target)
            if pred.bbox.shape != raw["proposal_boxes"].shape:
                raise RuntimeError("Native output proposal order/count changed")
            sx, sy = raw["image_size"][0]/raw["size"][0], raw["image_size"][1]/raw["size"][1]
            expected_crop = raw["proposal_boxes"]*torch.tensor([sx, sy, sx, sy])
            if not torch.allclose(expected_crop, raw["crop_boxes"], atol=1e-4, rtol=1e-5):
                raise RuntimeError("Crop coordinates disagree with actual input proposals")
            quality = paired_box_iou(raw["proposal_boxes"].numpy(), pred.bbox.cpu().numpy())
            before = native[:, 1:].argmax(-1).numpy()+1
            after = pred.get_field("pred_labels").cpu().numpy()
            groups = {"all":np.ones(n, bool), "positive":target>0,
                      "native_top50_positive":(target>0)&meta["endpoint"].numpy(),
                      "pre_correct_post_wrong":(target>0)&(before==target)&(after!=target),
                      "post_correct":(target>0)&(after==target),
                      "post_wrong":(target>0)&(after!=target)}
            for name, keep in groups.items():
                audit.setdefault(name, []).extend(quality[keep].tolist())
            all_iou.append(dict(image_id=iid, paired_iou=quality.tolist(), groups={k:int(v.sum()) for k,v in groups.items()}))
            for k in KS:
                metric.result["sgdet_recall"][k].clear()
        if number % 50 == 0 or number == len(p["inputs"]):
            progress(stage, "native_parity_box_alignment_and_instance_weights", number, len(p["inputs"]), start)
    stats = {k:dict(proposals=len(v), mean=float(np.mean(v)),
                    quantiles=np.quantile(v, [0, .1, .5, .9, 1]).tolist(),
                    below_05=int((np.asarray(v)<.5).sum()), below_085=int((np.asarray(v)<.85).sum()))
             for k,v in audit.items() if v}
    atomic_json(stage/"geometry_per_image.json", dict(protocol_sha256=reg, rows=all_iou))
    finish(stage, reg, len(manifest), manifest=manifest, geometry=stats, route_audit=route,
           interpretation="Prediction-box association diagnostic, not proof of a bug or causal mediation")


def evaluate(p, reg, lane, prediction_file):
    from vn_train_proposals import make_post, replay, ground_truth
    from vb_native import OfficialMetrics
    from vc_protocol import KS, summarize_rows
    from vq_experiment import assert_baseline
    from vw_experiment import scopes, scope_summary
    payload = torch.load(str(prediction_file), map_location="cpu")
    role, arm, epoch = payload["role"], payload["arm"], payload["epoch"]
    if role not in ["inner_validation", "held"] or arm not in ARMS[1:]:
        raise ValueError("Unsupported evaluation")
    if payload["protocol_sha256"] != reg or payload["image_ids"] != p["split"][role]:
        raise RuntimeError("Prediction/split contract mismatch")
    stage = OUT/lane/arm/("%s_epoch%d" % (role, epoch))
    if checked_done(stage, reg):
        if read(stage/"summary.json")["prediction_sha256"] != sha256(prediction_file):
            raise RuntimeError("Resumed predictions changed")
        return
    if role == "held":
        for fitted_arm in ARMS[1:]:
            if not checked_done(OUT/lane/fitted_arm/"train", reg):
                raise RuntimeError("All selections must be locked before held evaluation")
    _, post = make_post(stage/"runtime")
    metric = OfficialMetrics("sgdet")
    prepared = read(OUT/lane/"prepare/summary.json")
    inputs = {v["image_id"]:v for v in p["inputs"]}
    rows, extras = [], []
    start = time.monotonic()
    for n, iid in enumerate(payload["image_ids"], 1):
        entry = inputs[iid]
        if sha256(entry["raw"]) != entry["raw_sha256"]:
            raise RuntimeError("Raw drift")
        raw = torch.load(entry["raw"], map_location="cpu")
        meta_path = CACHE/lane/"meta"/(iid+".pt")
        if sha256(meta_path) != prepared["manifest"][iid]["meta_sha256"]:
            raise RuntimeError("Meta drift")
        meta = torch.load(str(meta_path), map_location="cpu")
        logits = payload["logits"][iid].float().cuda()
        if logits.shape != raw["native_logits"].shape or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid output logits")
        if epoch == 0 and not torch.equal(logits.cpu(), raw["native_logits"].float()):
            raise RuntimeError("No-op changed native logits")
        gt = ground_truth(raw)
        target = meta["target"].numpy()
        rec = dict(target=target, unique_proposal=meta["unique_proposal"].numpy(),
                   unique_labels=meta["unique_labels"].numpy(), gt_objects=meta["gt_objects"],
                   native_endpoint=meta["endpoint"].numpy())
        with torch.no_grad():
            pred = replay(post, raw, logits)
            row = metric.row(iid, pred, gt, dict(logits=logits), target)
            if epoch == 0:
                assert_baseline(row, prepared["manifest"][iid]["baseline"])
            extra = scopes(rec, logits, pred)
            base_pred = replay(post, raw, raw["native_logits"].float().cuda())
            base_extra = scopes(rec, raw["native_logits"].float(), base_pred)
        rows.append(row)
        extras.append(extra)
        atomic_json(stage/"images"/(iid+".json"), dict(image_id=iid, protocol_sha256=reg,
                    metrics=row, scopes=extra, baseline=prepared["manifest"][iid]["baseline"], baseline_scopes=base_extra))
        for k in KS:
            metric.result["sgdet_recall"][k].clear()
        if n % 50 == 0 or n == len(payload["image_ids"]):
            progress(stage, "full_native_sgdet_postprocessing", n, len(payload["image_ids"]), start)
    finish(stage, reg, len(rows), prediction_sha256=sha256(prediction_file),
           checkpoint_sha256=payload["checkpoint_sha256"], arm=arm, epoch=epoch,
           role=role, metrics=summarize_rows(rows), scopes=scope_summary(extras))


def summarize(p, reg, lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    from audit_identity_metric_history import paired_counts
    from vw_experiment import scope_summary
    choices, rows = {}, {}
    for arm in ARMS[1:]:
        done = read(OUT/lane/arm/"train/summary.json")
        if done["protocol_sha256"] != reg:
            raise RuntimeError("Wrong trained run")
        choices[arm] = done["selected_epoch"]
        stage = OUT/lane/arm/("held_epoch%d" % choices[arm])
        if not checked_done(stage, reg):
            raise RuntimeError("Incomplete held predictions")
        rows[arm] = [read(stage/"images"/(iid+".json")) for iid in p["split"]["held"]]
    metrics = {arm:[r["metrics"] for r in rr] for arm,rr in rows.items()}
    metrics["native"] = [r["baseline"] for r in rows[PRIMARY]]
    extras = {arm:[r["scopes"] for r in rr] for arm,rr in rows.items()}
    extras["native"] = [r["baseline_scopes"] for r in rows[PRIMARY]]
    n = len(p["split"]["held"])
    comparisons, scope_comparisons = {}, {}
    for reference, arm in [("native", a) for a in ARMS[1:]] + [("frozen_head", "adapt_plain"), ("adapt_plain", PRIMARY)]:
        b, u = metrics[reference], metrics[arm]
        support = np.array([r["class_recalls"][4] for r in b], float)
        other = np.array([r["class_recalls"][4] for r in u], float)
        den = np.array([r["positive_objects"] for r in b])
        if den.tolist() != [r["positive_objects"] for r in u] or not np.array_equal(np.isfinite(support), np.isfinite(other)):
            raise RuntimeError("Unpaired support")
        stats = policy_summary(np.arange(n), np.arange(n),
                np.array([y["post_nms_correct"]-x["post_nms_correct"] for x,y in zip(b,u)]),
                np.array([y["recalls"][4]-x["recalls"][4] for x,y in zip(b,u)]),
                np.nan_to_num(other-support), np.isfinite(support), den, np.zeros(n, int))
        stats["evaluated_images"] = stats.pop("selected_images")
        stats["interval_caveat"] = "Paired-image interval conditional on one selected fitted model; no adjustment for repeated adaptive exploration"
        key = arm+"_versus_"+reference
        comparisons[key] = stats
        scope_comparisons[key] = {}
        for scope in ["pre_nms", "post_nms", "unique_post", "native_top50_endpoint", "not_native_top50_endpoint"]:
            pair = [[dict(image_id=iid, **r[scope]) for iid,r in zip(p["split"]["held"], extras[k])] for k in [reference, arm]]
            scope_comparisons[key][scope] = paired_counts(pair[0], pair[1], "objects", "correct", draws=2000)
    checks = comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(OUT/lane, reg, n, smoke=p["smoke"], choices=choices, primary=PRIMARY,
           arms={a:summarize_rows(v) for a,v in metrics.items()},
           scopes={a:scope_summary(v) for a,v in extras.items()},
           comparisons=comparisons, scope_comparisons=scope_comparisons,
           primary_exploratory_numerical_screen=all(checks.values()),
           original_confirmation_min_images=1000, confirmation_coverage_satisfied=False,
           formal_gate_accepted=False, independent_confirmation=False,
           validation_accessed=False, test_accessed=False, reused_train_images=True,
           native_sgg_trained_on_these_images=True,
           next_step="Stop. No test, fresh gate, seed expansion or control promotion automatically.")


def main():
    from vk_gate_native import configure_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["register", "prepare", "evaluate", "summarize"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prediction_file", type=Path)
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(2)
    configure_backend()
    if args.stage == "register":
        _, reg, lane = register(args.smoke)
        print("[registered]", lane, reg, flush=True)
        return
    p, reg, lane = load(args.smoke)
    if args.stage == "prepare":
        prepare(p, reg, lane)
    elif args.stage == "evaluate":
        evaluate(p, reg, lane, args.prediction_file.resolve())
    else:
        summarize(p, reg, lane)


if __name__ == "__main__":
    main()
