"""Bounded, post-hoc SGDet capacity diagnostic on consumed V-K gate images."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources, paired_ratio, selected_ids
from repro_experiment import config, dataset, save_torch
from ve_experiment import lookup
from vb_native import OfficialMetrics, targets
from vc_protocol import summarize_rows, KS
from vk_gate_native import configure_backend, tensor_digest
from vk_gate_protocol import OUT as VK, CACHE, MANIFEST, CHECKPOINT, PRIMARY_SHA, ids
from vk_gate_math import repair
from r19_vk_decomposition import cached_prediction_check
from vl_policy import improves_identity, preserves_relations, candidate_order
import vj_mac as frozen


OUT = ROOT / "results/VL_sgdet_feasibility"
RAW = ROOT / "cache/R19_vk_mechanism/sgdet"
K50 = KS.index(50)


def all_candidates(features, baseline, state):
    expert = F.linear(features, state["expert_head"]["weight"], state["expert_head"]["bias"]).softmax(-1)
    _, p, q = frozen.probabilities(baseline, expert)
    visual, supported = frozen.visual_features(features, p, q, state["centers"], state["counts"])
    eligible = supported & (p.argmax(-1) != expert.argmax(-1)) & (p.argmax(-1) != q[:, 1:].argmax(-1))
    candidate = q.clamp_min(1e-30).log()
    inputs = torch.cat([frozen.confidence_features(p, expert), visual], -1)
    return candidate, eligible, p.max(-1)[0], inputs


def emit_progress(out, start, image_count, total, trials, stage, **extra):
    elapsed = time.monotonic() - start
    payload = dict(status="running", stage=stage, pid=os.getpid(), images=image_count,
        total=total, trials=trials, seconds=elapsed,
        eta_seconds=elapsed / image_count * (total - image_count) if image_count else None,
        timestamp=datetime.now(timezone.utc).isoformat(), **extra)
    atomic_json(out / "progress.json", payload)
    print(json.dumps(payload), flush=True)
    if elapsed > 7200:
        raise TimeoutError("V-L fixed two-hour execution budget reached")


def run(a):
    ensure_storage(); torch.set_num_threads(2); configure_backend()
    out = OUT / ("smoke" if a.smoke else "development200")
    guard = output_path(out / "execution.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = selected_ids(ids(), 200, "vl_sgdet_capacity_v1")
    if a.smoke:
        selected = selected[:3]
    if sha256(CHECKPOINT) != PRIMARY_SHA:
        raise RuntimeError("Frozen visual expert/router checkpoint changed")
    gate_sha = sha256(MANIFEST)
    inventory = {}
    for iid in selected:
        inventory[iid] = {name: sha256(path) for name, path in dict(
            raw=RAW / (iid + ".pt"), feature=CACHE / "sgdet/gate/features" / (iid + ".pt"),
            export=CACHE / "sgdet/gate/export" / (iid + ".pt"),
            update=CACHE / "sgdet/gate/updates" / (iid + ".pt")).items()}
    protocol = dict(version="vl_sgdet_capacity_v1", image_ids=selected, smoke=a.smoke,
        family="corrected PySGG Transformer", task="sgdet", device=a.device,
        input_sha256=inventory, original_gate_sha256=gate_sha, expert_router_sha256=PRIMARY_SHA,
        data_role="Consumed V-K gate reclassified as post-hoc development, NOT independent confirmation",
        alpha=.5, prototype_min_objects=5, candidate_order="ascending native foreground confidence, then proposal index",
        arms=["native", "frozen_vk", "unconditional", "pre_nms_oracle", "post_nms_oracle", "joint_oracle"],
        oracle_rule="One greedy pass; GT used to retain strictly positive post-NMS identity utility; joint arm also protects every supported predicate recall@50 and image R@50",
        semantics="Privileged feasible construction, NOT a proven optimum/upper bound and NOT a deployable repair",
        predicate_logits="frozen", pair_set="frozen", class_boxes="native", ranking="native postprocessor",
        stop="Cache parity failure or two hours; no threshold search, extra seeds, test-set evaluation or gate override",
        original_gate_accepted=False, confirmatory_claim=False,
        sources=sources(["vl_sgdet_feasibility.py", "vl_policy.py", "vk_gate_math.py", "vj_mac.py",
                         "repro_experiment.py", "vb_native.py", "vc_protocol.py", "evidence_completion.py"]))
    reg = lock(out / "protocol.json", protocol)
    if (out / "summary.json").exists():
        if read(out / "summary.json")["protocol_sha256"] != reg:
            raise RuntimeError("Existing result protocol differs")
        print("Already complete: " + str(out), flush=True)
        return
    cfg = config("sgdet", out)
    from pysgg.modeling.roi_heads.relation_head.inference import make_roi_relation_post_processor
    from pysgg.structures.bounding_box import BoxList
    post = make_roi_relation_post_processor(cfg)
    if post.use_relness_ranking or post.BCE_loss or post.attribute_on or post.use_gt_box:
        raise RuntimeError("Unsupported native postprocessing contract")
    ds = dataset(cfg, "val"); mapping = lookup(ds); metric = OfficialMetrics("sgdet")
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    records = []; trials = 0; start = time.monotonic()
    for n, iid in enumerate(selected, 1):
        path = out / "images" / (iid + ".json")
        if path.exists():
            record = read(path)
            if record["protocol_sha256"] != reg:
                raise RuntimeError("Resume provenance mismatch")
            records.append(record)
            continue
        raw = torch.load(str(RAW / (iid + ".pt")), map_location="cpu")
        feature = torch.load(str(CACHE / "sgdet/gate/features" / (iid + ".pt")), map_location="cpu")
        export = torch.load(str(CACHE / "sgdet/gate/export" / (iid + ".pt")), map_location="cpu")
        update = torch.load(str(CACHE / "sgdet/gate/updates" / (iid + ".pt")), map_location="cpu")
        if any(x["protocol_sha256"] != gate_sha for x in [feature, export, update]):
            raise RuntimeError("Input belongs to another gate")
        if (not torch.equal(raw["native_logits"], export["baseline"])
                or not torch.equal(raw["repaired_logits"], update["logits"])
                or not torch.equal(feature["crop_boxes"], export["crop_boxes"])
                or tensor_digest(raw["relation_logits"]) != export["predicate_sha256"]
                or tensor_digest(raw["pairs"]) != export["pair_sha256"]):
            raise RuntimeError("Raw/cache/crop/predicate/pair mismatch")
        gt = ds.get_groundtruth(mapping[iid], evaluation=True)
        y = targets(raw, gt, "sgdet")
        if not np.array_equal(y, raw["target"].numpy()):
            raise RuntimeError("Positive-proposal target matching changed")
        tensors = {k: raw[k].to(a.device) for k in ["native_logits", "repaired_logits", "proposal_boxes", "boxes_per_cls", "pairs", "relation_logits"]}

        def replay(logits):
            box = BoxList(tensors["proposal_boxes"].clone(), raw["size"], "xyxy")
            box.add_field("boxes_per_cls", tensors["boxes_per_cls"].clone())
            return post.forward(([tensors["relation_logits"]], [logits]), [tensors["pairs"]], [box])[0]

        def evaluate(logits):
            prediction = replay(logits)
            row = metric.row(iid, prediction, gt, dict(logits=logits), y)
            for k in KS:
                metric.result["sgdet_recall"][k].clear()
            return prediction, row

        with torch.no_grad():
            base_logits = tensors["native_logits"]
            base_pred, base_row = evaluate(base_logits)
            vk_pred, vk_row = evaluate(tensors["repaired_logits"])
            cached_prediction_check(base_pred, VK / "sgdet/gate/native/predictions" / (iid + ".npz"))
            cached_prediction_check(vk_pred, VK / "sgdet/gate/selective/predictions" / (iid + ".npz"))
            reproduced = repair(feature["features"], raw["native_logits"], state)
            if not torch.allclose(reproduced["logits"], raw["repaired_logits"], atol=2e-6, rtol=1e-5):
                raise RuntimeError("Frozen V-K repair cannot be reproduced")
            candidate, eligible, confidence, router_inputs = all_candidates(feature["features"], raw["native_logits"], state)
            candidates = candidate_order(eligible.nonzero(as_tuple=False).flatten().tolist(), confidence)
            candidate = candidate.to(a.device)
            unconditional = torch.where(eligible.to(a.device)[:, None], candidate, base_logits)
            _, unconditional_row = evaluate(unconditional)
            pre_help = eligible & (torch.tensor(y) > 0) & (raw["native_logits"][:, 1:].argmax(-1) + 1 != torch.tensor(y)) & (candidate.cpu()[:, 1:].argmax(-1) + 1 == torch.tensor(y))
            _, pre_row = evaluate(torch.where(pre_help.to(a.device)[:, None], candidate, base_logits))
            arm_logits = {name: base_logits.clone() for name in ["post_nms_oracle", "joint_oracle"]}
            arm_rows = {name: base_row for name in arm_logits}
            accepted = {name: [] for name in arm_logits}
            observations = []
            base_labels = base_pred.get_field("pred_labels").cpu()
            top50 = set(base_pred.get_field("rel_pair_idxs")[:50].cpu().flatten().tolist())
            for j, index in enumerate(candidates, 1):
                single = base_logits.clone(); single[index] = candidate[index]
                single_pred, single_row = evaluate(single); trials += 1
                observations.append(dict(proposal=index, target=int(y[index]),
                    foreground_confidence=float(confidence[index]), original_router_probability=float(reproduced["repair_probability"][index]),
                    baseline_top50_endpoint=index in top50, baseline_post_label=int(base_labels[index]),
                    single_post_label=int(single_pred.get_field("pred_labels")[index]),
                    post_identity_delta=single_row["post_nms_correct"] - base_row["post_nms_correct"],
                    R50_delta=single_row["recalls"][K50] - base_row["recalls"][K50],
                    class_R50_delta=[None if x is None else z-x for x,z in zip(base_row["class_recalls"][K50], single_row["class_recalls"][K50])],
                    joint_single_utility=improves_identity(base_row,single_row) and preserves_relations(base_row,single_row,K50)))
                for name in arm_logits:
                    if not accepted[name]:
                        trial_logits, trial_row = single, single_row
                    else:
                        trial_logits = arm_logits[name].clone(); trial_logits[index] = candidate[index]
                        _, trial_row = evaluate(trial_logits); trials += 1
                    keep = improves_identity(arm_rows[name], trial_row)
                    if name == "joint_oracle":
                        keep = keep and preserves_relations(arm_rows[name], trial_row, K50)
                    if keep:
                        arm_logits[name], arm_rows[name] = trial_logits, trial_row
                        accepted[name].append(index)
                if j % 25 == 0:
                    emit_progress(out,start,n-1,len(selected),trials,"candidate_audit",image_id=iid,candidate=j,candidates=len(candidates))
            rows = dict(native=base_row, frozen_vk=vk_row, unconditional=unconditional_row,
                        pre_nms_oracle=pre_row, **arm_rows)
            record = dict(image_id=iid, protocol_sha256=reg, candidates=len(candidates),
                native_and_vk_cache_parity=True, arms=rows, selected=accepted, single_updates=observations)
            # Keep label-free inputs and GT-derived diagnostic outcomes separate.
            save_torch(out / "features" / (iid + ".pt"), dict(image_id=iid, protocol_sha256=reg,
                router_inputs=router_inputs, candidate_indices=candidates,
                foreground_confidence=confidence, baseline_post_labels=base_labels,
                baseline_top50_endpoints=sorted(top50),
                proposal_boxes=raw["proposal_boxes"], boxes_per_cls=raw["boxes_per_cls"], size=raw["size"]))
            atomic_json(path, record); records.append(record)
        emit_progress(out,start,n,len(selected),trials,"image_complete",image_id=iid)
    reference = [r["arms"]["native"] for r in records]
    aggregates = {}
    for name in records[0]["arms"]:
        rows = [r["arms"][name] for r in records]
        aggregates[name] = dict(metrics=summarize_rows(rows),
            post_nms_identity=paired_ratio([r["post_nms_correct"] for r in rows], [r["post_nms_correct"] for r in reference], [r["positive_objects"] for r in reference]),
            R50=paired_ratio([r["recalls"][K50] for r in rows], [r["recalls"][K50] for r in reference], np.ones(len(rows))))
    diagnostic_gain = aggregates["joint_oracle"]["metrics"]["post_nms_object_top1"] - aggregates["native"]["metrics"]["post_nms_object_top1"]
    summary = dict(status="complete", images=len(records), smoke=a.smoke, protocol_sha256=reg,
        results=aggregates, total_candidates=sum(r["candidates"] for r in records),
        privileged_joint_identity_gain=diagnostic_gain,
        scientific_gate_pass=False, confirmatory_claim=False,
        interpretation="Oracle arms use GT for selection; only demonstrate a feasible privileged correction, not learned-method improvement or an optimal bound",
        elapsed_seconds=time.monotonic()-start)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", dict(status="complete",images=len(records),total=len(selected),summary=str(out / "summary.json")))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        atomic_json(OUT / ("smoke" if args.smoke else "development200") / "progress.json",
                    dict(status="failed", error=repr(error), pid=os.getpid()))
        raise
