"""R12: preserve detected proposals/visual evidence and intervene on identity paths."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from identity_intervention import SemanticReplay, tensor_leaves, fingerprint, stable_seed
from identity_evidence import geometry_assignment, image_record, summarize_records
from native_runtime import load_model, image_id, infer
from sgdet_identity_protocol import OUT, SPEC, verify, reference_gate


def conditions(labels, target, matched, confusion, counts, iid):
    yield "clean", None, "embedding"
    oracle = labels.clone()
    oracle[matched] = target[matched]
    yield "oracle", oracle, "embedding"
    yield "oracle_frequency", oracle, "frequency"
    yield "oracle_both", oracle, "both"
    eligible = ((labels == target) & matched).nonzero().flatten().cpu().numpy()
    ranks = np.argsort(np.argsort(-counts, kind="stable"), kind="stable")
    strata = ranks//30
    for seed in SPEC["seeds"]:
        rng = np.random.default_rng(stable_seed(seed, iid))
        order = rng.permutation(eligible)
        confusable, random = labels.cpu().numpy().copy(), labels.cpu().numpy().copy()
        for i in order:
            truth = int(target[i])
            row = confusion[truth].copy()
            row[0], row[truth] = -1, -1
            pool = np.array([c for c in range(1, 151) if c != truth])
            replacement = int(row.argmax()) if row.max() > 0 else int(pool[np.argmin(np.abs(counts[pool]-counts[truth]))])
            confusable[i] = replacement
            pool = np.array([c for c in range(1,151) if c != truth and strata[c] == strata[replacement]])
            if not len(pool):
                raise RuntimeError("Empty frequency-matched control stratum")
            random[i] = int(rng.choice(pool))
        for fraction in SPEC["strengths"]:
            selected = torch.as_tensor(order[:int(math.ceil(fraction*len(order)))], device=labels.device)
            for kind, values in [("confusable", confusable), ("matched_random", random)]:
                altered = labels.clone()
                altered[selected] = torch.as_tensor(values, device=labels.device)[selected]
                for route in SPEC["routes"]:
                    yield "%s_%s_s%d_p%d" % (kind, route, seed, int(fraction*100)), altered, route


def capture(model, cfg, transform, ds, index):
    predictor = model.roi_heads.relation.predictor
    holder = {}
    def before(module, args):
        holder["inputs"] = args
    def after(module, args, output):
        holder["output"] = output
    handles = [predictor.register_forward_pre_hook(before), predictor.register_forward_hook(after)]
    try:
        prediction, _ = infer(model, cfg, transform, ds, index)
    finally:
        for handle in handles:
            handle.remove()
    return holder, prediction


def evaluable_pairs(gt_relations, mapping, native_pairs):
    inverse = {int(gt): p for p, gt in enumerate(mapping) if gt >= 0}
    lookup = {tuple(pair): index for index, pair in enumerate(native_pairs.tolist())}
    if len(lookup) != len(native_pairs):
        raise RuntimeError("Duplicate native relation candidates")
    gt_rows, native_rows, pairs = [], [], []
    counts = dict(gt_relations=len(gt_relations), missing_endpoint=0, missing_native_candidate=0)
    for row, (s, o, predicate) in enumerate(gt_relations):
        if int(s) not in inverse or int(o) not in inverse:
            counts["missing_endpoint"] += 1
            continue
        pair = (inverse[int(s)], inverse[int(o)])
        if pair not in lookup:
            counts["missing_native_candidate"] += 1
            continue
        gt_rows.append(row)
        native_rows.append(lookup[pair])
        pairs.append(pair)
    counts["evaluable_relations"] = len(gt_rows)
    if sum(counts[k] for k in ["missing_endpoint","missing_native_candidate","evaluable_relations"]) != len(gt_relations):
        raise RuntimeError("GT relation denominator did not partition")
    return np.array(gt_rows, dtype=int), np.array(native_rows, dtype=int), np.array(pairs,dtype=int).reshape(-1,2), counts


def load(family, out):
    if family == "transformer":
        from repro_experiment import build
        weight = ROOT/"results/R9_reproduction/sgdet/training/model_final.pth"
        model, cfg, transform, provenance = build("sgdet", out, weights=weight)
        audit = json.loads((ROOT/"results/R9_reproduction/sgdet/eval_retrained/test/model_provenance.json").read_text())
        if provenance["checkpoint_sha256"] != audit["checkpoint_sha256"]:
            raise RuntimeError("R12 checkpoint differs from R9 reference evaluation")
    else:
        model, cfg, transform, provenance = load_model(family, "sgdet", out)
    model.eval()
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
        raise RuntimeError("R12 requires native detected proposals")
    if model.roi_heads.relation.rel_prop_on:
        raise RuntimeError("Target-dependent relation proposal branch requires a separate audit")
    return model, cfg, transform, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SPEC["families"], required=True)
    parser.add_argument("--stage", choices=list(SPEC["samples"]), required=True)
    args = parser.parse_args()
    ensure_storage()
    registration = verify()
    reference = reference_gate(args.family)
    if not reference["passed"]:
        raise RuntimeError("Reference gate not met: " + reference["reason"])
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    out = OUT/args.family/args.stage
    if args.stage == "test":
        gate = json.loads((out.parent/"dev/summary.json").read_text())
        if not gate.get("development_gate_passed"):
            raise RuntimeError("R12 development invariance/support gate not passed")
    model, cfg, transform, provenance = load(args.family, out)
    from repro_experiment import dataset
    ds = dataset(cfg, "test" if args.stage == "test" else "val")
    selected = sorted(range(len(ds)), key=lambda i: hashlib.sha256(("R12-20260930:"+image_id(ds,i)).encode()).digest())
    offset = {"validation":0, "dev":1000, "smoke":1100, "test":0}[args.stage]
    selected = selected[offset:offset+SPEC["samples"][args.stage]]
    protocol = dict(registration_sha256=registration, model=provenance, reference_gate=reference, stage=args.stage,
                    spec=SPEC, image_ids=[image_id(ds,i) for i in selected])
    pf = out/"protocol.json"
    if pf.exists() and json.loads(pf.read_text()) != protocol:
        raise RuntimeError("R12 resume protocol mismatch")
    atomic_json(pf, protocol)
    protocol_sha = sha256(pf)
    confusion, counts = np.zeros((151,151),dtype=np.int64), np.zeros(151,dtype=np.int64)
    if args.stage in ("dev","test"):
        with np.load(out.parent/"validation/confusion.npz") as stored:
            confusion, counts = stored["confusion"], stored["counts"]
    replay = SemanticReplay(model.roi_heads.relation.predictor)
    records, coverage, start, computed = [], [], time.monotonic(), 0
    for number, index in enumerate(selected):
        iid = image_id(ds,index)
        file, record_path = out/"images"/(iid+".npz"), out/"images"/(iid+".json")
        if file.exists() and record_path.exists():
            stored = json.loads(record_path.read_text())
            if stored["protocol_sha256"] != protocol_sha:
                raise RuntimeError("Image record registration mismatch")
            with np.load(file, allow_pickle=False) as z:
                if args.stage == "validation":
                    matched = z["proposal_to_gt"] >= 0
                    np.add.at(confusion, (z["object_gt"][matched],z["object_pred"][matched]), 1)
                    np.add.at(counts, z["object_gt"][matched], 1)
                records.append(image_record(z))
            coverage.append(stored)
            continue
        holder, native_prediction = capture(model,cfg,transform,ds,index)
        inputs = holder["inputs"]
        leaves = list(tensor_leaves(inputs))
        before = fingerprint(leaves)
        baseline, labels = replay.run(inputs)
        clean = baseline[1][0]
        if not torch.allclose(clean,holder["output"][1][0],atol=1e-5,rtol=1e-5):
            raise RuntimeError("Replay differs from native predictor")
        for route in SPEC["routes"]:
            noop, _ = replay.run(inputs, labels, route)
            if not torch.allclose(clean,noop[1][0],atol=1e-5,rtol=1e-5):
                raise RuntimeError("No-op changes predictions in " + route)
        proposals, native_pairs = inputs[0][0], inputs[1][0].detach().cpu().numpy()
        gt = ds.get_groundtruth(index,evaluation=True).resize(proposals.size)
        mapping, overlaps = geometry_assignment(proposals.convert("xyxy").bbox.detach().cpu().numpy(), gt.convert("xyxy").bbox.numpy())
        matched = torch.as_tensor(mapping>=0, device=labels.device)
        truth = labels.clone()
        gt_labels = gt.get_field("labels").long().to(labels.device)
        truth[matched] = gt_labels[torch.as_tensor(mapping[mapping>=0], device=labels.device)]
        relations = gt.get_field("relation_tuple").cpu().numpy()
        gt_rows, rows, pairs, denominator = evaluable_pairs(relations,mapping,native_pairs)
        pair_rows = torch.as_tensor(rows,device=labels.device)
        # Verify that the raw predictor replay also matches native postprocessed scores.
        lookup = {tuple(pair):r for r,pair in enumerate(native_pairs.tolist())}
        pp = native_prediction.get_field("rel_pair_idxs").detach().cpu().numpy()
        inds = [lookup[tuple(pair)] for pair in pp]
        native_prob = native_prediction.get_field("pred_rel_scores").detach().cpu()
        replay_prob = clean[torch.as_tensor(inds,device=labels.device)].softmax(-1).detach().cpu()
        error = float((native_prob-replay_prob).abs().max()) if len(inds) else 0.
        if error > 1e-5:
            raise RuntimeError("Native postprocessor relation scores differ")
        payload = dict(proposal_to_gt=mapping, matching_iou=overlaps, object_gt=truth.cpu().numpy(),
                       object_pred=labels.cpu().numpy(), relation_gt=relations[gt_rows,2], relation_pairs=pairs,
                       gt_relation_rows=gt_rows, native_candidate_rows=rows, proposal_boxes=proposals.bbox.detach().cpu().numpy(),
                       native_candidate_pairs=native_pairs, clean_prediction=(clean[pair_rows,1:].argmax(1)+1).cpu().numpy())
        payload.update(native_output_object_labels=native_prediction.get_field("pred_labels").detach().cpu().numpy(),
                       native_output_boxes=native_prediction.bbox.detach().cpu().numpy(),
                       native_output_relation_pairs=pp, native_output_relation_scores=native_prob.numpy())
        route_records = {}
        iterator = [("clean",None,"embedding")] if args.stage == "validation" else conditions(labels,truth,matched,confusion,counts,iid)
        for name, replacement, route in iterator:
            result = baseline if replacement is None else replay.run(inputs,replacement,route)[0]
            changed = torch.zeros_like(labels,dtype=torch.bool) if replacement is None else replacement != labels
            if bool((changed & ~matched).any()):
                raise RuntimeError("Intervention altered an unmatched proposal")
            if route == "embedding" and not torch.equal(result[0][0],baseline[0][0]):
                raise RuntimeError("Semantic route changed output object logits")
            if route in ("frequency","both") and not torch.allclose(result[0][0].sort(1)[0],baseline[0][0].sort(1)[0],atol=1e-5,rtol=1e-5):
                raise RuntimeError("Frequency route did not preserve posterior confidence multiset")
            payload["prediction_"+name] = (result[1][0][pair_rows,1:].argmax(1)+1).cpu().numpy()
            changed_cpu = changed.cpu().numpy()
            payload["incident_"+name] = changed_cpu[pairs[:,0]] | changed_cpu[pairs[:,1]]
            route_records[name] = dict(route=route,changed_nodes=int(changed.sum()))
        if before != fingerprint(list(tensor_leaves(inputs))):
            raise RuntimeError("Intervention mutated fixed inputs")
        if args.stage == "validation":
            np.add.at(confusion,(payload["object_gt"][mapping>=0],payload["object_pred"][mapping>=0]),1)
            np.add.at(counts,payload["object_gt"][mapping>=0],1)
        record = dict(image_id=iid, protocol_sha256=protocol_sha, invariance_passed=True, input_fingerprint=before,
                      native_probability_max_error=error, proposals=len(proposals), gt_objects=len(gt),
                      matched_objects=int(matched.sum()), native_pairs=len(native_pairs), **denominator, routes=route_records)
        temporary = output_path(file.with_suffix(".tmp"))
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(file)
        atomic_json(record_path,record)
        records.append(image_record(payload)); coverage.append(record); computed += 1
        if number%10 == 0 or number+1 == len(selected):
            elapsed = time.monotonic()-start
            progress = dict(stage=args.stage,images=number+1,total=len(selected),seconds=elapsed,
                            eta_seconds=elapsed/computed*(len(selected)-number-1))
            atomic_json(out/"progress.json",progress); print(json.dumps(progress),flush=True)
        del holder, inputs, leaves, baseline, noop, result, clean, native_prediction
    if args.stage == "validation":
        with output_path(out/"confusion.npz").open("wb") as stream:
            np.savez_compressed(stream,confusion=confusion,counts=counts)
    totals = {key:sum(row[key] for row in coverage) for key in ["proposals","gt_objects","matched_objects","gt_relations","evaluable_relations","missing_endpoint","missing_native_candidate"]}
    evaluable_images = sum(row["evaluable_relations"]>0 for row in coverage)
    summary = summarize_records(records)
    summary.update(status="complete",family=args.family,stage=args.stage,images=len(records),coverage=totals,
                   evaluable_images=evaluable_images,coverage_fraction=totals["evaluable_relations"]/totals["gt_relations"] if totals["gt_relations"] else None,
                   invariance_passed=all(row["invariance_passed"] for row in coverage), reference_gate=reference,
                   protocol_sha256=protocol_sha, elapsed_seconds=time.monotonic()-start)
    summary["object_label_scope"] = SPEC["identity_scope"]
    summary["proposal_stage"] = SPEC["proposal_stage"]
    summary["development_gate_passed"] = bool(args.stage=="dev" and summary["invariance_passed"] and
        len(records)==SPEC["dev_gate"]["images"] and evaluable_images>=SPEC["dev_gate"]["minimum_evaluable_images"] and
        totals["evaluable_relations"]>=SPEC["dev_gate"]["minimum_relations"])
    atomic_json(out/"summary.json",summary)
    if args.stage=="dev" and not summary["development_gate_passed"]:
        raise RuntimeError("Development support/invariance gate failed; do not run test")
    if args.stage=="smoke" and totals["evaluable_relations"] == 0:
        raise RuntimeError("Smoke has no evaluable relations; inspect matching before proceeding")
    print("[COMPLETE] R12 %s %s"%(args.family,args.stage),flush=True)


if __name__ == "__main__":
    main()
