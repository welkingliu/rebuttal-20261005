"""R2: intervene on the semantic embedding while holding visual inputs fixed.

The auxiliary frequency-only route permutes posterior class coordinates without
changing its probability multiset. TDE's average-context branch stays untouched.
This is an internal, GT-box-conditioned intervention, not real-world causality.
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, ensure_storage, atomic_json, output_path, sha256
from native_runtime import load_model, dataset, image_id, infer, assets

SOURCE_PATH = Path(__file__).resolve()


def stable_seed(seed, image):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + str(image)).encode()).digest()[:4], "big")


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)
    elif hasattr(value, "bbox"):
        yield value.bbox
        for key in value.fields():
            field = value.get_field(key)
            if isinstance(field, torch.Tensor):
                yield field


def fingerprint(tensors):
    digest = hashlib.sha256()
    for x in tensors:
        x = x.detach().cpu().contiguous().numpy()
        digest.update(str((x.shape, str(x.dtype))).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


class SemanticReplay:
    def __init__(self, predictor):
        self.predictor = predictor
        self.context = predictor.context_layer
        self.embedding = self.context.obj_embed2

    def run(self, inputs, replacement=None, route="embedding"):
        embedding_calls, context_calls = [], []

        def embedding_hook(module, args, output):
            embedding_calls.append(args[0].detach().clone())
            # The second call, when present, is TDE's average-context branch.
            if replacement is not None and route in ("embedding", "both") and len(embedding_calls) == 1:
                if args[0].shape != replacement.shape:
                    raise RuntimeError("Identity embedding order/cardinality mismatch")
                return F.embedding(replacement, module.weight)
            return output

        def context_hook(module, args, output):
            context_calls.append(output)
            if replacement is None or route not in ("frequency", "both") or len(context_calls) != 1:
                return output
            logits, labels = output[:2]
            changed = (labels != replacement).nonzero().flatten()
            shifted = logits.clone()
            # Swap old and new class coordinates: preserve entropy and confidence.
            shifted[changed, labels[changed]] = logits[changed, replacement[changed]]
            shifted[changed, replacement[changed]] = logits[changed, labels[changed]]
            return (shifted, replacement.clone()) + output[2:]

        hooks = [self.embedding.register_forward_hook(embedding_hook),
                 self.context.register_forward_hook(context_hook)]
        try:
            with torch.no_grad():
                output = self.predictor(*inputs)
        finally:
            for hook in hooks:
                hook.remove()
        if not embedding_calls or not context_calls:
            raise RuntimeError("No consumed identity embedding was observed")
        if not torch.equal(embedding_calls[0], context_calls[0][1]):
            raise RuntimeError("Embedding indices are not in native object order")
        return output, embedding_calls[0]


def capture_image(model, cfg, transform, ds, index):
    predictor = model.roi_heads["relation"].predictor
    sampler = model.roi_heads["relation"].samp_processor
    target = ds.get_groundtruth(index, evaluation=True)
    gt_rel = target.get_field("relation_tuple").long()
    pairs = torch.unique(gt_rel[:, :2], dim=0)
    holder = {}
    def before(module, args):
        holder["inputs"] = args
    def after(module, args, output):
        holder["output"] = output
    original_sampler = sampler.prepare_test_pairs
    sampler.prepare_test_pairs = lambda device, proposals: [pairs.to(device)]
    hooks = [predictor.register_forward_pre_hook(before), predictor.register_forward_hook(after)]
    try:
        infer(model, cfg, transform, ds, index)
    finally:
        sampler.prepare_test_pairs = original_sampler
        for h in hooks:
            h.remove()
    inputs = holder["inputs"]
    if not torch.equal(inputs[1][0].cpu(), pairs):
        raise RuntimeError("Candidate pairs changed")
    if len(inputs[0][0]) != len(target):
        raise RuntimeError("Predicted node order/cardinality differs from GT boxes")
    pair_lookup = {tuple(p): i for i, p in enumerate(pairs.tolist())}
    rows = torch.tensor([pair_lookup[tuple(p)] for p in gt_rel[:, :2].tolist()], device="cuda")
    return holder, target, gt_rel, rows


def conditions(labels, target, confusion, counts, iid):
    yield "clean", None, "embedding", 0, 0.
    yield "oracle", target.clone(), "embedding", 0, 1.
    eligible = (labels == target).nonzero().flatten().cpu().numpy()
    ranks = np.argsort(np.argsort(-counts, kind="stable"), kind="stable")
    strata = ranks // 30
    for seed in (17, 23, 31):
        rng = np.random.default_rng(stable_seed(seed, iid))
        order = rng.permutation(eligible)
        conf = labels.cpu().numpy().copy()
        rand = conf.copy()
        for index in order:
            gt = int(conf[index])
            row = confusion[gt].copy()
            row[0], row[gt] = -1, -1
            if row.max() > 0:
                replacement = int(row.argmax())
            else:
                pool = np.array([c for c in range(1, 151) if c != gt])
                replacement = int(pool[np.argmin(np.abs(counts[pool] - counts[gt]))])
            conf[index] = replacement
            pool = np.array([c for c in range(1, 151) if c != gt and strata[c] == strata[replacement]])
            rand[index] = int(rng.choice(pool))
        for fraction in (.25, .5, 1.):
            selected = order[:int(math.ceil(fraction * len(order)))]
            for name, values in (("confusable", conf), ("matched_random", rand)):
                altered = labels.clone()
                altered[torch.as_tensor(selected, device=labels.device)] = torch.as_tensor(values[selected], device=labels.device)
                yield "%s_s%d_p%d" % (name, seed, int(100*fraction)), altered, "embedding", seed, fraction
    # Limited route decomposition, not a repeated sweep of every condition.
    yield "oracle_frequency", target.clone(), "frequency", 0, 1.
    yield "oracle_both", target.clone(), "both", 0, 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["transformer", "tde_motifs"])
    parser.add_argument("--stage", required=True, choices=["validation", "dev", "test"])
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    out = ROOT / "results/R2" / args.family / args.stage
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, transform, provenance = load_model(args.family, "sgcls", out)
    ds = dataset(cfg, "test" if args.stage == "test" else "val")
    selected = sorted(range(len(ds)), key=lambda i: hashlib.sha256(("R2-20260929:"+image_id(ds,i)).encode()).digest())
    if args.stage == "dev":
        selected = selected[1000:1100]
    else:
        selected = selected[:1000 if args.stage == "validation" else 2000]
    if args.samples:
        selected = selected[:args.samples]
    protocol = dict(model=provenance, stage=args.stage, image_ids=[image_id(ds,i) for i in selected],
                    pair_protocol="all annotated GT pairs, unique rows inferred once; duplicate GT predicate annotations retained",
                    background_policy="foreground predicate argmax excluding id 0",
                    frequency_control="main route preserves native object logits and predicted labels; auxiliary route swaps posterior coordinates",
                    seeds=[17,23,31], strengths=[.25,.5,1.], confusion_source="validation only",
                    conditional_corruption="Only originally correct identities are eligible",
                    repair="Oracle replacement of incorrect native identities",
                    context_average_branch="unchanged", code_sha256=sha256(SOURCE_PATH))
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise RuntimeError("Existing R2 protocol differs; use a distinct output directory")
    atomic_json(protocol_file, protocol)
    confusion = np.zeros((151,151), dtype=np.int64)
    counts = np.zeros(151, dtype=np.int64)
    if args.stage != "validation":
        source = out.parent / "validation/confusion.npz"
        with np.load(source) as data:
            confusion, counts = data["confusion"], data["counts"]
        if args.stage == "test":
            gate = json.loads((out.parent / "dev/summary.json").read_text())
            if not gate["invariance_passed"] or gate["images"] < 100:
                raise RuntimeError("R2 requires a 100-image held-out development gate")
            r1=json.loads((ROOT/"results/R1"/args.family/"sgcls/summary.json").read_text())
            if not r1["full_split"] or not all(v["passes_existing_tolerance"] for v in r1["reproduction_checks"].values()):
                raise RuntimeError("Formal R2 requires full SGCls reference reproduction; retain development as diagnostic only")
    replay = SemanticReplay(model.roi_heads["relation"].predictor)
    started = time.monotonic()
    evidence = []
    for number, index in enumerate(selected):
        iid = image_id(ds, index)
        path = out / "images" / (iid + ".npz")
        record_path = out / "images" / (iid + ".json")
        if record_path.exists() and path.exists():
            row = json.loads(record_path.read_text())
            if row["protocol_sha256"] != sha256(protocol_file):
                raise RuntimeError("Resume provenance mismatch")
            evidence.append(row)
            if args.stage == "validation":
                with np.load(path) as z:
                    np.add.at(confusion, (z["object_gt"], z["object_pred"]), 1)
                    np.add.at(counts, z["object_gt"], 1)
            continue
        holder, target, gt_rel, pair_rows = capture_image(model, cfg, transform, ds, index)
        inputs = holder["inputs"]
        leaves = list(tensor_leaves(inputs))
        before = fingerprint(leaves)
        baseline, labels = replay.run(inputs)
        native = holder["output"][1][0]
        clean = baseline[1][0]
        if not torch.allclose(native, clean, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Predictor replay is not identical to native inference")
        noop, _ = replay.run(inputs, labels, "embedding")
        if not torch.allclose(clean, noop[1][0], atol=1e-5, rtol=1e-5):
            raise RuntimeError("No-op identity intervention changed relation output")
        truth = target.get_field("labels").long().cuda()
        gt = gt_rel[:,2].numpy()
        clean_pred = (clean[pair_rows,1:].argmax(1)+1).cpu().numpy()
        logits = baseline[0][0].detach().cpu().numpy()
        cache_comparison = None
        if args.stage == "test":
            cache=assets(args.family,"sgcls")[-1]/"predictions/sgcls"/(iid+".npz")
            with np.load(cache) as z:
                lookup={tuple(p):i for i,p in enumerate(z["pred_rel_pairs"].tolist())}
                shared=[(i,lookup[tuple(p)]) for i,p in enumerate(gt_rel[:,:2].tolist()) if tuple(p) in lookup]
                if shared:
                    a,b=zip(*shared)
                    probs=clean[pair_rows].softmax(-1).cpu().numpy()
                    ref=z["pred_rel_scores"][list(b)]
                    error=float(np.abs(probs[list(a)]-ref).max())
                    cache_comparison=dict(shared_gt_rows=len(shared),max_abs_probability_error=error,
                                          top1_agreement=float((probs[list(a),1:].argmax(1)==ref[:,1:].argmax(1)).mean()))
                    if error>1e-3:
                        raise RuntimeError("Live clean output differs from audited cache: %s"%cache_comparison)
        payload = dict(object_gt=truth.cpu().numpy(), object_pred=labels.cpu().numpy(),
                       object_logits=logits, relation_gt=gt, relation_pairs=gt_rel[:,:2].numpy(),
                       clean_prediction=clean_pred, clean_logits=clean[pair_rows].detach().cpu().numpy())
        stats = {}
        if args.stage == "validation":
            np.add.at(confusion, (payload["object_gt"], payload["object_pred"]), 1)
            np.add.at(counts, payload["object_gt"], 1)
        else:
            for name, replacement, route, seed, fraction in conditions(labels, truth, confusion, counts, iid):
                if replacement is None:
                    result = baseline
                    changed = torch.zeros_like(labels, dtype=torch.bool)
                else:
                    result, _ = replay.run(inputs, replacement, route)
                    changed = replacement != labels
                if route == "embedding" and not torch.equal(result[0][0], baseline[0][0]):
                    raise RuntimeError("Main semantic intervention altered output object logits")
                prediction = (result[1][0][pair_rows,1:].argmax(1)+1).cpu().numpy()
                incident = (changed[gt_rel[:,0].cuda()] | changed[gt_rel[:,1].cuda()]).cpu().numpy()
                hit, initial = prediction==gt, clean_pred==gt
                stats[name] = dict(route=route, seed=seed, fraction=fraction,
                    changed_nodes=int(changed.sum()), objects=len(truth), relations=len(gt),
                    correct=int(hit.sum()), clean_correct=int(initial.sum()),
                    wrong_to_correct=int((hit & ~initial).sum()), correct_to_wrong=int((~hit & initial).sum()),
                    touched_relations=int(incident.sum()), touched_correct=int(hit[incident].sum()),
                    touched_clean_correct=int(initial[incident].sum()))
                payload["prediction_"+name] = prediction
                payload["incident_"+name] = incident
        if before != fingerprint(list(tensor_leaves(inputs))):
            raise RuntimeError("Visual features, geometry, or candidate pairs were mutated")
        row = dict(image_id=iid, protocol_sha256=sha256(protocol_file), feature_fingerprint=before,
                   invariance_passed=True, stats=stats, cached_clean_comparison=cache_comparison,
                   native_replay_max_abs=float((native-clean).abs().max()))
        np.savez_compressed(output_path(path), **payload)
        atomic_json(record_path,row)
        evidence.append(row)
        if (number + 1) % 10 == 0:
            progress = dict(images=number+1, total=len(selected), seconds=time.monotonic()-started)
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress),flush=True)
        del holder, inputs, leaves, baseline, noop, clean, native
    if args.stage == "validation":
        np.savez_compressed(output_path(out / "confusion.npz"), confusion=confusion, counts=counts)
    aggregate = {}
    if args.stage != "validation":
        for name in evidence[0]["stats"]:
            rows = [r["stats"][name] for r in evidence]
            values = np.array([[r["correct"],r["clean_correct"],r["relations"]] for r in rows],dtype=float)
            rng = np.random.default_rng(17)
            boots = []
            for _ in range(2000):
                s = values[rng.integers(0,len(values),len(values))].sum(0)
                boots.append((s[0]-s[1])/s[2])
            sums = values.sum(0)
            aggregate[name] = dict(hit1=sums[0]/sums[2], clean_hit1=sums[1]/sums[2],
                delta_hit1=(sums[0]-sums[1])/sums[2], bootstrap_95ci=np.quantile(boots,[.025,.975]).tolist(),
                bootstrap_unit="image", relations=int(sums[2]), images=len(rows),
                changed_nodes=sum(r["changed_nodes"] for r in rows),
                wrong_to_correct=sum(r["wrong_to_correct"] for r in rows),
                correct_to_wrong=sum(r["correct_to_wrong"] for r in rows))
    atomic_json(out / "summary.json", dict(status="complete", stage=args.stage, images=len(evidence),
        invariance_passed=all(r["invariance_passed"] for r in evidence), results=aggregate,
        elapsed_seconds=time.monotonic()-started, protocol_sha256=sha256(protocol_file),
        significance_note="Seeds describe intervention randomness, not independent checkpoint training"))
    print("[COMPLETE] R2",args.family,args.stage,flush=True)


if __name__ == "__main__":
    main()
