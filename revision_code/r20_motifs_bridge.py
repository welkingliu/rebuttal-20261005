"""Identity and matched visual controls on the independently reproduced plain Motifs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources, selected_ids, paired_ratio
from r16_motifs import build, dataset, RUN
from identity_intervention import SemanticReplay, tensor_leaves, fingerprint, conditions
from paired_visual_control import selected_nodes, mask


OUT = ROOT / "results/R20_plain_motifs_bridge"


def capture(model, cfg, transform, ds, index, image=None):
    from maskrcnn_benchmark.structures.image_list import to_image_list
    relation = model.roi_heads.relation
    target = ds.get_groundtruth(index, evaluation=True)
    gt = target.get_field("relation_tuple").long()
    pairs = torch.unique(gt[:, :2], dim=0)
    if image is None: image = ds[index][0]
    tensor, resized = transform(image, target)
    holder = {}
    def before(module, args): holder["inputs"] = args
    def after(module, args, output): holder["output"] = output
    hooks = [relation.predictor.register_forward_pre_hook(before), relation.predictor.register_forward_hook(after)]
    previous = relation.samp_processor.prepare_test_pairs
    relation.samp_processor.prepare_test_pairs = lambda device, proposals: [pairs.to(device)]
    try:
        with torch.no_grad():
            model(to_image_list([tensor.cuda()], size_divisible=32), [resized.to("cuda")])
    finally:
        relation.samp_processor.prepare_test_pairs = previous
        for hook in hooks: hook.remove()
    if not torch.equal(holder["inputs"][1][0].cpu(), pairs) or len(holder["inputs"][0][0]) != len(target):
        raise RuntimeError("GT support or candidate order changed")
    mapping = {tuple(p): i for i, p in enumerate(pairs.tolist())}
    rows = torch.tensor([mapping[tuple(p)] for p in gt[:, :2].tolist()], device="cuda")
    return holder, target, gt, rows


def check_native_cache(iid, pairs, scores):
    path = RUN / "sgcls/formal/test/predictions" / (iid + ".npz")
    with np.load(path) as z:
        mapping = {tuple(p): i for i, p in enumerate(z["pred_rel_pairs"].tolist())}
        kept = [(n, mapping[tuple(p)]) for n, p in enumerate(pairs.tolist()) if tuple(p) in mapping]
        if len(kept) != len(pairs): raise RuntimeError("Formal native cache missing GT pairs")
        a, b = zip(*kept)
        probabilities = scores.softmax(-1).cpu().numpy()
        error = float(np.abs(probabilities[list(a)] - z["pred_rel_scores"][list(b)]).max())
        if error > 1e-4: raise RuntimeError("Corrected native checkpoint/cache disagreement: " + str(error))
    return error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "validation", "identity", "visual"])
    a = p.parse_args(); ensure_storage(); torch.set_num_threads(2)
    torch.manual_seed(666); np.random.seed(666)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    out = OUT / a.stage
    source = RUN / "sgcls/formal"
    reference = read(source / "summary.json")
    if not reference["test"]["reproduction_passed"] or reference["test"]["images"] != 26446:
        raise RuntimeError("Plain Motifs has not passed full SGCls reproduction")
    weight = source / "best.pth"
    model, cfg = build("sgcls", out, weight); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    from maskrcnn_benchmark.data.transforms import build_transforms
    from reproduction_pair_audit import ChunkUnion
    transform = build_transforms(cfg, is_train=False)
    chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
    ds = dataset(cfg, "val" if a.stage in ("smoke", "validation") else "test")
    mapping = {Path(path).stem: i for i, path in enumerate(ds.filenames)}
    selected = selected_ids(mapping, 2000 if a.stage in ("identity", "visual") else 1100, "R2-20260929:")
    selected = selected[1000:1003] if a.stage == "smoke" else selected[:1000] if a.stage == "validation" else selected
    protocol = dict(version="r20_reproduced_plain_motifs_v1", stage=a.stage, task="sgcls", image_ids=selected,
        checkpoint=str(weight), checkpoint_sha256=sha256(weight), reference_summary_sha256=sha256(source / "summary.json"),
        reference_iteration=reference["selected_iteration"], training_seed=666, intervention_seeds=[17, 23, 31],
        support="GT boxes; original object ROI features, geometry and annotated candidate pairs fixed for identity intervention",
        identity_routes=["embedding", "frequency", "both"], corruption_strengths=[.25, .5, 1.],
        visual="First 200 hash-ordered eligible test images; one max-degree vs annotated degree-zero node, area ratio 0.5..2",
        visual_strengths=[.25, .5, 1.], visual_scope="Spatial visual evidence, NOT identity-only; annotation-relative controls",
        inference="All annotated GT pairs inferred once; duplicate predicate annotations retained; foreground predicate argmax",
        summary="Image-clustered descriptive paired bootstrap; intervention seeds are not independent training seeds",
        historical_results_replaced=False, test_selection="Locked by image hashes, never intervention outcome",
        sources=sources(["r20_motifs_bridge.py", "evidence_completion.py", "r16_motifs.py", "identity_intervention.py", "paired_visual_control.py"]))
    if a.stage in ("identity", "visual"):
        if not read(OUT / "smoke/summary.json")["invariance_passed"]: raise RuntimeError("Smoke audit failed")
        protocol["confusion_sha256"] = sha256(OUT / "validation/confusion.npz")
        if read(OUT / "validation/summary.json")["images"] != 1000: raise RuntimeError("Incomplete validation control source")
    reg = lock(out / "protocol.json", protocol)
    confusion = np.zeros((151, 151), dtype=np.int64); counts = np.zeros(151, dtype=np.int64)
    if a.stage in ("identity", "visual"):
        with np.load(OUT / "validation/confusion.npz") as z: confusion, counts = z["confusion"], z["counts"]
    replay = SemanticReplay(model.roi_heads.relation.predictor)
    records = []; excluded = []; start = time.monotonic()
    for n, iid in enumerate(selected, 1):
        if a.stage == "visual" and len(records) == 200: break
        path = out / "images" / (iid + ".json")
        if path.exists():
            row = read(path)
            if row["protocol_sha256"] != reg: raise RuntimeError("Resume provenance mismatch")
            records.append(row)
            if a.stage == "validation":
                with np.load(out / "images" / (iid + ".npz")) as z:
                    np.add.at(confusion, (z["object_gt"], z["object_pred"]), 1)
                    np.add.at(counts, z["object_gt"], 1)
            continue
        target = ds.get_groundtruth(mapping[iid], evaluation=True)
        gt = target.get_field("relation_tuple").numpy()
        nodes = None
        if a.stage == "visual":
            nodes, reason = selected_nodes(target.bbox.numpy(), gt[:, :2])
            if nodes is None:
                excluded.append(dict(image_id=iid, reason=reason)); continue
        holder, target, gt_rel, pair_rows = capture(model, cfg, transform, ds, mapping[iid])
        inputs = holder["inputs"]; before = fingerprint(list(tensor_leaves(inputs)))
        clean, labels = replay.run(inputs)
        native = holder["output"][1][0]
        if not torch.allclose(native, clean[1][0], atol=1e-5, rtol=1e-5): raise RuntimeError("Native replay drift")
        noop, _ = replay.run(inputs, labels, "both")
        if not torch.equal(clean[1][0], noop[1][0]): raise RuntimeError("No-op intervention not exact")
        scores = clean[1][0][pair_rows]; truth = target.get_field("labels").long().cuda()
        relation_truth = gt_rel[:, 2].numpy(); prediction = (scores[:, 1:].argmax(-1)+1).cpu().numpy()
        hit = prediction == relation_truth
        error = check_native_cache(iid, gt_rel[:, :2], scores) if a.stage in ("identity", "visual") else None
        payload = dict(object_gt=truth.cpu().numpy(), object_pred=labels.cpu().numpy(),
                       relation_gt=relation_truth, pairs=gt_rel[:, :2].numpy(), clean_prediction=prediction)
        stats = {}
        if a.stage == "validation":
            np.add.at(confusion, (payload["object_gt"], payload["object_pred"]), 1)
            np.add.at(counts, payload["object_gt"], 1)
        elif a.stage == "visual":
            image = ds[mapping[iid]][0]
            for mode, node in [("key", nodes[0]), ("unrelated", nodes[1])]:
                for strength in (.25, .5, 1.):
                    modified = mask(image, target.bbox.numpy()[node], strength)
                    if np.array_equal(np.asarray(image), np.asarray(modified)): raise RuntimeError("Image intervention no-op")
                    other, _, _, ix = capture(model, cfg, transform, ds, mapping[iid], modified)
                    value = (other["output"][1][0][ix, 1:].argmax(-1)+1).cpu().numpy()
                    name = "%s_%.2f" % (mode, strength)
                    stats[name] = dict(correct=int((value == relation_truth).sum()), clean_correct=int(hit.sum()), relations=len(hit))
                    payload[name] = value
                    del other
        else:
            for name, replacement, route, seed, fraction in conditions(labels, truth, confusion, counts, iid):
                altered = clean if replacement is None else replay.run(inputs, replacement, route)[0]
                if route == "embedding" and not torch.equal(altered[0][0], clean[0][0]): raise RuntimeError("Semantic route changed object output")
                value = (altered[1][0][pair_rows, 1:].argmax(-1)+1).cpu().numpy()
                corrected = value == relation_truth
                stats[name] = dict(correct=int(corrected.sum()), clean_correct=int(hit.sum()), relations=len(hit),
                    changed_nodes=int((replacement != labels).sum()) if replacement is not None else 0,
                    wrong_to_correct=int((corrected & ~hit).sum()), correct_to_wrong=int((~corrected & hit).sum()),
                    route=route, seed=seed, fraction=fraction)
                payload["prediction_" + name] = value
        if before != fingerprint(list(tensor_leaves(inputs))): raise RuntimeError("Input evidence mutated")
        row = dict(image_id=iid, protocol_sha256=reg, invariance_passed=True, stats=stats,
                   native_cache_max_error=error, native_replay_max_error=float((native-clean[1][0]).abs().max()),
                   relations=len(hit), objects=len(truth), nodes=nodes, feature_fingerprint=before)
        with output_path(out / "images" / (iid + ".npz")).open("wb") as handle: np.savez_compressed(handle, **payload)
        atomic_json(path, row); records.append(row)
        if len(records) % 10 == 0 or n == len(selected):
            elapsed = time.monotonic()-start; done = len(records) if a.stage == "visual" else n
            total = 200 if a.stage == "visual" else len(selected)
            value = dict(stage=a.stage, images=done, total=total, examined_candidates=n, seconds=elapsed, eta_seconds=elapsed/done*(total-done))
            atomic_json(out / "progress.json", value); print(json.dumps(value), flush=True)
        del inputs, holder, clean, noop, native, scores
    chunk.close()
    results = {}
    for name in records[0]["stats"]:
        stats = [r["stats"][name] for r in records]
        results[name] = paired_ratio([r["correct"] for r in stats], [r["clean_correct"] for r in stats], [r["relations"] for r in stats])
    if a.stage == "visual":
        for strength in (.25, .5, 1.):
            key, unrelated = "key_%.2f" % strength, "unrelated_%.2f" % strength
            results["key_minus_unrelated_%.2f" % strength] = paired_ratio(
                [r["stats"][key]["correct"] for r in records], [r["stats"][unrelated]["correct"] for r in records], [r["relations"] for r in records])
    if a.stage == "validation":
        np.savez_compressed(output_path(out / "confusion.npz"), confusion=confusion, counts=counts)
    atomic_json(out / "summary.json", dict(status="complete", images=len(records), relations=sum(r["relations"] for r in records),
        protocol_sha256=reg, invariance_passed=all(r["invariance_passed"] for r in records), results=results,
        excluded=excluded, matched_visual_target_met=len(records)==200 if a.stage=="visual" else None,
        no_retraining=True, evidence_scope="Reproduced plain Motifs SGCls bridge, not new universal causal claim",
        elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__": main()
