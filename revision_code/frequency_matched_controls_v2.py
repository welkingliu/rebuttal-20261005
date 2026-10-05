"""Bounded post-hoc GT-vs-random frequency-input diagnostic; no training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def replacements(labels, truth, counts, seed, image):
    labels, truth, counts = map(np.asarray, (labels, truth, counts))
    changed = labels != truth
    result = labels.copy()
    ranks = np.argsort(np.argsort(-counts[1:], kind="stable"), kind="stable")
    strata = np.full(len(counts), -1, dtype=int)
    strata[1:] = ranks // 30
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        (str(seed) + ":frequency-repair:" + str(image)).encode()).digest()[:4], "big"))
    for i in np.flatnonzero(changed):
        pool = np.array([c for c in range(1, len(counts))
                         if c not in (int(labels[i]), int(truth[i]))
                         and strata[c] == strata[truth[i]]])
        if not len(pool):
            raise RuntimeError("No matched wrong-label control; do not relax strata")
        result[i] = rng.choice(pool)
    if not np.array_equal(result != labels, changed):
        raise RuntimeError("Changed positions/count differ from GT intervention")
    return result


def main():
    import torch
    from common import ROOT, ensure_storage, atomic_json, sha256
    from evidence_completion import lock, paired_ratio
    from identity_intervention import SemanticReplay, fingerprint, tensor_leaves
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["motifs", "transformer"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_storage()
    root = ROOT / "results/R23_frequency_matched_v2"
    out = root / args.family / ("smoke" if args.smoke else "formal")
    if not args.smoke:
        check = json.loads((root / args.family / "smoke/summary.json").read_text())
        if check["status"] != "complete":
            raise RuntimeError("Smoke must pass before full run")
    torch.set_num_threads(2)
    seed = 666 if args.family == "motifs" else 17
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    if args.family == "motifs":
        import r20_motifs_native_bridge as bridge
        import r20_motifs_bridge as base
        from maskrcnn_benchmark.data.transforms import build_transforms
        from reproduction_pair_audit import ChunkUnion
        source = ROOT / "results/R20_plain_motifs_native_bridge/identity"
        validation = source.parent / "validation/confusion.npz"
        model, cfg = base.build("sgcls", out, base.RUN / "sgcls/formal/best.pth")
        transform = build_transforms(cfg, is_train=False)
        chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
        ds = base.dataset(cfg, "test")
    else:
        import r15_identity as bridge
        import sgdet_identity as base
        from repro_experiment import dataset
        bridge.verify()
        source = ROOT / "results/R15_transformer_identity/transformer/test"
        validation = source.parent / "validation/confusion.npz"
        model, cfg, transform, provenance = bridge.load("transformer", out)
        base.infer = bridge.checked_infer
        ds = dataset(cfg, "test")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    previous = json.loads((source / "protocol.json").read_text())
    ids = previous["image_ids"][:3] if args.smoke else previous["image_ids"]
    read = lambda path: json.loads(path.read_text())
    if read(source / "summary.json")["status"] != "complete":
        raise RuntimeError("Incomplete parent experiment")
    with np.load(validation) as z:
        counts = z["counts"]
    protocol = dict(version="frequency_matched_wrong_nodes_v2", family=args.family,
                    image_ids=ids, seeds=[17, 23, 31], strength=1,
                    parent_protocol_sha256=sha256(source / "protocol.json"),
                    validation_counts_sha256=sha256(validation),
                    sources={name: sha256(HERE / name) for name in
                             ['frequency_matched_controls_v2.py', 'r20_motifs_native_bridge.py',
                              'r20_motifs_bridge.py', 'r15_identity.py', 'sgdet_identity.py',
                              'identity_intervention.py', 'evidence_completion.py']},
                    amendment="v1 stopped on singleton bin when background was ranked; v2 ranks only 150 foreground classes; no outcomes used to choose bins",
                    intervention="Only matched originally wrong nodes; same nodes as GT replacement",
                    random="Five bins of 30 foreground classes from validation frequency; exclude background/native/GT",
                    inference="Fixed visual inputs and candidate pairs; original frequency route",
                    statistics="10000 image-bootstrap draws; seed17; no multiplicity correction",
                    scope="Post-hoc diagnostic on previously inspected test images, not repair or confirmation")
    digest = lock(out / "protocol.json", protocol)
    mapping = {Path(p).stem: i for i, p in enumerate(ds.filenames)}
    replay = SemanticReplay(model.roi_heads.relation.predictor)
    records = []
    start = time.monotonic()
    for number, iid in enumerate(ids, 1):
        path = out / "images" / (iid + ".json")
        if path.exists():
            row = read(path)
            if row["protocol_sha256"] != digest:
                raise RuntimeError("Resume mismatch")
            records.append(row)
            continue
        with np.load(source / "images" / (iid + ".npz")) as z:
            prior = {key: z[key] for key in z.files}
        index = mapping[iid]
        if args.family == "motifs":
            holder, target, gt, rows = bridge.capture(model, cfg, transform, ds, index)
            gt_labels = target.get_field("labels").cpu().numpy()
            if not np.array_equal(gt_labels, prior["object_gt"]):
                raise RuntimeError("GT labels differ from parent")
        else:
            holder, native = base.capture(model, cfg, transform, ds, index)
            rows = torch.as_tensor(prior["native_candidate_rows"], device="cuda")
            if not np.array_equal(holder["inputs"][1][0].cpu().numpy(), prior["native_candidate_pairs"]):
                raise RuntimeError("SGDet candidates differ from parent")
        inputs = holder["inputs"]
        before = fingerprint(list(tensor_leaves(inputs)))
        clean, labels = replay.run(inputs)
        if not torch.allclose(clean[1][0], holder["output"][1][0], atol=1e-5, rtol=1e-5):
            raise RuntimeError("Native replay mismatch")
        if not np.array_equal(labels.cpu().numpy(), prior["object_pred"]):
            raise RuntimeError("Object labels differ from parent")
        clean_pred = (clean[1][0][rows, 1:].argmax(1) + 1).cpu().numpy()
        if not np.array_equal(clean_pred, prior["clean_prediction"]):
            raise RuntimeError("Clean predicate labels differ from parent")
        if args.family == "motifs":
            base.check_native_cache(iid, prior["pairs"], clean[1][0][rows])
        noop, _ = replay.run(inputs, labels, "frequency")
        if not torch.equal(noop[1][0], clean[1][0]):
            raise RuntimeError("Frequency no-op mismatch")
        truth = prior["object_gt"]
        interventions = {"oracle": truth}
        for rs in (17, 23, 31):
            interventions["random_s%d" % rs] = replacements(prior["object_pred"], truth, counts, rs, iid)
        predictions, correct = {}, {"clean": int((clean_pred == prior["relation_gt"]).sum())}
        for name, values in interventions.items():
            altered, _ = replay.run(inputs, torch.as_tensor(values, device=labels.device), "frequency")
            if not torch.allclose(altered[0][0].sort(1)[0], clean[0][0].sort(1)[0], atol=1e-5, rtol=1e-5):
                raise RuntimeError("Confidence multiset changed")
            pred = (altered[1][0][rows, 1:].argmax(1) + 1).cpu().numpy()
            predictions[name] = pred
            correct[name] = int((pred == prior["relation_gt"]).sum())
        key = "prediction_oracle_frequency"
        if not np.array_equal(predictions["oracle"], prior[key]):
            raise RuntimeError("Oracle differs from parent")
        if before != fingerprint(list(tensor_leaves(inputs))):
            raise RuntimeError("Fixed input mutated")
        row = dict(image_id=iid, protocol_sha256=digest, parent_npz_sha256=sha256(source / "images" / (iid + ".npz")),
                   relations=len(prior["relation_gt"]), correct=correct,
                   changed_nodes=int((truth != prior["object_pred"]).sum()), invariance_passed=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), relation_gt=prior["relation_gt"],
                            object_native=prior["object_pred"], **{"labels_"+k:v for k,v in interventions.items()},
                            **predictions)
        atomic_json(path, row)
        records.append(row)
        if number % 10 == 0 or number == len(ids):
            progress = dict(status="running", images=number, total=len(ids), seconds=time.monotonic()-start)
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        del holder, inputs, clean, altered, noop
    eligible = [r for r in records if r["relations"] > 0]
    contrasts = {}
    for rs in (17, 23, 31):
        for first, second in [("oracle", "random_s%d" % rs), ("random_s%d" % rs, "clean")]:
            contrasts[first+"_minus_"+second] = paired_ratio(
                [r["correct"][first] for r in eligible], [r["correct"][second] for r in eligible],
                [r["relations"] for r in eligible], draws=10000, seed=17)
    atomic_json(out / "summary.json", dict(status="complete", images=len(records),
                evaluable_images=len(eligible), relations=sum(r["relations"] for r in records),
                protocol_sha256=digest, contrasts=contrasts, invariance_passed=True,
                independent_confirmation=False, seconds=time.monotonic()-start))
    print("[COMPLETE] " + str(out), flush=True)


if __name__ == "__main__":
    main()
