"""R14: fixed-checkpoint, validation-only audit of the extra PySGG pair cap."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, OLD, atomic_json, ensure_storage, output_path, sha256
from native_runtime import image_id, infer
from repro_experiment import build, dataset, recall_row
from repro_protocol import RUN, immutable, verify
from vb_native import compare_predictions

OUT = ROOT / "results/R14_pair_cap"
SOURCE_FOLDER = Path(__file__).resolve().parent
SPEC = dict(version="r14_validation_pair_cap_v1", tasks=["sgcls", "sgdet"],
            split="val", images=5000, arms=["pysgg_cap_2048", "reference_all_pairs"],
            union_chunk=256, bootstrap_trials=2000, invariant_tolerance=1e-4,
            selection="all validation images; no test evaluation or checkpoint selection",
            reference="Scene-Graph-Benchmark.pytorch sampling.prepare_test_pairs retains every non-self pair",
            scope="same model weights and detections; only candidate-pair truncation changes",
            boundary="execution and validation differences do not certify reference reproduction")


def concat_chunks(chunks):
    first = chunks[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(chunks, dim=0)
    if isinstance(first, tuple):
        return tuple(concat_chunks([x[j] for x in chunks]) for j in range(len(first)))
    raise TypeError("Unexpected union feature type")


class ChunkUnion:
    def __init__(self, module, chunk):
        self.module, self.chunk, self.original = module, chunk, module.forward
        module.forward = self.forward

    def forward(self, features, proposals, pairs=None):
        if self.module.training:
            raise RuntimeError("Chunking with training BatchNorm would change the method")
        if len(proposals) != 1 or pairs is None or len(pairs) != 1:
            raise RuntimeError("R14 requires image batch one")
        if len(pairs[0]) <= self.chunk:
            return self.original(features, proposals, pairs)
        return concat_chunks([self.original(features, proposals, [p])
                              for p in pairs[0].split(self.chunk)])

    def close(self):
        self.module.forward = self.original


def compare_shared(cap, full):
    for field in ["pred_labels", "pred_scores"]:
        a, b = cap.get_field(field), full.get_field(field)
        if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5):
            raise RuntimeError("Changing pair cap changed object outputs: " + field)
    if not torch.allclose(cap.bbox, full.bbox, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Changing pair cap changed boxes")
    a = cap.get_field("rel_pair_idxs").cpu().tolist()
    b = full.get_field("rel_pair_idxs").cpu().tolist()
    lookup = {tuple(pair): j for j, pair in enumerate(b)}
    if not all(tuple(pair) in lookup for pair in a):
        raise RuntimeError("Full-pair prediction did not contain every capped pair")
    index = torch.tensor([lookup[tuple(pair)] for pair in a], device=cap.bbox.device)
    error = float((cap.get_field("pred_rel_scores") - full.get_field("pred_rel_scores")[index]).abs().max())
    if error > SPEC["invariant_tolerance"]:
        raise RuntimeError("Shared-pair probabilities changed: %g" % error)
    return dict(shared_pairs=len(a), full_pairs=len(b), shared_probability_max_error=error)


def mean_recalls(rows, arm):
    r = np.asarray([row[arm]["R"] for row in rows], dtype=float)
    c = np.asarray([row[arm]["class_recall"] for row in rows], dtype=float)
    counts = np.isfinite(c).sum(0)
    macro = np.divide(np.nansum(c, axis=0), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0).mean(1)
    return r.mean(0), macro


def summary(rows, bootstrap=False):
    from repro_experiment import KS
    result = {}
    for arm in SPEC["arms"]:
        r, m = mean_recalls(rows, arm)
        result[arm] = dict(R=dict(zip(map(str, KS), r.tolist())), mR=dict(zip(map(str, KS), m.tolist())))
    result["images"] = len(rows)
    result["truncated_images"] = sum(x["pair_comparison"]["full_pairs"] > x["pair_comparison"]["shared_pairs"] for x in rows)
    result["max_shared_probability_error"] = max(x["pair_comparison"]["shared_probability_max_error"] for x in rows)
    if bootstrap:
        rng = np.random.default_rng(14001)
        r = [np.asarray([x[arm]["R"][KS.index(50)] for x in rows]) for arm in SPEC["arms"]]
        samples = [(r[1][idx] - r[0][idx]).mean() for idx in
                   (rng.integers(0, len(rows), len(rows)) for _ in range(SPEC["bootstrap_trials"]))]
        result["paired_R50_change"] = dict(value=float((r[1] - r[0]).mean()),
                                             bootstrap_95ci=np.quantile(samples, [.025, .975]).tolist(),
                                             unit="image", trials=SPEC["bootstrap_trials"])
    return result


def fingerprints():
    here = SOURCE_FOLDER
    ref = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch"
    native = OLD / "external/official_repos/PySGG/pysgg"
    files = [here / name for name in ["reproduction_pair_audit.py", "repro_experiment.py", "repro_protocol.py",
                                     "native_runtime.py", "vb_native.py", "vb_protocol.py", "common.py"]]
    files += [ref / "maskrcnn_benchmark/modeling/roi_heads/relation_head/sampling.py",
              native / "modeling/roi_heads/relation_head/sampling.py",
              native / "modeling/roi_heads/relation_head/inference.py",
              native / "modeling/roi_heads/relation_head/roi_relation_feature_extractors.py"]
    return {str(p): sha256(p) for p in files}


def evaluate(task):
    verify()
    out = OUT / task
    weights = RUN / task / "training/model_final.pth" if task == "sgdet" else None
    model, cfg, transform, provenance = build(task, out, weights=weights)
    model.eval()
    ds = dataset(cfg, "val")
    if len(ds) != SPEC["images"]:
        raise RuntimeError("Unexpected validation split cardinality")
    protocol = dict(spec=SPEC, sources=fingerprints(), checkpoint=provenance,
                    image_ids=[image_id(ds, i) for i in range(len(ds))])
    pf = out / "protocol.json"
    immutable(pf, protocol)
    digest = sha256(pf)
    relation = model.roi_heads.relation
    sampler = relation.samp_processor
    if sampler.max_proposal_pairs != 2048:
        raise RuntimeError("Expected original cap=2048")
    # Verify chunking is numerically neutral before the paired audit.
    noop = []
    for index in range(3):
        a, _ = infer(model, cfg, transform, ds, index)
        chunk = ChunkUnion(relation.union_feature_extractor, SPEC["union_chunk"])
        try:
            b, _ = infer(model, cfg, transform, ds, index)
            noop.append(compare_predictions(a, b))
        finally:
            chunk.close()
    atomic_json(out / "integration_check.json", dict(status="complete", chunk_noop_errors=noop,
                protocol_sha256=digest))
    chunk = ChunkUnion(relation.union_feature_extractor, SPEC["union_chunk"])
    rows = []
    started = time.monotonic()
    computed = 0
    from export_pysgg_vg_task import convert_prediction
    try:
        for index in range(len(ds)):
            iid = image_id(ds, index)
            path = out / "images" / (iid + ".json")
            if path.exists():
                row = json.loads(path.read_text())
                if row["protocol_sha256"] != digest:
                    raise RuntimeError("Stale R14 resume row")
                rows.append(row)
                continue
            gt = ds.get_groundtruth(index, evaluation=True)
            sampler.max_proposal_pairs = 2048
            cap, _ = infer(model, cfg, transform, ds, index)
            sampler.max_proposal_pairs = 1000000
            full, _ = infer(model, cfg, transform, ds, index)
            check = compare_shared(cap, full)
            row = dict(image_id=iid, protocol_sha256=digest, pair_comparison=check)
            for arm, pred in zip(SPEC["arms"], [cap, full]):
                value = recall_row(pred, gt, task, set())
                row[arm] = {key: value[key] for key in ["R", "class_recall", "relations"]}
                dest = output_path(out / "predictions" / arm / (iid + ".npz"))
                temp = dest.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, **convert_prediction(pred.to("cpu")))
                temp.replace(dest)
            prior = RUN / task / ("eval_retrained" if task == "sgdet" else "eval_corrected") / "val/images" / (iid + ".json")
            expected = json.loads(prior.read_text())
            if not np.allclose(row[SPEC["arms"][0]]["R"], expected["R"], atol=1e-10):
                raise RuntimeError("Capped control no longer matches R9 cached validation recall")
            atomic_json(path, row)
            rows.append(row)
            computed += 1
            if computed == 1 or (index + 1) % 25 == 0 or index + 1 == len(ds):
                elapsed = time.monotonic() - started
                progress = dict(images=index + 1, total=len(ds), seconds=elapsed,
                                eta_seconds=elapsed / computed * (len(ds) - index - 1),
                                detail="fixed weights; cap2048 versus all pairs; validation only")
                atomic_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if (index + 1) % 250 == 0:
                atomic_json(out / "partial_summary.json", dict(status="partial", **summary(rows)))
    finally:
        sampler.max_proposal_pairs = 2048
        chunk.close()
    atomic_json(out / "summary.json", dict(status="complete", protocol_sha256=digest,
                seconds=time.monotonic() - started, reference_reproduction_claim=False,
                **summary(rows, bootstrap=True)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=SPEC["tasks"])
    parser.add_argument("--gpu", required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    name = "R14_pair_cap_" + args.task
    state = ROOT / "status" / (name + ".json")
    lock = (ROOT / "status" / ("gpu%d.resource.lock" % args.gpu)).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = dict(name=name, pid=os.getpid(), gpu=[args.gpu], status="running",
                  command=list(__import__("sys").argv), progress_file=str(OUT / args.task / "progress.json"),
                  completion=str(OUT / args.task / "summary.json"), log=os.environ.get("SGG_JOB_LOG"))
    atomic_json(state, record)
    try:
        evaluate(args.task)
    except Exception as exc:
        atomic_json(state, dict(record, status="failed", reason=str(exc)))
        raise
    else:
        atomic_json(state, dict(record, status="complete"))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
