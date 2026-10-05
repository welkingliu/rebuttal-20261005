"""CPU-only R9 cache audit with the released official zero-shot denominator."""
import json
import os
from functools import reduce
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, OLD, ensure_storage, atomic_json, sha256
from native_runtime import configure, dataset, image_id
from repro_experiment import KS
from repro_protocol import RUN, immutable

SOURCE = Path(__file__).resolve()
OUT = ROOT / "results/R14_official_zeroshot"


def audit(task):
    from pysgg.data.datasets.evaluation.vg.sgg_eval import SGRecall
    out = OUT / task
    cfg = configure("transformer", task, out)
    ds = dataset(cfg, "test")
    source = RUN / task / ("eval_retrained" if task == "sgdet" else "eval_corrected") / "test"
    zero_path = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch"
    zero = {tuple(map(int, row)) for row in torch.load(str(zero_path), map_location="cpu").tolist()}
    protocol = dict(task=task, source_protocol_sha256=sha256(source / "protocol.json"),
                    official_zero_sha256=sha256(zero_path), code_sha256=sha256(SOURCE),
                    image_ids=[image_id(ds, i) for i in range(len(ds))], iou="inclusive pixel, >=0.5",
                    score_order="retain native cached triplet order, do not re-sort ties",
                    policy="recompute from unchanged cache; no new model fitting or checkpoint selection")
    pf = out / "protocol.json"
    immutable(pf, protocol)
    digest = sha256(pf)
    rows = []
    start = time.monotonic()
    for i in range(len(ds)):
        iid = image_id(ds, i)
        row_path = out / "images" / (iid + ".json")
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if row["protocol_sha256"] != digest:
                raise RuntimeError("Stale metric audit row")
            rows.append(row)
            continue
        gt = ds.get_groundtruth(i, evaluation=True)
        triples = gt.get_field("relation_tuple").numpy().astype(int)
        labels = gt.get_field("labels").numpy().astype(int)
        with np.load(source / "predictions" / (iid + ".npz")) as z:
            score = z["pred_entity_scores"][:, 1:]
            classes = score.argmax(1) + 1
            local = dict(pred_rel_inds=z["pred_rel_pairs"][:100], rel_scores=z["pred_rel_scores"][:100],
                         gt_rels=triples, gt_classes=labels, gt_boxes=gt.bbox.numpy(),
                         pred_classes=classes, obj_scores=score.max(1),
                         pred_boxes=z["pred_boxes"] * np.asarray([*gt.size, *gt.size], dtype=np.float32))
        collector = {}
        evaluator = SGRecall(collector)
        evaluator.register_container(task)
        collector[task + "_recall"] = {k: [] for k in KS}
        matches = evaluator.calculate_recall({"iou_thres": .5}, local, task)["pred_to_gt"] if len(local["pred_rel_inds"]) else []
        eligible = {j for j, (s, o, p) in enumerate(triples) if (int(labels[s]), int(labels[o]), int(p)) in zero}
        rr, zr, mr = [], [], []
        support = np.bincount(triples[:, 2], minlength=51)[1:]
        for k in KS:
            hit = reduce(np.union1d, matches[:k], np.array([], dtype=int)).astype(int)
            rr.append(len(hit) / len(triples))
            zr.append(len(set(hit) & eligible) / len(eligible) if eligible else None)
            count = np.bincount(triples[hit, 2], minlength=51)[1:]
            mr.append([float(h / n) if n else None for h, n in zip(count, support)])
        original = json.loads((source / "images" / (iid + ".json")).read_text())
        if original["protocol_sha256"] != protocol["source_protocol_sha256"]:
            raise RuntimeError("Original prediction protocol mismatch")
        delta = float(np.max(np.abs(np.asarray(rr) - original["R"])))
        if delta > 1e-10 or not np.allclose(np.asarray(mr, dtype=float), np.asarray(original["class_recall"], dtype=float), equal_nan=True, atol=1e-10):
            raise RuntimeError("CPU audit changed native R/mR for image " + iid)
        row = dict(image_id=iid, protocol_sha256=digest, R=rr, zR=zr, class_recall=mr,
                   zero_shot_relations=len(eligible), cached_native_recall_max_error=delta)
        atomic_json(row_path, row)
        rows.append(row)
        if (i + 1) % 250 == 0 or i == 0:
            elapsed = time.monotonic() - start
            p = dict(task=task, images=i + 1, total=len(ds), seconds=elapsed,
                     eta_seconds=elapsed / (i + 1) * (len(ds) - i - 1))
            atomic_json(OUT / "progress.json", p)
            print(json.dumps(p), flush=True)
    r = np.asarray([x["R"] for x in rows]).mean(0)
    z = np.asarray([x["zR"] for x in rows], dtype=float)
    counts = np.isfinite(z).sum(0)
    zr = [float(np.nansum(z[:, i]) / counts[i]) if counts[i] else None for i in range(len(KS))]
    c = np.asarray([x["class_recall"] for x in rows], dtype=float)
    n = np.isfinite(c).sum(0)
    mr = np.divide(np.nansum(c, 0), n, out=np.zeros_like(n, dtype=float), where=n > 0).mean(1)
    record = dict(status="complete", task=task, images=len(rows), protocol_sha256=digest,
                  R=dict(zip(map(str, KS), r.tolist())), mR=dict(zip(map(str, KS), mr.tolist())),
                  official_zR=dict(zip(map(str, KS), zr)),
                  official_zero_shot_relation_rows=sum(x["zero_shot_relations"] for x in rows),
                  official_zero_shot_images=int(counts[0]), R_mR_unchanged=True,
                  supersedes_only="R9 zero-shot metric contract; old results retained as reconstructed-set diagnostics")
    atomic_json(out / "summary.json", record)


def main():
    ensure_storage()
    torch.set_num_threads(2)
    state = ROOT / "status/R14_official_zeroshot.json"
    record = dict(status="running", pid=os.getpid(), gpu=[], command=[str(SOURCE)],
                  completion=str(OUT / "summary.json"), progress_file=str(OUT / "progress.json"),
                  log=str(ROOT / "logs/R14_official_zeroshot.log"))
    atomic_json(state, record)
    try:
        for task in ["sgcls", "sgdet"]:
            audit(task)
        atomic_json(OUT / "summary.json", dict(status="complete", tasks=["sgcls", "sgdet"]))
    except Exception as exc:
        atomic_json(state, dict(record, status="failed", reason=str(exc)))
        raise
    atomic_json(state, dict(record, status="complete"))


if __name__ == "__main__":
    main()
