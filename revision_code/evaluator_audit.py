"""R1: independent official/unified graph-constrained evaluation on the same cache."""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from common import ROOT, OLD, ensure_storage, atomic_json, sha256
from native_runtime import assets, configure, dataset, image_id
from sgg_core.audits.standard_sgg_eval import _build_ranked_triplets, _ground_truth, _matched_gt_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["transformer", "tde_motifs"], required=True)
    parser.add_argument("--task", choices=["predcls", "sgcls", "sgdet"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    out = ROOT / "results/R1" / args.family / (args.task + ("_smoke%d" % args.limit if args.limit else ""))
    cfg = configure(args.family, args.task, out)
    ds = dataset(cfg, "test")
    manifest, _, _, _, cache = assets(args.family, args.task)
    from pysgg.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGMeanRecall
    result = {}
    official = SGRecall(result)
    mean = SGMeanRecall(result, 51, ds.ind_to_predicates)
    official.register_container(args.task)
    mean.register_container(args.task)
    ks = [1, 5, 10, 20, 50, 100]
    result[args.task + "_recall"] = {k: [] for k in ks}
    result[args.task + "_mean_recall"] = {k: 0. for k in ks}
    result[args.task + "_mean_recall_collect"] = {k: [[] for _ in range(51)] for k in ks}
    result[args.task + "_mean_recall_list"] = {k: [] for k in ks}
    recalls, per_class = defaultdict(list), {k: defaultdict(list) for k in ks}
    mismatches, ids = [], []
    limit = args.limit or len(ds)
    start = time.monotonic()
    for i in range(min(limit, len(ds))):
        iid = image_id(ds, i)
        ids.append(iid)
        gt = ds.get_groundtruth(i, evaluation=True)
        relations = gt.get_field("relation_tuple").long()
        labels = gt.get_field("labels").long()
        w, h = gt.size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        batch = dict(boxes=gt.bbox.float() / scale, entity_labels=labels,
                     rel_pairs=relations[:, :2], rel_labels=relations[:, 2])
        path = cache / "predictions" / args.task / (iid + ".npz")
        with np.load(path, allow_pickle=False) as z:
            pred = {k: torch.from_numpy(z[k].copy()) for k in z.files if z[k].dtype.kind in "fiub"}
        entity = pred["pred_entity_scores"]
        if not torch.allclose(entity.sum(1), torch.ones(len(entity)), atol=1e-3):
            raise RuntimeError("Cache entity scores not categorical probabilities")
        scores, classes = entity[:, 1:].max(1)
        classes += 1
        boxes = pred["pred_boxes"] * scale
        if args.task in ("predcls", "sgcls"):
            if pred["pred_boxes"].shape != batch["boxes"].shape:
                raise RuntimeError("GT entity cardinality mismatch")
            boxes = gt.bbox.float()
        if args.task == "predcls":
            classes, scores = labels, torch.ones(len(labels))
        pairs, rel_scores = pred["pred_rel_pairs"].long(), pred["pred_rel_scores"].float()
        rank_scores = scores[pairs[:, 0]] * scores[pairs[:, 1]] * rel_scores[:, 1:].max(1)[0]
        order = torch.argsort(rank_scores, descending=True)[:100]
        local = dict(pred_rel_inds=pairs[order].numpy(), rel_scores=rel_scores[order].numpy(),
                     gt_rels=relations.numpy(), gt_classes=labels.numpy(), gt_boxes=gt.bbox.numpy(),
                     pred_classes=classes.numpy(), pred_boxes=boxes.numpy(), obj_scores=scores.numpy())
        local = official.calculate_recall({"iou_thres": .5}, local, args.task)
        mean.collect_mean_recall_items({"iou_thres": .5}, local, args.task)
        ranked = _build_ranked_triplets(pred, batch, args.task, True).top(100)
        g = _ground_truth(batch)
        for k in ks:
            hit = _matched_gt_indices(ranked.top(k), g, .5)
            value = len(hit) / len(relations)
            recalls[k].append(value)
            for cls in relations[:, 2].unique().tolist():
                idx = set((relations[:, 2] == cls).nonzero().flatten().tolist())
                per_class[k][cls].append(len(hit & idx) / len(idx))
            ref = result[args.task + "_recall"][k][-1]
            if abs(ref - value) > 1e-8:
                mismatches.append(dict(image_id=iid, k=k, official=ref, unified=value))
        if (i + 1) % 100 == 0:
            progress = dict(images=i + 1, total=min(limit, len(ds)), seconds=time.monotonic() - start,
                            mismatch_rows=len(mismatches))
            atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    mean.calculate_mean_recall(args.task)
    metrics = {}
    for k in ks:
        r = float(np.mean(result[args.task + "_recall"][k]))
        mr = float(result[args.task + "_mean_recall"][k])
        u = float(np.mean(recalls[k]))
        um = float(np.mean([np.mean(per_class[k][c]) if c in per_class[k] else 0 for c in range(1, 51)]))
        metrics[str(k)] = dict(official_R=r, unified_R=u, delta_R=u-r,
                              official_image_macro_mR=mr, unified_image_macro_mR=um, delta_mR=um-mr)
    reference = {k: v for k,v in manifest["reference_metrics"].items() if k.lower().startswith(args.task + "/")}
    checks = {}
    for key, value in reference.items():
        metric = key.split("/")[1]
        k = metric.split("@")[1]
        actual = metrics[k]["official_R" if metric.startswith("R@") else "official_image_macro_mR"]
        checks[key] = dict(reference=value, official_actual=actual, delta=actual-value,
                           passes_existing_tolerance=abs(actual-value) <= manifest["reproduction_tolerance"])
    atomic_json(out / "summary.json", dict(status="complete", family=args.family, task=args.task,
        images=len(ids), smoke=bool(args.limit), cache=str(cache), cache_metadata_sha256=sha256(cache/"metadata.json"),
        metrics=metrics, reproduction_checks=checks, mismatch_rows=len(mismatches), mismatch_examples=mismatches[:100],
        protocol_note="Official uses inclusive pixel IoU; unified uses normalized continuous IoU. mR here uses the official image/class macro definition over all 50 predicates.",
        canonical_test_images=26446, full_split=len(ids)==26446, image_ids=ids))
    print("[COMPLETE] R1", args.family, args.task, flush=True)


if __name__ == "__main__":
    main()
