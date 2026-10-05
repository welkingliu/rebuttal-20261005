"""V-D: finite shrinkage search, untouched full-system development and gate."""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from common import ROOT, atomic_json, ensure_storage, sha256
from extension_protocol import SPEC, verify, immutable
from native_runtime import dataset, image_id, infer
from vc_experiment import setup, protocol, load_features, checkpoint, save_npz
from vc_protocol import ResidualHead, choose_ids, summarize_rows, paired_gate
from vc_native import OfficialMetrics, targets, compare_predictions


VD = ROOT / "results/VD"


class ShrunkHead(nn.Module):
    def __init__(self, head, alpha):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Residual shrinkage must be in [0,1]")
        self.head, self.alpha = head, alpha

    def forward(self, features, logits):
        if self.alpha == 0:
            return logits
        return logits + self.alpha * (self.head(features, logits) - logits)


def shortlist():
    result = {}
    for task in SPEC["VD"]["tasks"]:
        out = VD / task
        x, logits, y = load_features(task, "development")
        positive = y > 0
        baseline = float(((logits[:, 1:].argmax(1) + 1)[positive] == y[positive]).float().mean())
        candidates = []
        for mode in SPEC["VD"]["heads"]:
            path = checkpoint(task, mode, 17)
            state = torch.load(str(path), map_location="cpu")
            if state["cache_protocol_sha256"] != sha256(ROOT / "cache/VC" / task / "protocol.json"):
                raise RuntimeError("V-C source cache changed")
            head = ResidualHead().eval(); head.load_state_dict(state["state"])
            with torch.no_grad():
                delta = head(x, logits) - logits
                for alpha in SPEC["VD"]["alphas"]:
                    updated = logits + alpha * delta
                    acc = float(((updated[:, 1:].argmax(1) + 1)[positive] == y[positive]).float().mean())
                    candidates.append(dict(mode=mode, alpha=alpha, accuracy=acc,
                                           gain=acc - baseline, source_sha256=sha256(path),
                                           name=mode + "_alpha" + str(alpha)))
        candidates.sort(key=lambda c: (-c["gain"], c["alpha"], c["mode"]))
        selected = [c for c in candidates if c["gain"] >= SPEC["VD"]["cached_min_gain"]][:2]
        record = dict(status="complete", task=task, baseline_accuracy=baseline,
                      positive_objects=int(positive.sum()), candidates=candidates, selected=selected,
                      note="Cached pre-NMS selection only; not an end-to-end improvement claim")
        immutable(out / "shortlist.json", record)
        result[task] = record
    atomic_json(VD / "shortlist.json", dict(status="complete", tasks=result))
    print(json.dumps({t: r["selected"] for t, r in result.items()}), flush=True)


def frozen_splits(task, ds):
    all_ids = {image_id(ds, i) for i in range(len(ds))}
    previous = json.loads((ROOT / "cache/VB" / task / "protocol.json").read_text())
    vc = protocol(task)
    excluded = set(previous["development"] + previous["gate"] + vc["development"] + vc["gate"])
    pool = all_ids - excluded
    dev = choose_ids(pool, 500, "VD_development:")
    gate = choose_ids(pool - set(dev), 1000, "VD_gate:")
    if len(dev) != 500 or len(gate) != 1000 or set(dev) & set(gate):
        raise RuntimeError("Insufficient unused native validation images")
    value = dict(development=dev, gate=gate, excluded=sorted(excluded),
                 source_vc_sha256=sha256(ROOT / "cache/VC" / task / "protocol.json"))
    immutable(VD / task / "splits.json", value)
    return value


def evaluate(task, split, candidates):
    model, cfg, transform, provenance, patch = setup(task, VD / task / "runtime")
    ds = dataset(cfg, "val")
    lookup = {image_id(ds, i): i for i in range(len(ds))}
    ids = frozen_splits(task, ds)[split]
    results = {}
    for candidate in [None] + candidates:
        name = "native" if candidate is None else candidate["name"]
        out = VD / task / split / name
        patch.enabled = candidate is not None
        patch.head = None
        if candidate:
            path = checkpoint(task, candidate["mode"], 17)
            if sha256(path) != candidate["source_sha256"]:
                raise RuntimeError("Candidate checkpoint changed after selection")
            head = ResidualHead().cuda().eval()
            head.load_state_dict(torch.load(str(path), map_location="cpu")["state"])
            patch.head = ShrunkHead(head, candidate["alpha"]).cuda().eval()
        pf = out / "protocol.json"
        immutable(pf, dict(task=task, split=split, image_ids=ids, candidate=candidate, model=provenance,
                           downstream="native context, predicates, class-NMS and ranking are rerun; no rank freezing"))
        metrics = OfficialMetrics(task); rows = []; start = time.monotonic()
        for n, iid in enumerate(ids):
            dest = out / "images" / (iid + ".json")
            if dest.exists():
                row = json.loads(dest.read_text())
                if row["protocol_sha256"] != sha256(pf):
                    raise RuntimeError("Stale V-D image record")
                rows.append(row); continue
            pred, _ = infer(model, cfg, transform, ds, lookup[iid])
            if n == 0 and candidate is None:
                patch.head = ShrunkHead(ResidualHead().cuda().eval(), 0.)
                patch.enabled = True
                same, _ = infer(model, cfg, transform, ds, lookup[iid])
                error = compare_predictions(pred, same)
                atomic_json(out / "zero_check.json", dict(max_error=error))
                patch.enabled = False; patch.head = None
            gt = ds.get_groundtruth(lookup[iid], evaluation=True)
            row = metrics.row(iid, pred, gt, patch.capture, targets(patch.capture, gt, task))
            row["protocol_sha256"] = sha256(pf)
            # Retain final predictions so no later audit depends on an aggregate only.
            save_npz(out / "predictions" / (iid + ".npz"), boxes=pred.bbox.cpu().numpy(),
                     labels=pred.get_field("pred_labels").cpu().numpy(),
                     scores=pred.get_field("pred_scores").cpu().numpy(),
                     pairs=pred.get_field("rel_pair_idxs").cpu().numpy(),
                     relation_scores=pred.get_field("pred_rel_scores").cpu().numpy())
            atomic_json(dest, row); rows.append(row)
            if (n + 1) % 25 == 0:
                value = dict(task=task, split=split, candidate=name, images=n + 1, total=len(ids), seconds=time.monotonic() - start)
                atomic_json(out / "progress.json", value); print(json.dumps(value), flush=True)
        atomic_json(out / "summary.json", dict(status="complete", **summarize_rows(rows)))
        results[name] = rows
    patch.close()
    return results


def task_run(task):
    shortlist_record = json.loads((VD / task / "shortlist.json").read_text())
    candidates = shortlist_record["selected"]
    if not candidates:
        atomic_json(VD / task / "summary.json", dict(status="complete", accepted=False,
                    reason="No cached development candidate improved identity by 0.2pp; stopped before new validation or test"))
        return
    rows = evaluate(task, "development", candidates)
    base = summarize_rows(rows["native"])
    eligible = []; reports = []
    for candidate in candidates:
        result = summarize_rows(rows[candidate["name"]])
        gain = result["post_nms_object_top1"] - base["post_nms_object_top1"]
        rd, md = result["R"]["50"] - base["R"]["50"], result["mR"]["50"] - base["mR"]["50"]
        good = gain >= .005 and rd >= -.005 and md >= -.005
        reports.append(dict(candidate=candidate, identity_gain=gain, R50_delta=rd, mR50_delta=md, eligible=good))
        if good:
            eligible.append((gain, candidate))
    eligible.sort(key=lambda p: (-p[0], p[1]["alpha"], p[1]["mode"]))
    winner = eligible[0][1] if eligible else None
    immutable(VD / task / "selection.json", dict(reports=reports, selected=winner))
    if winner is None:
        atomic_json(VD / task / "summary.json", dict(status="complete", accepted=False,
                    reason="No full-system development candidate met identity and relation constraints; gate and test untouched"))
        return
    gate = evaluate(task, "gate", [winner])
    decision = paired_gate(gate["native"], gate[winner["name"]])
    atomic_json(VD / task / "summary.json", dict(status="complete", accepted=decision["accepted"],
                candidate=winner, decision=decision, formal_test_run=False,
                note="Independent validation gate, single exploratory seed; not a formal test result"))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["shortlist", "task", "report"])
    p.add_argument("--task", choices=["sgcls", "sgdet"])
    args = p.parse_args(); ensure_storage(); verify(); torch.set_num_threads(4)
    if args.stage == "shortlist":
        shortlist()
    elif args.stage == "task":
        if args.task is None:
            p.error("task stage requires --task")
        task_run(args.task)
    else:
        results = {t: json.loads((VD / t / "summary.json").read_text()) for t in SPEC["VD"]["tasks"]}
        atomic_json(VD / "summary.json", dict(status="complete", accepted=all(r["accepted"] for r in results.values()),
                    tasks=results, additional_seeds_run=False, test_accessed=False,
                    next_step="Only a passing independent pilot can justify a separately registered confirmation; no automatic expansion"))


if __name__ == "__main__":
    main()
