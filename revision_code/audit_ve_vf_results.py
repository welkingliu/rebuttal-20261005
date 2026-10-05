"""Post-run read-only checks and paired uncertainty for the V-E/V-F pilots."""
import json

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, sha256
from vc_protocol import KS, summarize_rows, paired_gate
from ve_protocol import verify as verify_ve
from vf_protocol import verify as verify_vf

OUT = ROOT / "results/VE_VF_postrun_20261002"


def read(path):
    return json.loads(path.read_text())


def bootstrap(base, changed, samples=2000, seed=20261002):
    n = len(base)
    count = np.array([r["positive_objects"] for r in base])
    if not np.array_equal(count, [r["positive_objects"] for r in changed]):
        raise RuntimeError("Object denominator changed")
    post = np.array([b["post_nms_correct"] - a["post_nms_correct"] for a, b in zip(base, changed)])
    pre = np.array([b["positive_correct"] - a["positive_correct"] for a, b in zip(base, changed)])
    r = np.array([b["recalls"][KS.index(50)] - a["recalls"][KS.index(50)] for a, b in zip(base, changed)])
    acls = np.array([a["class_recalls"][KS.index(50)] for a in base], dtype=float)
    bcls = np.array([b["class_recalls"][KS.index(50)] for b in changed], dtype=float)
    if not np.array_equal(np.isfinite(acls), np.isfinite(bcls)):
        raise RuntimeError("Predicate support changed")
    support = np.isfinite(acls)
    delta = np.nan_to_num(bcls - acls)

    def metrics(ix):
        total = count[ix].sum()
        counts = support[ix].sum(0)
        mr = np.divide(delta[ix].sum(0), counts, out=np.zeros(50), where=counts > 0).mean()
        return [post[ix].sum() / total, pre[ix].sum() / total, r[ix].mean(), mr]

    names = ["post_nms_identity", "pre_nms_identity", "R50", "mR50"]
    rng = np.random.default_rng(seed)
    draws = np.array([metrics(rng.integers(0, n, n)) for _ in range(samples)])
    limits = np.quantile(draws, [.025, .975], axis=0)
    return dict(images=n, objects=int(count.sum()),
                correct_object_net_change_post_nms=int(post.sum()),
                correct_object_net_change_pre_nms=int(pre.sum()),
                deltas={name: dict(point=point, paired_image_95_ci=limits[:, k].tolist())
                        for k, (name, point) in enumerate(zip(names, metrics(np.arange(n))))},
                samples=samples, seed=seed, units="fraction; multiply by100 for percentage points",
                scope="post-hoc descriptive paired CIs, not a changed acceptance rule; single seed")


def rows_for(task, mode, ids):
    folder = ROOT / "results/VE" / task / "gate" / (mode + "_seed17")
    protocol = read(folder / "protocol.json")
    if protocol["image_ids"] != ids:
        raise RuntimeError("Wrong evaluation image order")
    expected = set(ids)
    if {p.stem for p in (folder / "images").glob("*.json")} != expected:
        raise RuntimeError("Incomplete image rows")
    if {p.stem for p in (folder / "predictions").glob("*.npz")} != expected:
        raise RuntimeError("Incomplete prediction cache")
    rows = [read(folder / "images" / (i + ".json")) for i in ids]
    if any(r["image_id"] != iid or r["protocol_sha256"] != sha256(folder / "protocol.json") for iid, r in zip(ids, rows)):
        raise RuntimeError("Image/protocol mismatch")
    result = summarize_rows(rows)
    saved = read(folder / "summary.json")
    for k in ["object_top1", "post_nms_object_top1", "ece", "foreground_nll", "foreground_brier"]:
        if not np.isclose(result[k], saved[k], atol=1e-12, rtol=0):
            raise RuntimeError("Saved scalar differs from per-image records: " + k)
    for k in ["R", "mR"]:
        if any(not np.isclose(result[k][str(n)], saved[k][str(n)], atol=1e-12, rtol=0) for n in KS):
            raise RuntimeError("Saved recall differs from per-image records")
    if mode != "native":
        path = ROOT / "checkpoints/VE" / task / (mode + "_seed17.pth")
        if sha256(path) != protocol["adapter_sha256"]:
            raise RuntimeError("Evaluated adapter hash mismatch")
    return rows, result


def main():
    ensure_storage(); torch.set_num_threads(2)
    registrations = dict(VE=verify_ve(), VF=verify_vf())
    ve = {}
    decision = read(ROOT / "results/VE/pilot_decision.json")
    for task in ["sgcls", "sgdet"]:
        ids = read(ROOT / "results/VE" / task / "splits.json")["gate"]
        rows, stats = {}, {}
        for mode in ["native", "supervised", "relation_aware"]:
            rows[mode], stats[mode] = rows_for(task, mode, ids)
        comparisons = {}
        for mode in ["supervised", "relation_aware"]:
            actual_gate = paired_gate(rows["native"], rows[mode])
            if actual_gate != decision["tasks"][task][mode]:
                raise RuntimeError("Stored gate does not reproduce")
            comparisons[mode + "_minus_native"] = bootstrap(rows["native"], rows[mode])
        comparisons["relation_aware_minus_supervised"] = bootstrap(rows["supervised"], rows["relation_aware"])
        ve[task] = dict(metrics=stats, comparisons=comparisons, row_and_prediction_coverage_per_arm=len(ids),
                        note="SGDet identities are conditional on positive input proposals, not AP or unique-GT accuracy")
        print(json.dumps(dict(task=task, comparisons=comparisons)), flush=True)
    vf = {}
    selected = read(ROOT / "results/VF/sgcls/splits.json")
    for split in ["train", "development"]:
        files = list((ROOT / "cache/VF/sgcls" / split).glob("*.pt"))
        if {p.stem for p in files} != set(selected[split]):
            raise RuntimeError("Incomplete V-F feature cache")
    for mode in ["supervised", "score_protected"]:
        folder = ROOT / "results/VF/sgcls/training" / mode
        summary = read(folder / "summary.json")
        path = ROOT / "checkpoints/VF/sgcls" / (mode + "_seed17.pth")
        payload = torch.load(str(path), map_location="cpu")
        if sha256(path) != summary["checkpoint_sha256"] or payload["epoch"] != summary["selected_epoch"]:
            raise RuntimeError("V-F selected checkpoint mismatch")
        optimum = max(summary["history"], key=lambda h: (h["development"]["post_nms_top1"], -h["development"]["foreground_nll"]))
        if optimum["epoch"] != summary["selected_epoch"] or payload["development"] != summary["development"]:
            raise RuntimeError("V-F checkpoint selection did not follow registered rule")
        if not any(bool(x.abs().sum() > 0) for k, x in payload["head"].items() if k.startswith("output.")):
            raise RuntimeError("V-F output head was never updated")
        vf[mode] = dict(selected_epoch=summary["selected_epoch"], completed_epochs=len(summary["history"]),
                        history=summary["history"], selected_development=summary["development"],
                        classifier_update_verified=True)
    absent = not (ROOT / "results/VF/sgcls/gate").exists() and not (ROOT / "results/VF/sgdet").exists()
    if not absent:
        raise RuntimeError("Unexpected V-F expansion after development rejection")
    result = dict(status="complete", registrations=registrations, VE=ve, VF=vf,
                  vf_fresh_gate_not_evaluated=True, vf_sgdet_not_started=True,
                  no_new_training=True, paired_bootstrap_samples=2000,
                  primary_decisions_unchanged=True,
                  scientific_scope="exploratory seed17 validation pilots, not full-test mitigation evidence")
    atomic_json(OUT / "summary.json", result)
    print("[VERIFIED] " + str(OUT / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
