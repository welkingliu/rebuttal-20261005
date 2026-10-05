"""V-G development-only feasibility: train-memory evidence for selective identity repair."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, sha256
from repro_experiment import save_torch
from repro_protocol import immutable
from vf_protocol import verify as verify_vf

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VG_retrieval_development"
STATUS = ROOT / "status/VG_retrieval_development.json"
SPEC = dict(version="vg_train_memory_selective_identity_v1", family="transformer", task="sgcls",
    train_images=5000, development_images=500, seed=17,
    features="frozen native 4096-D ROI features only; normalized; no contextual feature retrieval",
    bank="at most256 objects per GT class; deterministic image/object hash; training images only",
    k=32, temperature=.07, per_class_cap=256,
    alphas=[.1, .25, .5], uncertainty_thresholds=[.5, .7], neighbor_consensus_min=.5,
    correction="modify only uncertain predictions where weighted neighbor majority disagrees; preserve background mass",
    scores="convex mixture of native foreground posterior and retrieval posterior; new scores must be used downstream",
    controls="native, retrieval-only foreground classifier, nonselective alpha0.25 fusion",
    selection="six fixed candidates; highest development Top1, then fewer damaged objects, then lower alpha/threshold",
    requirements=dict(identity_gain_min=.005, correction_precision_min=.6, macro_accuracy_not_lower=True),
    gate_policy="no fresh gate or test is read by this feasibility program; passing is not mitigation success",
    expansion="full native SGCls/SGDet and original relation noninferiority gates still required before any efficacy claim",
    caveat="exploratory cached-feature retrieval inspired by retrieval adapters, not a Tip-Adapter reproduction",
    reference="https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/154_ECCV_2022_paper.php")


def read(path):
    return json.loads(path.read_text())


def nearest_posterior(features, keys, labels, k=32, temperature=.07):
    features = F.normalize(features[:, :4096].float(), dim=-1)
    scores, ids = (features @ keys.t()).topk(min(k, len(keys)), dim=-1)
    weights = (scores / temperature).softmax(-1)
    posterior = features.new_zeros((len(features), 151))
    posterior.scatter_add_(1, labels[ids], weights)
    return posterior


def fusion(logits, retrieval, alpha, threshold, consensus=.5, selective=True):
    if not 0 <= alpha <= 1:
        raise ValueError("Invalid probability mixing weight")
    p = logits.softmax(-1)
    mass = p[:, 1:].sum(-1, keepdim=True)
    native_conf, native_class = (p[:, 1:] / mass.clamp_min(1e-12)).max(-1)
    memory_conf, memory_class = retrieval[:, 1:].max(-1)
    eligible = ((native_conf < threshold) & (memory_conf >= consensus) & (memory_class != native_class)
                if selective else torch.ones(len(p), dtype=torch.bool, device=p.device))
    q = p.clone()
    q[:, 1:] = (1 - alpha) * p[:, 1:] + alpha * mass * retrieval[:, 1:]
    q = torch.where(eligible[:, None], q, p)
    # Return original logits exactly on untouched proposals and for alpha=0.
    updated = torch.where(eligible[:, None], q.clamp_min(1e-12).log(), logits) if alpha else logits
    return updated, eligible


def record(iid, logits, baseline, y, eligible=None):
    prob = logits.softmax(-1)
    pred = prob[:, 1:].argmax(-1) + 1
    before = baseline[:, 1:].argmax(-1) + 1
    old, new = before == y, pred == y
    bins = torch.zeros((15, 3), device=logits.device)
    confidence = prob[:, 1:].max(-1)[0]
    for i in range(15):
        select = torch.clamp((confidence * 15).long(), max=14) == i
        bins[i] = torch.stack([select.sum(), confidence[select].sum(), new[select].sum()])
    return dict(image_id=iid, objects=len(y), baseline_correct=int(old.sum()), correct=int(new.sum()),
        repaired=int((~old & new).sum()), damaged=int((old & ~new).sum()),
        label_flips=int((before != pred).sum()), eligible=int(eligible.sum()) if eligible is not None else len(y),
        class_count=torch.bincount(y, minlength=151)[1:].cpu().tolist(),
        class_correct=torch.bincount(y[new], minlength=151)[1:].cpu().tolist(),
        nll_sum=float(F.cross_entropy(logits, y, reduction="sum")), calibration_bins=bins.cpu().tolist())


def summarize(rows):
    out = {k: sum(r[k] for r in rows) for k in ["objects", "baseline_correct", "correct", "repaired", "damaged", "label_flips", "eligible", "nll_sum"]}
    counts = np.sum([r["class_count"] for r in rows], axis=0)
    correct = np.sum([r["class_correct"] for r in rows], axis=0)
    bins = np.sum([r["calibration_bins"] for r in rows], axis=0)
    supported = counts > 0
    out.update(top1=out["correct"] / out["objects"], delta=(out["correct"] - out["baseline_correct"]) / out["objects"],
        macro_accuracy=float((correct[supported] / counts[supported]).mean()),
        correction_precision=out["repaired"] / (out["repaired"] + out["damaged"]) if out["repaired"] + out["damaged"] else None,
        foreground_nll=out["nll_sum"] / out["objects"], ece=float(np.abs(bins[:, 1] - bins[:, 2]).sum() / out["objects"]),
        class_count=counts.tolist(), class_correct=correct.tolist())
    return out


def progress(stage, completed, total, started):
    elapsed = time.monotonic() - started
    value = dict(detail=stage, images=completed, total=total, seconds=elapsed,
                 eta_seconds=elapsed / max(1, completed) * (total - completed))
    atomic_json(OUT / "progress.json", value); print(json.dumps(value), flush=True)


def build_bank(ids, expected_registration):
    buckets = {i: [] for i in range(1, 151)}
    started = time.monotonic()
    for j, iid in enumerate(ids, 1):
        path = ROOT / "cache/VF/sgcls/train" / (iid + ".pt")
        item = torch.load(str(path), map_location="cpu")
        if item["registration_sha256"] != expected_registration or item["image_id"] != iid:
            raise RuntimeError("Training memory provenance mismatch")
        x, y = item["features"][:, :4096], item["targets"]
        if not torch.isfinite(x).all() or not bool(((y >= 1) & (y <= 150)).all()):
            raise RuntimeError("Invalid training bank entry")
        for index, label in enumerate(y.tolist()):
            key = hashlib.sha256((iid + ":%d:17" % index).encode()).hexdigest()
            buckets[label].append((key, iid, index, x[index].clone()))
        if j % 500 == 0:
            progress("training-only ROI memory", j, len(ids), started)
    features, labels, provenance = [], [], []
    counts = {str(i): len(bucket) for i, bucket in buckets.items()}
    for label, values in buckets.items():
        for key, iid, index, x in sorted(values, key=lambda v: v[0])[:SPEC["per_class_cap"]]:
            features.append(x); labels.append(label)
            provenance.append(dict(image_id=iid, object_index=index, label=label))
    keys = F.normalize(torch.stack(features).float(), dim=-1)
    values = torch.tensor(labels, dtype=torch.long)
    path = ROOT / "cache/VG_retrieval/bank.pt"
    save_torch(path, dict(keys=keys, labels=values, provenance=provenance,
                          training_class_counts=counts, source_registration=expected_registration))
    atomic_json(OUT / "bank.json", dict(path=str(path), sha256=sha256(path), objects=len(keys),
                class_counts=counts, retained_counts=torch.bincount(values, minlength=151).tolist(),
                training_images=len(ids), no_development_or_test_in_memory=True))
    return keys.cuda(), values.cuda()


def run():
    source = verify_vf()
    splits = read(ROOT / "results/VF/sgcls/splits.json")
    if set(splits["train"]) & set(splits["development"]) or len(splits["train"]) != 5000 or len(splits["development"]) != 500:
        raise RuntimeError("Invalid development split")
    immutable(OUT / "protocol.json", dict(spec=SPEC, source_sha256=sha256(HERE / "retrieval_repair.py"),
        source_registration=source, split_sha256=sha256(ROOT / "results/VF/sgcls/splits.json"),
        note="Only prior training and development IDs are accessed. Reserved gate IDs are not evaluated."))
    torch.backends.cuda.matmul.allow_tf32 = False
    keys, values = build_bank(splits["train"], source)
    candidates = [("selective_a%.2f_t%.1f" % (a, t), a, t) for a in SPEC["alphas"] for t in SPEC["uncertainty_thresholds"]]
    rows = {name: [] for name in ["native", "retrieval_only", "nonselective_a0.25"] + [c[0] for c in candidates]}
    started = time.monotonic()
    with torch.no_grad():
        for j, iid in enumerate(splits["development"], 1):
            item = torch.load(str(ROOT / "cache/VF/sgcls/development" / (iid + ".pt")), map_location="cpu")
            if item["registration_sha256"] != source or item["image_id"] != iid:
                raise RuntimeError("Development cache provenance mismatch")
            baseline, y = item["baseline"].cuda(), item["targets"].cuda()
            memory = nearest_posterior(item["features"].cuda(), keys, values, SPEC["k"], SPEC["temperature"])
            rows["native"].append(record(iid, baseline, baseline, y))
            retrieval, mask = fusion(baseline, memory, 1., 1., selective=False)
            rows["retrieval_only"].append(record(iid, retrieval, baseline, y, mask))
            full, mask = fusion(baseline, memory, .25, 1., selective=False)
            rows["nonselective_a0.25"].append(record(iid, full, baseline, y, mask))
            for name, alpha, threshold in candidates:
                updated, mask = fusion(baseline, memory, alpha, threshold, SPEC["neighbor_consensus_min"])
                rows[name].append(record(iid, updated, baseline, y, mask))
            if j % 50 == 0:
                progress("development retrieval feasibility; no fitting or test access", j, 500, started)
    summaries = {name: summarize(v) for name, v in rows.items()}
    expected = read(ROOT / "results/VF/sgcls/training/supervised/initial_development.json")
    if abs(summaries["native"]["top1"] - expected["native_top1"]) > 1e-12:
        raise RuntimeError("Replayed native development accuracy does not match V-F")
    checks, eligible = {}, []
    for name, alpha, threshold in candidates:
        row = summaries[name]
        checks[name] = dict(identity_gain=row["delta"] >= .005,
                           correction_precision=row["correction_precision"] is not None and row["correction_precision"] >= .6,
                           macro_accuracy=row["macro_accuracy"] >= summaries["native"]["macro_accuracy"])
        if all(checks[name].values()):
            eligible.append((name, alpha, threshold))
    best = max(eligible, key=lambda c: (summaries[c[0]]["top1"], -summaries[c[0]]["damaged"], -c[1], -c[2])) if eligible else None
    atomic_json(OUT / "image_records.json", rows)
    atomic_json(OUT / "summary.json", dict(status="complete", development_eligible=best is not None,
        selected=dict(name=best[0], alpha=best[1], threshold=best[2]) if best else None,
        metrics=summaries, checks=checks, protocol_sha256=sha256(OUT / "protocol.json"),
        images=500, fresh_gate_evaluated=False, test_evaluated=False, trained_parameters=0,
        interpretation="Development-selected exploratory feasibility only; standard SGG/noninferiority still untested",
        next_action="register native full-system gate before expanding" if best else "stop this configuration; no gate/test/search expansion"))
    print(json.dumps(dict(development_eligible=bool(best), selected=best,
          metrics={k: {f: v[f] for f in ["top1", "delta", "macro_accuracy", "repaired", "damaged", "correction_precision"]} for k, v in summaries.items()})), flush=True)


def main():
    ensure_storage(); torch.set_num_threads(2)
    state = dict(status="waiting_gpu", gpu=[1], pid=os.getpid(),
        command=[str(HERE / "retrieval_repair.py")], progress_file=str(OUT / "progress.json"),
        completion=str(OUT / "summary.json"), log=str(ROOT / "logs/VG_retrieval_development.log"))
    atomic_json(STATUS, state)
    lock = (ROOT / "status/gpu1.resource.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("Use physical GPU1 only")
        if subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            raise RuntimeError("GPU1 is occupied")
        atomic_json(STATUS, dict(state, status="running")); run()
        atomic_json(STATUS, dict(state, status="complete"))
    except Exception as exc:
        atomic_json(STATUS, dict(state, status="failed", reason=str(exc))); raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
