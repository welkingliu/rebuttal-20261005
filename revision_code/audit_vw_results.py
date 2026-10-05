"""Separate completed V-W audit; never changes selected models or original gates."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from common import ROOT, ensure_storage, sha256, atomic_json
from evidence_completion import read


def close(a, b, label):
    if not np.isclose(a, b, atol=1e-12, rtol=0):
        raise RuntimeError("Reaggregation mismatch: " + label)


def bootstrap(rows, reference, arm, draws=20000):
    keys = ["pre_nms", "post_nms", "unique_post", "native_top50_endpoint", "not_native_top50_endpoint"]
    denominators, changes = [], []
    for key in keys:
        denominators.append([r["scopes"][reference][key]["objects"] for r in rows])
        changes.append([r["scopes"][arm][key]["correct"]-r["scopes"][reference][key]["correct"] for r in rows])
    denominator = np.asarray(denominators, dtype=float).T
    delta = np.asarray(changes, dtype=float).T
    rdiff = np.array([r["metrics"][arm]["recalls"][4]-r["metrics"][reference]["recalls"][4] for r in rows])
    b = np.array([r["metrics"][reference]["class_recalls"][4] for r in rows], dtype=float)
    c = np.array([r["metrics"][arm]["class_recalls"][4] for r in rows], dtype=float)
    support = np.isfinite(b).astype(float)
    if not np.array_equal(np.isfinite(b), np.isfinite(c)):
        raise RuntimeError("Unpaired predicate support")
    mdiff = np.nan_to_num(c-b)
    n = len(rows)
    rng = np.random.default_rng(17029)
    samples = []
    for start in range(0, draws, 100):
        ix = rng.integers(0, n, (min(100, draws-start), n))
        weights = np.stack([np.bincount(i, minlength=n) for i in ix]).astype(float)
        denom = weights @ denominator
        if (denom <= 0).any():
            raise RuntimeError("Empty bootstrap support")
        support_sum = weights @ support
        macro = np.divide(weights @ mdiff, support_sum, out=np.zeros_like(support_sum), where=support_sum > 0).mean(1)
        samples.append(np.column_stack([(weights @ delta)/denom, weights @ rdiff/n, macro]))
    boot = 100*np.concatenate(samples)
    return dict(draws=draws, seed=17029, image_order="lexicographic image_id, shared resamples across all scopes",
        paired_image_95ci_pp=dict(zip(keys+["R50", "mR50"], np.quantile(boot, [.025, .975], axis=0).T.tolist())),
        one_sided_95_lower_pp=dict(zip(keys+["R50", "mR50"], np.quantile(boot, .05, axis=0).tolist())),
        caveat="Descriptive, fitted models not refitted; no adaptive-selection/multiplicity correction; not a replacement gate")


def main():
    ensure_storage()
    torch.set_num_threads(2)
    started = time.monotonic()
    out = ROOT / "results/VW_endpoint_relation/train3000"
    protocol, summary = read(out/"protocol.json"), read(out/"summary.json")
    reg = sha256(out/"protocol.json")
    if summary["protocol_sha256"] != reg:
        raise RuntimeError("Summary protocol mismatch")
    for name, digest in protocol["sources"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise RuntimeError("Changed source: " + name)
    rows, choices = [], []
    record_hash = hashlib.sha256()
    for fold in range(5):
        stage = out/("fold%d" % fold)
        split, done, info = [read(stage/name) for name in ["split.json", "summary.json", "training_summary.json"]]
        path = ROOT/"checkpoints/VW_endpoint_relation/train3000"/("fold%d.pth" % fold)
        digest = sha256(path)
        payload = torch.load(str(path), map_location="cpu")
        if any(v["protocol_sha256"] != reg for v in [split, done, info, payload]):
            raise RuntimeError("Fold provenance drift")
        if done["checkpoint_sha256"] != digest or info["checkpoint_sha256"] != digest:
            raise RuntimeError("Checkpoint drift")
        train, held, fit, inner = [set(split[k]) for k in ["training", "held", "fit", "inner_validation"]]
        if train & held or fit & inner or fit | inner != train or train | held != set(protocol["image_ids"]):
            raise RuntimeError("Invalid image split")
        if len(held) != 600 or set(payload["fit_ids"]) != train or set(payload["held_ids"]) != held:
            raise RuntimeError("Fit/held provenance mismatch")
        if done["epochs"] != info["epochs"] or payload["epochs"] != info["epochs"]:
            raise RuntimeError("Selection mismatch")
        for arm, history in info["histories"].items():
            if [v["epoch"] for v in history["inner"]] != list(range(1, 9)):
                raise RuntimeError("Missing inner training epochs")
            if len(history["refit"]) != info["epochs"][arm]:
                raise RuntimeError("Refit epoch mismatch")
            if info["epochs"][arm] == 0 and any(torch.count_nonzero(v) for v in payload["states"][arm].values()):
                raise RuntimeError("No-op weights nonzero")
        audit = info["route_gradient_audit"]
        if not audit["all_checks_passed"] or any(v <= 0 for v in audit["loss_parameter_gradient_norms"].values()):
            raise RuntimeError("Gradient/route preflight failed")
        for iid in split["held"]:
            p = stage/"images"/(iid+".json")
            record_hash.update((iid+sha256(p)).encode())
            row = read(p)
            if row["image_id"] != iid or row["fold"] != fold or row["protocol_sha256"] != reg or row["checkpoint_sha256"] != digest:
                raise RuntimeError("Per-image drift")
            for arm in protocol["arms"]:
                native, value = row["metrics"]["native"], row["metrics"][arm]
                scope = row["scopes"][arm]
                if value["positive_objects"] != native["positive_objects"]:
                    raise RuntimeError("Fixed denominator changed")
                for key, field in [("pre_nms", "positive_correct"), ("post_nms", "post_nms_correct")]:
                    if scope[key]["correct"] != value[field] or scope[key]["objects"] != value["positive_objects"]:
                        raise RuntimeError("Readout scope mismatch")
                for field in ["objects", "correct"]:
                    if scope["native_top50_endpoint"][field]+scope["not_native_top50_endpoint"][field] != scope["post_nms"][field]:
                        raise RuntimeError("Endpoint strata do not partition proposals")
                if info["epochs"].get(arm) == 0 and (scope != row["scopes"]["native"] or value != native):
                    raise RuntimeError("No-op result mismatch")
            rows.append(row)
        choices.append(dict(fold=fold, checkpoint_sha256=digest, epochs=info["epochs"], all_inner_fits_ran_8=True,
                            route_gradient_checks_passed=True))
    if len(rows) != 3000 or {r["image_id"] for r in rows} != set(protocol["image_ids"]):
        raise RuntimeError("Duplicate/incomplete outer records")
    rows.sort(key=lambda r:r["image_id"])
    for arm in protocol["arms"]:
        values = [r["metrics"][arm] for r in rows]
        n = sum(v["positive_objects"] for v in values)
        close(sum(v["positive_correct"] for v in values)/n, summary["arms"][arm]["object_top1"], arm+" pre")
        close(sum(v["post_nms_correct"] for v in values)/n, summary["arms"][arm]["post_nms_object_top1"], arm+" post")
        close(np.mean([v["recalls"][4] for v in values]), summary["arms"][arm]["R"]["50"], arm+" R50")
    print("Sources, five checkpoints, splits, all15 inner fits and3000 image records verified", flush=True)
    comparisons = {}
    for reference in ["native", "uniform_ce", "endpoint_ce"]:
        comparisons["endpoint_relation_versus_"+reference] = bootstrap(rows, reference, "endpoint_relation")
        print("Bootstrap complete: endpoint_relation vs "+reference, flush=True)
    report = dict(status="complete", protocol_sha256=reg, source_sha256=sha256(Path(__file__)),
        original_summary_sha256=sha256(out/"summary.json"), per_image_manifest_sha256=record_hash.hexdigest(),
        records_verified=3000, choices=choices, comparisons=comparisons,
        original_gate_unchanged=True, fresh_prediction_inference_run=False,
        ci_note="The two old2000-draw routines used different image ordering; small boundary differences are Monte Carlo variation. These20000-draw checks share a canonical order, without changing original decisions.",
        elapsed_seconds=time.monotonic()-started)
    dest = out/"audit_20261004.json"
    if dest.exists():
        raise FileExistsError("Do not overwrite an earlier audit")
    atomic_json(dest, report)
    print(str(dest), flush=True)


if __name__ == "__main__":
    main()
