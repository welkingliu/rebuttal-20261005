"""Read-only diagnosis of the completed V-E SGCls pilot, not a new gate."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, sha256
from native_runtime import infer
from repro_experiment import dataset
from repro_protocol import immutable
from sgdet_identity import capture
from vc_protocol import choose_ids, paired_gate
from ve_experiment import setup, lookup, checkpoint, read, split_path
from ve_native import objective_terms
from ve_protocol import verify

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VE_failure_audit_20261001"
STATUS = ROOT / "status/VE_failure_audit.json"
MODES = ["supervised", "relation_aware"]
SPEC = dict(version="ve_sgcls_failure_audit_v1", task="sgcls", seed=17,
            train_images=500, development_images=500, gradient_images=32,
            bootstrap_samples=2000, bootstrap_seed=17041,
            selection="fixed hash order from already registered train/development IDs",
            scope="post-hoc mechanism audit; no fitting, gate reselection, or test inference",
            class_groups="three groups of 50 classes sorted by frequency in the 5000 training images",
            coordinate_bound="optimistic independent-coordinate 0.1*rms bound, not attainable capacity",
            stop="one audit only; no automatic hyperparameter search")


def transitions(target, before, after):
    target, before, after = map(np.asarray, (target, before, after))
    if target.shape != before.shape or before.shape != after.shape:
        raise ValueError("Unaligned object labels")
    b, a = before == target, after == target
    return dict(objects=len(target), baseline_correct=int(b.sum()), updated_correct=int(a.sum()),
                repaired=int((~b & a).sum()), damaged=int((b & ~a).sum()),
                wrong_to_different_wrong=int((~b & ~a & (before != after)).sum()),
                label_flips=int((before != after).sum()))


def cosine(a, b):
    norm = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / norm) if norm > 1e-20 else None


def confidence_interval(rows):
    values = np.asarray([[r["updated_correct"] - r["baseline_correct"], r["objects"]] for r in rows])
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    draws = []
    for _ in range(SPEC["bootstrap_samples"]):
        ix = rng.integers(0, len(values), len(values))
        draws.append(values[ix, 0].sum() / values[ix, 1].sum())
    return dict(delta=float(values[:, 0].sum() / values[:, 1].sum()),
                paired_image_bootstrap_95_ci=np.quantile(draws, [.025, .975]).tolist())


def sum_transitions(rows):
    keys = ["objects", "baseline_correct", "updated_correct", "repaired", "damaged",
            "wrong_to_different_wrong", "label_flips"]
    out = {key: sum(x[key] for x in rows) for key in keys}
    if out["objects"]:
        out.update(confidence_interval(rows))
        out["baseline_top1"] = out["baseline_correct"] / out["objects"]
        out["updated_top1"] = out["updated_correct"] / out["objects"]
    return out


def progress(stage, done, total, started):
    elapsed = time.monotonic() - started
    value = dict(images=done, total=total, detail=stage, seconds=elapsed,
                 eta_seconds=elapsed / done * (total - done) if done else None)
    atomic_json(OUT / "progress.json", value)
    print(json.dumps(value), flush=True)


def gate_audit(ds, mapping, splits, groups):
    base = ROOT / "results/VE/sgcls/gate"
    rows = {mode: [] for mode in ["native"] + MODES}
    counts = {mode: [] for mode in MODES}
    grouped = {mode: {name: [] for name in groups} for mode in MODES}
    classes = {mode: {i: [] for i in range(1, 151)} for mode in MODES}
    for iid in splits["gate"]:
        gt = ds.get_groundtruth(mapping[iid], evaluation=True)
        target = gt.get_field("labels").numpy()
        predicted, boxes = {}, {}
        for mode in ["native"] + MODES:
            folder = base / (mode + "_seed17")
            row = read(folder / "images" / (iid + ".json"))
            rows[mode].append(row)
            with np.load(folder / "predictions" / (iid + ".npz"), allow_pickle=False) as z:
                scores = z["pred_entity_scores"]
                if scores.shape != (len(target), 151):
                    raise RuntimeError("SGCls prediction/GT count mismatch")
                predicted[mode] = scores[:, 1:].argmax(1) + 1
                boxes[mode] = z["pred_boxes"].copy()
            actual = int((predicted[mode] == target).sum())
            if row["post_nms_correct"] != actual or row["positive_objects"] != len(target):
                raise RuntimeError("Saved row does not match per-object cache")
        for mode in MODES:
            if not np.array_equal(boxes[mode], boxes["native"]):
                raise RuntimeError("SGCls proposal alignment changed")
            result = transitions(target, predicted["native"], predicted[mode])
            counts[mode].append(dict(image_id=iid, **result))
            for name, ids in groups.items():
                ix = np.isin(target, ids)
                grouped[mode][name].append(transitions(target[ix], predicted["native"][ix], predicted[mode][ix]))
            for label in np.unique(target):
                ix = target == label
                classes[mode][int(label)].append(transitions(target[ix], predicted["native"][ix], predicted[mode][ix]))
    result = {}
    for mode in MODES:
        result[mode] = dict(**sum_transitions(counts[mode]),
            unchanged_gate=paired_gate(rows["native"], rows[mode]),
            frequency_groups={name: sum_transitions(r) for name, r in grouped[mode].items()},
            per_class={str(i): sum_transitions(r) for i, r in classes[mode].items() if r})
        atomic_json(OUT / (mode + "_gate_transitions.json"), counts[mode])
    atomic_json(OUT / "gate_audit.json", result)
    return result


def feature_row(x, head, out_obj, y, native):
    with torch.no_grad():
        changed = head(x)
        logits = out_obj(changed)
        before = native[:, 1:].argmax(1) + 1
        after = logits[:, 1:].argmax(1) + 1
        row = transitions(y.cpu().numpy(), before.cpu().numpy(), after.cpu().numpy())
        rms = x.square().mean(-1).sqrt().clamp_min(1e-6)
        delta = changed - x
        tanh = torch.tanh(head.up(F.gelu(head.down(F.layer_norm(x, (x.shape[-1],))))))
        wrong = before != y
        margin = native.gather(1, before[:, None])[:, 0] - native.gather(1, y[:, None])[:, 0]
        limit = head.scale * rms * (out_obj.weight[before] - out_obj.weight[y]).abs().sum(-1)
        direction = (logits - native).gather(1, y[:, None])[:, 0] - (logits - native).gather(1, before[:, None])[:, 0]
        row.update(baseline_nll_sum=float(F.cross_entropy(native, y, reduction="sum")),
                   updated_nll_sum=float(F.cross_entropy(logits, y, reduction="sum")),
                   relative_update_rms_sum=float((delta.square().mean(-1).sqrt() / rms).sum()),
                   saturated_coordinates=int((tanh.abs() >= .95).sum()), coordinates=x.numel(),
                   wrong_objects=int(wrong.sum()), wrong_with_margin_improved=int((wrong & (direction > 0)).sum()),
                   wrong_impossible_even_under_independent_coordinate_bound=int((wrong & (margin > limit)).sum()),
                   max_abs_logit_change=float((logits - native).abs().max()))
    return row, logits


def aggregate_features(rows):
    out = sum_transitions(rows)
    n = out["objects"]
    for name in ["baseline_nll_sum", "updated_nll_sum", "relative_update_rms_sum"]:
        out[name.replace("_sum", "_mean")] = sum(r[name] for r in rows) / n
    for name in ["wrong_objects", "wrong_with_margin_improved", "wrong_impossible_even_under_independent_coordinate_bound"]:
        out[name] = sum(r[name] for r in rows)
    out["coordinate_saturation_fraction"] = sum(r["saturated_coordinates"] for r in rows) / sum(r["coordinates"] for r in rows)
    out["max_abs_logit_change"] = max(r["max_abs_logit_change"] for r in rows)
    return out


def grad_row(student, teacher, y, pairs, head):
    terms = objective_terms(student, teacher, y, pairs)
    params = tuple(head.parameters())
    grads = {}
    for name, loss in terms.items():
        vectors = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        grads[name] = torch.cat([(v if v is not None else torch.zeros_like(p)).detach().flatten().cpu()
                                 for v, p in zip(vectors, params)])
    protection = sum(v for k, v in grads.items() if k != "object_ce")
    ce = grads["object_ce"]
    row = dict(losses={k: float(v.detach()) for k, v in terms.items()},
               norms={k: float(v.norm()) for k, v in grads.items()},
               cosines_with_ce={k: cosine(ce, v) for k, v in grads.items() if k != "object_ce"},
               protection_norm=float(protection.norm()), protection_ce_cosine=cosine(ce, protection),
               total_ce_cosine=cosine(ce, ce + protection))
    return row, grads


def run():
    reg = verify()
    splits = read(split_path("sgcls"))
    states, hashes = {}, {}
    for mode in MODES:
        path = checkpoint("sgcls", mode, 17)
        value = torch.load(str(path), map_location="cpu")
        history = read(ROOT / "results/VE/sgcls/training" / (mode + "_seed17") / "summary.json")
        if value["registration_sha256"] != reg or value["split_sha256"] != sha256(split_path("sgcls")):
            raise RuntimeError("Audit checkpoint is not the registered V-E pilot")
        hashes[mode] = sha256(path)
        if hashes[mode] != history["checkpoint_sha256"]:
            raise RuntimeError("Audit checkpoint differs from the evaluated checkpoint")
        states[mode] = value["head"]
    selections = dict(train=choose_ids(splits["train"], SPEC["train_images"], "VE_audit_train:"),
                      development=choose_ids(splits["development"], SPEC["development_images"], "VE_audit_dev:"))
    manifest = dict(spec=SPEC, registration_sha256=reg, checkpoint_sha256=hashes,
                    source_sha256=sha256(HERE / "ve_failure_audit.py"), selections=selections,
                    gate_was_already_observed=True, does_not_modify_original_gate=True)
    immutable(OUT / "protocol.json", manifest)
    model, cfg, transform, provenance, head, patch = setup("sgcls", OUT)
    datasets = {"train": dataset(cfg, "train"), "development": dataset(cfg, "val")}
    mappings = {k: lookup(ds) for k, ds in datasets.items()}
    histogram = np.zeros(151, dtype=int)
    for iid in splits["train"]:
        gt = datasets["train"].get_groundtruth(mappings["train"][iid], evaluation=True)
        histogram += np.bincount(gt.get_field("labels").numpy(), minlength=151)
    order = sorted(range(1, 151), key=lambda i: (-histogram[i], i))
    groups = dict(head=order[:50], body=order[50:100], tail=order[100:])
    atomic_json(OUT / "class_groups.json", dict(training_counts=histogram.tolist(), groups=groups))
    progress("paired gate cache audit; no new gate evaluation", 0, 1000, time.monotonic())
    gate = gate_audit(datasets["development"], mappings["development"], splits, groups)
    for mode in MODES:
        print(json.dumps(dict(mode=mode, gate={k: v for k, v in gate[mode].items() if k not in ("per_class", "frequency_groups")})), flush=True)

    holder = {}
    hook = patch.context.context_obj.register_forward_hook(lambda module, inputs, output: holder.update(features=output.detach()))
    all_rows, gradient_rows, gradient_sums = {}, {m: [] for m in MODES}, {m: {} for m in MODES}
    started, completed = time.monotonic(), 0
    for split in ["development", "train"]:
        rows = {mode: [] for mode in MODES}
        ds, mapping = datasets[split], mappings[split]
        for position, iid in enumerate(selections[split]):
            index = mapping[iid]
            patch.enabled = False
            do_grad = split == "development" and position < SPEC["gradient_images"]
            replay, prediction = capture(model, cfg, transform, ds, index) if do_grad else (None, infer(model, cfg, transform, ds, index)[0])
            x, native = holder["features"].clone(), patch.capture["logits"].clone()
            y = ds.get_groundtruth(index, evaluation=True).get_field("labels").cuda()
            if not torch.equal(native[:, 1:].argmax(1) + 1, prediction.get_field("pred_labels")):
                raise RuntimeError("Native SGCls identity route is not foreground argmax")
            for mode in MODES:
                head.load_state_dict(states[mode]); head.eval()
                row, expected = feature_row(x, head, patch.context.out_obj, y, native)
                rows[mode].append(dict(image_id=iid, **row))
                if do_grad:
                    patch.enabled = True
                    student = model.roi_heads.relation.predictor(*replay["inputs"])
                    if not torch.allclose(student[0][0], expected, atol=1e-5, rtol=1e-5):
                        raise RuntimeError("Cached context/output identity path differs from native replay")
                    grow, vectors = grad_row(student, replay["output"], y, replay["inputs"][1][0], head)
                    gradient_rows[mode].append(dict(image_id=iid, **grow))
                    for key, vector in vectors.items():
                        gradient_sums[mode][key] = gradient_sums[mode].get(key, torch.zeros_like(vector)) + vector
                    del student, vectors
                    patch.enabled = False
            del replay, x, native, prediction
            completed += 1
            if completed % 25 == 0:
                progress("SGCls context/margin audit: " + split, completed, 1000, started)
        all_rows[split] = {mode: aggregate_features(r) for mode, r in rows.items()}
        atomic_json(OUT / (split + "_feature_rows.json"), rows)
        atomic_json(OUT / (split + "_summary.json"), all_rows[split])
    gradients = {}
    for mode in MODES:
        means = {key: v / len(gradient_rows[mode]) for key, v in gradient_sums[mode].items()}
        ce = means["object_ce"]
        protection = sum(v for key, v in means.items() if key != "object_ce")
        gradients[mode] = dict(images=len(gradient_rows[mode]),
            mean_gradient_norms={key: float(v.norm()) for key, v in means.items()},
            mean_gradient_cosines_with_ce={key: cosine(ce, v) for key, v in means.items() if key != "object_ce"},
            mean_protection_norm=float(protection.norm()), mean_protection_ce_cosine=cosine(ce, protection),
            mean_total_ce_cosine=cosine(ce, ce + protection),
            per_image_median_norms={key: float(np.median([r["norms"][key] for r in gradient_rows[mode]])) for key in means})
    atomic_json(OUT / "gradient_rows.json", gradient_rows)
    hook.remove(); patch.close()
    if verify() != reg or any(sha256(checkpoint("sgcls", m, 17)) != hashes[m] for m in MODES):
        raise RuntimeError("Original V-E assets changed during read-only audit")
    atomic_json(OUT / "summary.json", dict(status="complete", protocol_sha256=sha256(OUT / "protocol.json"),
        checkpoint_sha256=hashes, feature_audit=all_rows, gradient_audit=gradients,
        gate_audit=str(OUT / "gate_audit.json"), seconds=time.monotonic() - started,
        note="post-hoc SGCls diagnosis only; not a new mitigation success or fresh gate"))


def main():
    ensure_storage()
    state = dict(status="waiting_gpu", gpu=[1], pid=os.getpid(), command=[sys.executable, str(HERE / "ve_failure_audit.py")],
                 completion=str(OUT / "summary.json"), progress_file=str(OUT / "progress.json"),
                 log=str(ROOT / "logs/VE_failure_audit_20261001.log"))
    atomic_json(STATUS, state)
    lock = (ROOT / "status/gpu1.resource.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("Audit must run on physical GPU1")
        other = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if other:
            raise RuntimeError("GPU1 is not idle: " + other)
        atomic_json(STATUS, dict(state, status="running"))
        torch.set_num_threads(2)
        run()
        atomic_json(STATUS, dict(state, status="complete"))
    except Exception as exc:
        atomic_json(STATUS, dict(state, status="failed", reason=str(exc)))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
