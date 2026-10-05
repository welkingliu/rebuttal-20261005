"""V-I native no-op audits and full-system held-out evaluation."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, output_path, sha256, ensure_storage
from native_runtime import infer
from repro_experiment import dataset, save_torch
from vb_native import targets, OfficialMetrics, compare_predictions
from vc_protocol import paired_gate, summarize_rows
from ve_experiment import setup as ve_setup, lookup, native_cache_check
from vf_native import ReadoutPatch
from vi_protocol import OUT, CACHE, CKPT, SPEC, read, verify


class FusionPatch(ReadoutPatch):
    def __init__(self, model):
        super().__init__(model, None)
        self.update = None

    def _after(self, module, inputs, output):
        baseline, relations = output[0][0], output[1][0]
        logits = baseline
        if self.enabled:
            update = self.update
            if update is None or update["size"] != self.capture["size"]:
                raise RuntimeError("Missing or mismatched fusion update")
            if (update["native_logits"].shape != baseline.shape or
                    not torch.allclose(update["native_logits"].to(baseline), baseline, atol=1e-5, rtol=1e-5) or
                    not torch.allclose(update["boxes"].to(baseline), self.capture["proposal_boxes"], atol=1e-4, rtol=1e-5)):
                raise RuntimeError("Native proposal order/baseline changed")
            logits = update["logits"].to(baseline)
            if logits.shape != baseline.shape or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid updated object logits")
        self.capture.update(baseline=baseline.detach(), logits=logits.detach(), relation_logits=relations.detach())
        return ([logits], output[1]) + tuple(output[2:])


def setup(task, out):
    model, cfg, transform, provenance, _, old = ve_setup(task, out)
    old.close()
    return model, cfg, transform, provenance, FusionPatch(model)


def budget(deadline):
    if time.time() >= deadline:
        raise TimeoutError("V-I queue budget exhausted")


def progress(out, detail, done, total, started):
    elapsed = time.monotonic() - started
    row = dict(detail=detail, images=done, total=total, seconds=elapsed,
               eta_seconds=elapsed / max(done, 1) * (total - done))
    atomic_json(out / "progress.json", row)
    print(__import__("json").dumps(row), flush=True)


def smoke(task, deadline):
    out = OUT / task / "smoke"
    model, cfg, transform, provenance, patch = setup(task, out)
    ds = dataset(cfg, "val"); mapping = lookup(ds); rows = []
    for iid in read(OUT / "splits.json")["development"][:3]:
        budget(deadline); patch.enabled = False
        base, _ = infer(model, cfg, transform, ds, mapping[iid])
        cache_error = native_cache_check(base, task, iid)
        c = patch.capture
        patch.update = dict(native_logits=c["baseline"].clone(), logits=c["baseline"].clone(),
                            boxes=c["proposal_boxes"].clone(), size=c["size"])
        patch.enabled = True
        noop, _ = infer(model, cfg, transform, ds, mapping[iid])
        no_error = compare_predictions(base, noop)
        old_rel = patch.capture["relation_logits"].clone()
        patch.update["logits"] = patch.update["native_logits"] + torch.linspace(-.2, .2, 151, device="cuda")
        altered, _ = infer(model, cfg, transform, ds, mapping[iid])
        if not torch.equal(old_rel, patch.capture["relation_logits"]):
            raise RuntimeError("Fusion unexpectedly changed predicate logits")
        delta = float((base.get_field("pred_scores") - altered.get_field("pred_scores")).abs().max())
        if delta <= 1e-8 or not patch.capture.get("route_checked"):
            raise RuntimeError("Object update failed to reach native postprocessing")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Native SGG unfrozen")
        rows.append(dict(image_id=iid, noop=no_error, native_cache_errors=cache_error, synthetic_score_delta=delta))
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", protocol_sha256=verify(),
        checks=rows, synthetic_update_discarded=True, heldout_gate_accessed=False))


def allowed(task):
    if not read(OUT / "development/summary.json")["development_eligible"]:
        raise RuntimeError("Development gate rejected V-I")
    if task == "sgdet" and not read(OUT / "sgcls/decision.json")["accepted"]:
        raise RuntimeError("SGCls gate did not permit SGDet")


def export(task, deadline):
    allowed(task)
    out = OUT / task / "export"
    model, cfg, transform, provenance, patch = setup(task, out)
    ds = dataset(cfg, "val"); mapping = lookup(ds)
    ids = read(OUT / "splits.json")["gate"]
    if len(ids) != 1000 or not set(ids) <= set(mapping):
        raise RuntimeError("Incomplete native held-out gate")
    metric = OfficialMetrics(task); rows = []; started = time.monotonic()
    reg = verify()
    for n, iid in enumerate(ids, 1):
        budget(deadline)
        pred, _ = infer(model, cfg, transform, ds, mapping[iid])
        native_cache_check(pred, task, iid)
        c = patch.capture
        gt = ds.get_groundtruth(mapping[iid], evaluation=True)
        row = metric.row(iid, pred, gt, c, targets(c, gt, task))
        row["protocol_sha256"] = reg
        atomic_json(OUT / task / "gate/native/images" / (iid + ".json"), row); rows.append(row)
        image = str(Path(ds.filenames[mapping[iid]]).resolve())
        save_torch(CACHE / task / "export" / (iid + ".pt"), dict(image_id=iid, image=image,
            boxes=c["proposal_boxes"].cpu(), baseline=c["baseline"].cpu(), size=c["size"], protocol_sha256=reg))
        if n % 25 == 0:
            progress(out, task + " native gate export", n, len(ids), started)
    patch.close()
    atomic_json(OUT / task / "gate/native/summary.json", dict(status="complete", protocol_sha256=reg, **summarize_rows(rows)))
    atomic_json(out / "summary.json", dict(status="complete", images=len(ids), protocol_sha256=reg,
                                          baseline_sha256=provenance["checkpoint_sha256"]))


def evaluate(task, deadline):
    allowed(task)
    out = OUT / task / "gate/selective"
    if read(OUT / task / "expert/summary.json")["checkpoint_sha256"] != sha256(CKPT / "selected.pth"):
        raise RuntimeError("Frozen selector changed before gate")
    model, cfg, transform, provenance, patch = setup(task, out)
    ds = dataset(cfg, "val"); mapping = lookup(ds)
    ids = read(OUT / "splits.json")["gate"]
    metric = OfficialMetrics(task); rows = []; started = time.monotonic(); reg = verify()
    from export_pysgg_vg_task import convert_prediction
    patch.enabled = True
    for n, iid in enumerate(ids, 1):
        budget(deadline)
        update = torch.load(str(CACHE / task / "updates" / (iid + ".pt")), map_location="cpu")
        if update["protocol_sha256"] != reg or update["checkpoint_sha256"] != sha256(CKPT / "selected.pth"):
            raise RuntimeError("Fusion result provenance mismatch")
        patch.update = update
        pred, _ = infer(model, cfg, transform, ds, mapping[iid])
        if not patch.capture.get("route_checked"):
            raise RuntimeError("Native postprocessing bypassed updated scores")
        gt = ds.get_groundtruth(mapping[iid], evaluation=True)
        row = metric.row(iid, pred, gt, patch.capture, targets(patch.capture, gt, task))
        row.update(protocol_sha256=reg, selected_proposals=int(update["selected"].sum()))
        atomic_json(out / "images" / (iid + ".json"), row); rows.append(row)
        path = output_path(out / "predictions" / (iid + ".npz")); temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **convert_prediction(pred.to("cpu"))); temp.replace(path)
        if n % 25 == 0:
            progress(out, task + " native fusion gate", n, len(ids), started)
    patch.close()
    atomic_json(out / "summary.json", dict(status="complete", protocol_sha256=reg, **summarize_rows(rows)))


def decision(task):
    ids = read(OUT / "splits.json")["gate"]
    base = [read(OUT / task / "gate/native/images" / (i + ".json")) for i in ids]
    changed = [read(OUT / task / "gate/selective/images" / (i + ".json")) for i in ids]
    result = paired_gate(base, changed)
    atomic_json(OUT / task / "decision.json", dict(status="complete", protocol_sha256=verify(),
        **result, test_evaluated=False, heldout_from_adapter_fitting=True))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["smoke", "export", "evaluate", "decision"])
    p.add_argument("--task", choices=["sgcls", "sgdet"], required=True)
    p.add_argument("--deadline", type=float, required=True)
    args = p.parse_args(); ensure_storage(); verify(); torch.set_num_threads(2); budget(args.deadline)
    if args.stage == "decision":
        decision(args.task)
    else:
        {"smoke": smoke, "export": export, "evaluate": evaluate}[args.stage](args.task, args.deadline)


if __name__ == "__main__":
    main()
