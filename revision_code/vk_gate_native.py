"""Native SGCls/SGDet export, routing smoke and paired V-K acceptance."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, ensure_storage, output_path, sha256
from native_runtime import infer
from repro_experiment import dataset, save_torch
from ve_experiment import lookup, native_cache_check
from vi_native import setup
from vb_native import OfficialMetrics, compare_predictions, targets
from vc_protocol import summarize_rows, paired_gate
from vk_gate_protocol import OUT, CACHE, MANIFEST, PRIMARY_SHA, SPEC, read, verify, paths, eligible, ids


def configure_backend():
    # Match the audited historical native caches, not the crop-encoder runtime.
    backend = SPEC["native_backend"]
    torch.manual_seed(SPEC["seed"])
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]


def budget(deadline):
    if time.time() >= deadline: raise TimeoutError("V-K native execution budget reached")


def progress(folder, task, stage, done, total, started):
    elapsed = time.monotonic() - started
    value = dict(stage=stage, task=task, images=done, total=total, seconds=elapsed,
        eta_seconds=elapsed / max(done, 1) * (total - done))
    atomic_json(folder / "progress.json", value); print(__import__("json").dumps(value), flush=True)


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def crop_geometry(capture, gt, task):
    # SGCls support is supplied GT: retain its exact original coordinates so
    # resize/unresize roundoff cannot change integer crop boundaries.
    sx, sy = gt.size[0] / capture["size"][0], gt.size[1] / capture["size"][1]
    boxes = capture["proposal_boxes"].cpu() * torch.tensor([sx, sy, sx, sy])
    if task == "sgcls":
        if boxes.shape != gt.bbox.shape or not torch.allclose(boxes, gt.bbox.cpu(), atol=1e-3, rtol=1e-5):
            raise RuntimeError("Supplied SGCls support lost GT alignment")
        boxes = gt.bbox.cpu().clone()
    return boxes


def load_checked(path, registration, iid):
    value = torch.load(str(path), map_location="cpu")
    if value["protocol_sha256"] != registration or value["image_id"] != iid:
        raise RuntimeError("Native cache provenance mismatch: " + str(path))
    return value


def export(args):
    eligible(args.task, args.smoke)
    out, cache = paths(args.task, args.smoke); folder = out / "export"
    registration = verify(full=True)
    model, cfg, transform, provenance, patch = setup(args.task, folder)
    if provenance["checkpoint_sha256"] != read(MANIFEST)["baseline_assets"][args.task]["sha256"]:
        raise RuntimeError("Native checkpoint differs from registered baseline")
    ds = dataset(cfg, "val"); mapping = lookup(ds); selected = ids(args.smoke)
    if not set(selected) <= set(mapping): raise RuntimeError("Incomplete native validation image coverage")
    metric = OfficialMetrics(args.task); rows = []; started = time.monotonic(); smoke_checks = []
    from export_pysgg_vg_task import convert_prediction
    progress(folder, args.task, "native_export", 0, len(selected), started)
    for n, iid in enumerate(selected, 1):
        budget(args.deadline)
        target_path, row_path = cache / "export" / (iid + ".pt"), out / "native/images" / (iid + ".json")
        if target_path.exists() and row_path.exists() and not args.smoke:
            row = read(row_path); load_checked(target_path, registration, iid)
            if row["protocol_sha256"] != registration: raise RuntimeError("Native row drift")
            rows.append(row)
        else:
            patch.enabled = False
            pred, _ = infer(model, cfg, transform, ds, mapping[iid])
            errors = native_cache_check(pred, args.task, iid)
            c = patch.capture
            gt = ds.get_groundtruth(mapping[iid], evaluation=True)
            y = targets(c, gt, args.task)
            if len(pred) != len(y) or not c.get("route_checked"):
                raise RuntimeError("Native postprocessing changed proposal count/order contract")
            row = metric.row(iid, pred, gt, c, y)
            row.update(protocol_sha256=registration, native_reference_errors=errors)
            item = dict(image_id=iid, image=str(Path(ds.filenames[mapping[iid]]).resolve()),
                image_size=gt.size, crop_boxes=crop_geometry(c, gt, args.task),
                boxes=c["proposal_boxes"].cpu(), baseline=c["baseline"].cpu(), size=c["size"],
                protocol_sha256=registration, predicate_sha256=tensor_digest(c["relation_logits"]),
                pair_sha256=tensor_digest(c["pairs"]), native_checkpoint_sha256=provenance["checkpoint_sha256"])
            save_torch(target_path, item)
            path = output_path(out / "native/predictions" / (iid + ".npz")); temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **convert_prediction(pred.to("cpu"))); temp.replace(path)
            atomic_json(row_path, row); rows.append(row)
            if args.smoke:
                patch.update = dict(native_logits=item["baseline"], logits=item["baseline"], boxes=item["boxes"], size=item["size"])
                patch.enabled = True
                noop, _ = infer(model, cfg, transform, ds, mapping[iid])
                error = compare_predictions(pred, noop)
                if tensor_digest(patch.capture["relation_logits"]) != item["predicate_sha256"]:
                    raise RuntimeError("No-op changed predicate logits")
                patch.update["logits"] = item["baseline"] + torch.linspace(-.2, .2, 151)
                changed, _ = infer(model, cfg, transform, ds, mapping[iid])
                delta = float((pred.get_field("pred_scores") - changed.get_field("pred_scores")).abs().max())
                if delta <= 1e-8 or not patch.capture.get("route_checked"):
                    raise RuntimeError("Synthetic object update did not reach native scores")
                if tensor_digest(patch.capture["relation_logits"]) != item["predicate_sha256"]:
                    raise RuntimeError("Object update changed predicate logits")
                smoke_checks.append(dict(image_id=iid, noop_error=error, score_delta=delta, reference_errors=errors))
        if n % 10 == 0 or n == len(selected): progress(folder, args.task, "native_export", n, len(selected), started)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise RuntimeError("Native model unfrozen")
    patch.close()
    atomic_json(out / "native/summary.json", dict(status="complete", protocol_sha256=registration, **summarize_rows(rows)))
    atomic_json(folder / "summary.json", dict(status="complete", protocol_sha256=registration, images=len(selected),
        checkpoint_sha256=provenance["checkpoint_sha256"], smoke_checks=smoke_checks, synthetic_updates_discarded=True))


def evaluate(args):
    eligible(args.task, args.smoke)
    out, cache = paths(args.task, args.smoke); folder = out / "selective"
    registration = verify(full=True)
    features = read(out / "features/summary.json")
    if features["protocol_sha256"] != registration or features["checkpoint_sha256"] != PRIMARY_SHA:
        raise RuntimeError("V-K feature/update provenance differs")
    model, cfg, transform, provenance, patch = setup(args.task, folder)
    ds = dataset(cfg, "val"); mapping = lookup(ds); selected = ids(args.smoke)
    metric = OfficialMetrics(args.task); rows = []; started = time.monotonic()
    from export_pysgg_vg_task import convert_prediction
    patch.enabled = True
    progress(folder, args.task, "native_repaired_inference", 0, len(selected), started)
    for n, iid in enumerate(selected, 1):
        budget(args.deadline)
        row_path = folder / "images" / (iid + ".json")
        if row_path.exists():
            row = read(row_path)
            if row["protocol_sha256"] != registration: raise RuntimeError("Repaired row drift")
            rows.append(row)
        else:
            item = load_checked(cache / "export" / (iid + ".pt"), registration, iid)
            update = load_checked(cache / "updates" / (iid + ".pt"), registration, iid)
            if update["checkpoint_sha256"] != PRIMARY_SHA or not torch.equal(update["native_logits"], item["baseline"]):
                raise RuntimeError("Frozen update differs from exported proposal logits")
            patch.update = update
            pred, _ = infer(model, cfg, transform, ds, mapping[iid])
            c = patch.capture
            if (not c.get("route_checked") or tensor_digest(c["relation_logits"]) != item["predicate_sha256"]
                    or tensor_digest(c["pairs"]) != item["pair_sha256"]):
                raise RuntimeError("Predicate/structural outputs changed or object route bypassed")
            gt = ds.get_groundtruth(mapping[iid], evaluation=True); y = targets(c, gt, args.task)
            row = metric.row(iid, pred, gt, c, y)
            if row["positive_objects"] != read(out / "native/images" / (iid + ".json"))["positive_objects"]:
                raise RuntimeError("Identity denominator changed")
            row.update(protocol_sha256=registration, selected_proposals=int(update["selected"].sum()),
                predicate_logits_unchanged=True, native_postprocessing_checked=True)
            path = output_path(folder / "predictions" / (iid + ".npz")); temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **convert_prediction(pred.to("cpu"))); temp.replace(path)
            atomic_json(row_path, row); rows.append(row)
        if n % 10 == 0 or n == len(selected): progress(folder, args.task, "native_repaired_inference", n, len(selected), started)
    patch.close()
    atomic_json(folder / "summary.json", dict(status="complete", protocol_sha256=registration,
        smoke=args.smoke, checkpoint_sha256=PRIMARY_SHA, **summarize_rows(rows)))


def decision(args):
    if args.smoke: raise ValueError("A smoke run is not an acceptance gate")
    eligible(args.task); out, _ = paths(args.task); selected = ids()
    registration = verify(full=True)
    for mode in ["native", "selective"]:
        if read(out / mode / "summary.json")["protocol_sha256"] != registration:
            raise RuntimeError("Incomplete or incompatible gate stage")
    rows = {mode:[read(out / mode / "images" / (i + ".json")) for i in selected] for mode in ["native", "selective"]}
    for values in rows.values():
        if any(x["protocol_sha256"] != registration for x in values): raise RuntimeError("Gate row mismatch")
    result = paired_gate(rows["native"], rows["selective"])
    value = dict(status="complete", task=args.task, protocol_sha256=registration, **result,
        native=summarize_rows(rows["native"]), repaired=summarize_rows(rows["selective"]),
        checkpoint_sha256=PRIMARY_SHA, test_evaluated=False, extra_seeds_run=False,
        held_out_from_adapter_fitting=True, native_validation_previously_audited=True)
    atomic_json(OUT / args.task / "decision.json", value)
    print(__import__("json").dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["export", "evaluate", "decision"])
    p.add_argument("--task", choices=SPEC["tasks"], required=True)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args(); ensure_storage(); torch.set_num_threads(2); budget(a.deadline)
    if a.stage != "decision":
        configure_backend()
    {"export":export, "evaluate":evaluate, "decision":decision}[a.stage](a)


if __name__ == "__main__": main()
