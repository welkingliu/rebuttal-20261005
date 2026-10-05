"""Follow-up to R5/R10: the same six-encoder capacity comparison with a longer cap.

The cap is increased because SAM's validation NLL was still improving at epoch
100, not because of its held-out accuracy. Test metrics never select a fit.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, OLD, atomic_json, ensure_storage, sha256
from probe_capacity_extension import paths, save_torch
from probe_capacity import (LinearObjectProbe, deterministic_image_split, frequency_groups,
    batched_logits, fit_temperature, evaluate_object_logits, relationship_endpoint_summary, paired_accuracy_delta)

OUT = ROOT / "results/probe_convergence"
MODELS = ["sam_vit_b", "resnet50", "dinov2_b", "siglip2_b", "radio_v25_b", "cradio_v4_so400m"]
SPEC = dict(models=MODELS, seeds=[17, 23, 31], architectures=["linear", "mlp"],
            view="pred_mask", hidden_dim=256, max_epochs=500, patience=20, device="cpu", threads=4,
            selection="Minimum validation NLL only; original per-model learning rate, decay, split and batch",
            trigger="SAM linear best validation epoch equals the previous 100-epoch cap",
            interpretation="Decoder and optimization sensitivity of supplied-mask probes, not end-to-end recognition")


def train(x, y, mask, val, classes, architecture, seed, cfg, progress, max_epochs=500, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LinearObjectProbe(x.shape[1], classes) if architecture == "linear" else nn.Sequential(
        nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Linear(256, classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    indices = mask.nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    best_loss, best, stale, history = float("inf"), None, 0, []
    for epoch in range(1, max_epochs + 1):
        started = time.monotonic()
        model.train()
        order = indices[torch.randperm(len(indices), generator=generator)]
        total = 0.
        for start in range(0, len(order), cfg["probe_batch_size"]):
            ix = order[start:start + cfg["probe_batch_size"]]
            loss = F.cross_entropy(model(x[ix]), y[ix])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        logits = batched_logits(model, x[val], "cpu", cfg["probe_batch_size"])
        loss = float(F.cross_entropy(logits, y[val]))
        if not np.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss")
        improved = loss < best_loss
        if improved:
            best_loss, stale = loss, 0
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        row = dict(epoch=epoch, train_cross_entropy=total/len(order), validation_cross_entropy=loss,
                   improved=improved, seconds=time.monotonic()-started)
        history.append(row)
        progress(epoch, row)
        if stale >= patience:
            break
    model.load_state_dict(best)
    return model, history


def contract():
    files = [Path(__file__), Path(__file__).with_name("probe_capacity.py"),
             Path(__file__).with_name("probe_capacity_extension.py"), Path(__file__).with_name("common.py"),
             OLD/"sgg_core/audits/object_grounding.py"]
    return dict(spec=SPEC, sources={str(p): sha256(p) for p in files},
                assets={model: {str(p): sha256(p) for p in paths(model)} for model in MODELS})


def run(registration):
    completed, started, summaries = 0, time.monotonic(), {}
    for name in MODELS:
        tp, ep, sp = paths(name)
        source = json.loads(sp.read_text())
        cfg = source["config"]
        if cfg.get("feature_normalization", "none") != "none":
            raise RuntimeError("Unexpected feature normalization")
        data = torch.load(tp, map_location="cpu", weights_only=False)
        test = torch.load(ep, map_location="cpu", weights_only=False)
        if set(data["image_ids"]) & set(test["image_ids"]):
            raise RuntimeError("Train/test overlap")
        y, target = data["labels"].long(), test["labels"].long()
        mask, val = deterministic_image_split(data["image_ids"], cfg["validation_fraction"], seed=997)
        if int(mask.sum()) != source["image_disjoint_probe_split"]["train_objects"]:
            raise RuntimeError("Changed training split")
        if int(val.sum()) != source["image_disjoint_probe_split"]["validation_objects"]:
            raise RuntimeError("Changed validation split")
        if len(target) != source["eval_objects"]:
            raise RuntimeError("Changed test support")
        classes = source["num_classes"]
        groups = frequency_groups(y[mask], classes)
        x, z = data["views"]["pred_mask"].float(), test["views"]["pred_mask"].float()
        rows, predictions = [], {}
        for seed in SPEC["seeds"]:
            for architecture in SPEC["architectures"]:
                stem = "%s_seed%d" % (architecture, seed)
                result_path = OUT/name/(stem+".json")
                prediction_path = OUT/name/(stem+"_predictions.npz")
                def progress(epoch, row):
                    elapsed = time.monotonic()-started
                    atomic_json(OUT/"progress.json", dict(stage="probe_training", images=completed, total=36,
                        detail="%s | %s | seed %d | epoch %d/500" % (name, architecture, seed, epoch),
                        seconds=elapsed, eta_seconds=elapsed/completed*(36-completed) if completed else None))
                    print(json.dumps(dict(model=name, architecture=architecture, seed=seed, **row)), flush=True)
                if result_path.exists() and prediction_path.exists():
                    row = json.loads(result_path.read_text())
                    if row["registration_sha256"] != registration:
                        raise RuntimeError("Result registration mismatch")
                    with np.load(prediction_path, allow_pickle=False) as payload:
                        pred = torch.from_numpy(payload["predictions"].copy())
                else:
                    model, history = train(x, y, mask, val, classes, architecture, seed, cfg, progress)
                    temperature = fit_temperature(batched_logits(model, x[val], "cpu", 512), y[val])
                    logits = batched_logits(model, z, "cpu", 512)
                    pred = logits.argmax(1)
                    best_epoch = min(history, key=lambda row: row["validation_cross_entropy"])["epoch"]
                    row = dict(seed=seed, architecture=architecture, history=history, best_epoch=best_epoch,
                        reached_epoch_cap=len(history)==500, validation_selected_at_cap=best_epoch==500,
                        registration_sha256=registration, temperature=temperature,
                        metrics=evaluate_object_logits(logits, target, test["image_ids"], groups,
                            areas=test["areas"], mask_iou=test["mask_iou"], temperature=temperature),
                        relationship_endpoints=relationship_endpoint_summary(pred, target, test["graph_records"],
                            mask_iou=test["mask_iou"], bootstrap_seed=seed+3000))
                    save_torch(ROOT/"checkpoints/probe_convergence"/name/(stem+".pt"),
                               dict(state_dict=model.state_dict(), registration_sha256=registration,
                                    architecture=architecture, temperature=temperature))
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = prediction_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, logits=logits.numpy(), predictions=pred.numpy(),
                                            labels=target.numpy(), image_ids=np.asarray(test["image_ids"]))
                    temporary.replace(prediction_path)
                    atomic_json(result_path, row)
                    del model
                rows.append(row)
                predictions[(architecture, seed)] = pred
                completed += 1
        paired = {str(seed): paired_accuracy_delta(predictions[("linear", seed)], predictions[("mlp", seed)],
                   target, test["image_ids"], seed=seed+1000) for seed in SPEC["seeds"]}
        summary = dict(status="complete", source_summary=str(sp), spec=SPEC, registration_sha256=registration,
                       runs=rows, paired_mlp_minus_linear=paired)
        atomic_json(OUT/name/"summary.json", summary)
        summaries[name] = dict(paired=paired, runs_at_cap=sum(row["reached_epoch_cap"] for row in rows),
                               validation_selected_at_cap=sum(row["validation_selected_at_cap"] for row in rows))
    atomic_json(OUT/"summary.json", dict(status="complete", completed_runs=completed, models=summaries,
                                        seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    path = ROOT/"manifests/probe_convergence.json"
    fingerprint = contract()
    if args.register:
        if path.exists() and json.loads(path.read_text())["contract"] != fingerprint:
            raise RuntimeError("Already registered differently")
        if not path.exists():
            atomic_json(path, dict(contract=fingerprint, registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print("[REGISTERED] Six encoders, 36 CPU probe fits, cap 500 / patience 20", flush=True)
        return
    if json.loads(path.read_text())["contract"] != fingerprint:
        raise RuntimeError("Source or assets changed")
    lock = (ROOT/"status/probe_convergence.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status="running", pid=os.getpid(), gpu=[], completion=str(OUT/"summary.json"),
                 progress_file=str(OUT/"progress.json"), log=str(ROOT/"logs/probe_convergence.log"))
    atomic_json(ROOT/"status/probe_convergence.json", state)
    try:
        run(sha256(path))
        atomic_json(ROOT/"status/probe_convergence.json", dict(state, status="complete"))
    except Exception as exc:
        atomic_json(ROOT/"status/probe_convergence.json", dict(state, status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()
