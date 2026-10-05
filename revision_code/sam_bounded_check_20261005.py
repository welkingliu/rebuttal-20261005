"""SAM-only bounded refit, selected by validation NLL, never test outcomes.

Restart from the original seed because historical optimizer states were not
saved. This is not a continuation from the validation-selected checkpoint.
"""
import json
import os
from pathlib import Path
import time

import torch

from common import ROOT, OLD, atomic_json, ensure_storage, sha256
from probe_capacity_extension import paths, save_torch
from probe_convergence import train
from probe_capacity import (deterministic_image_split, frequency_groups, batched_logits,
    fit_temperature, evaluate_object_logits, relationship_endpoint_summary)

OUT = ROOT / "results/SAM_bounded_20261005"


def main():
    ensure_storage()
    torch.set_num_threads(4)
    tp, ep, sp = paths("sam_vit_b")
    spec = dict(model="sam_vit_b", architecture="linear", seeds=[17,23,31], max_epochs=2000,
                patience=20, view="pred_mask", selection="minimum validation NLL only",
                test_use="one evaluation after selection; previously inspected test, not independent confirmation",
                refit="original initialization and data order; not checkpoint continuation",
                source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name("probe_convergence.py"),tp,ep,sp]})
    protocol = OUT / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise RuntimeError("Sealed protocol differs")
    atomic_json(protocol, spec)
    old = json.loads(sp.read_text())
    cfg = old["config"]
    data = torch.load(tp, map_location="cpu", weights_only=False)
    test = torch.load(ep, map_location="cpu", weights_only=False)
    if set(data["image_ids"]) & set(test["image_ids"]):
        raise RuntimeError("Train/test overlap")
    mask, val = deterministic_image_split(data["image_ids"], cfg["validation_fraction"], seed=997)
    for name, actual in [("train_objects",int(mask.sum())),("validation_objects",int(val.sum()))]:
        if old["image_disjoint_probe_split"][name] != actual:
            raise RuntimeError("Split changed")
    y, target = data["labels"].long(), test["labels"].long()
    x, z = data["views"]["pred_mask"].float(), test["views"]["pred_mask"].float()
    groups = frequency_groups(y[mask], old["num_classes"])
    rows = []
    started = time.time()
    for index, seed in enumerate(spec["seeds"]):
        def progress(epoch, row):
            state = dict(status="running", pid=os.getpid(), seed=seed, completed_seeds=index,
                         epoch=epoch, max_epochs=2000, validation_nll=row["validation_cross_entropy"],
                         elapsed_seconds=time.time()-started)
            atomic_json(OUT / "progress.json", state)
            if epoch == 1 or epoch % 25 == 0:
                print(json.dumps(state), flush=True)
        model, history = train(x,y,mask,val,old["num_classes"],"linear",seed,cfg,progress,max_epochs=2000,patience=20)
        temperature = fit_temperature(batched_logits(model,x[val],"cpu",512),y[val])
        logits = batched_logits(model,z,"cpu",512)
        result = dict(seed=seed, best_epoch=min(history,key=lambda h:h["validation_cross_entropy"])["epoch"],
                      reached_epoch_cap=len(history)==2000, history=history,
                      metrics=evaluate_object_logits(logits,target,test["image_ids"],groups,areas=test["areas"],mask_iou=test["mask_iou"],temperature=temperature),
                      relationship_endpoints=relationship_endpoint_summary(logits.argmax(1),target,test["graph_records"],mask_iou=test["mask_iou"],bootstrap_seed=seed+3000))
        atomic_json(OUT / ("seed_%d.json"%seed),result)
        save_torch(ROOT / "checkpoints/SAM_bounded_20261005" / ("seed_%d.pt"%seed),dict(state_dict=model.state_dict(),temperature=temperature,protocol_sha256=sha256(protocol)))
        rows.append(result)
        print("[SEED COMPLETE] " + str(seed) + " best_epoch=" + str(result["best_epoch"]),flush=True)
    atomic_json(OUT/"summary.json",dict(status="complete",protocol=spec,runs=rows))
    atomic_json(OUT/"progress.json",dict(status="complete",completed_seeds=3,elapsed_seconds=time.time()-started))
    print("[COMPLETE] SAM bounded refit; do not extend based on test results",flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(OUT/"progress.json",dict(status="failed",error=repr(exc)))
        raise
