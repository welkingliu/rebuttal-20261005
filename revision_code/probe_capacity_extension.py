"""R10: extend the registered linear/MLP sensitivity check to the other four encoders."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, OLD, atomic_json, output_path, ensure_storage, sha256, sole_match
from probe_capacity import (LinearObjectProbe, deterministic_image_split, frequency_groups,
    batched_logits, fit_temperature, evaluate_object_logits, relationship_endpoint_summary, paired_accuracy_delta)

OUT=ROOT/"results/R10_probe_capacity"
MODELS=["resnet50","siglip2_b","radio_v25_b","sam_vit_b"]
SPEC=dict(models=MODELS,seeds=[17,23,31],architectures=["linear","mlp"],view="pred_mask",
          hidden_dim=256,max_epochs=100,patience=10,device="cpu",threads=4,
          hyperparameters="same original per-encoder learning rate, weight decay and batch as R5; no test-based tuning",
          interpretation="Decoder-capacity sensitivity on supplied SAM-mask features; not autonomous or complete-system recognition",
          rationale="Complete all six original encoders, including the previously untested SAM endpoint, without adding new backbones")


def paths(model):
    return [sole_match("data/derived/features/experiment_1a/%s/*train_5000*.pt"%model),
            sole_match("data/derived/features/experiment_1a/%s/*eval_1000*.pt"%model),
            OLD/"artifacts/experiment_1a/exp1a_converged_20260718_174828"/model/"summary.json"]


def fingerprint():
    files=[Path(__file__),Path(__file__).with_name("probe_capacity.py"),Path(__file__).with_name("common.py"),
           OLD/"sgg_core/audits/object_grounding.py"]
    return dict(spec=SPEC,sources={str(p):sha256(p) for p in files},
                assets={m:{str(p):sha256(p) for p in paths(m)} for m in MODELS})


def save_torch(path,payload):
    path=output_path(path);temp=path.with_suffix(".tmp")
    torch.save(payload,temp);temp.replace(path)


def train(x,y,mask,val,classes,architecture,seed,cfg,progress):
    torch.manual_seed(seed);np.random.seed(seed)
    model=LinearObjectProbe(x.shape[1],classes) if architecture=="linear" else nn.Sequential(
        nn.Linear(x.shape[1],256),nn.ReLU(),nn.Linear(256,classes))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    indices=mask.nonzero().flatten();generator=torch.Generator().manual_seed(seed)
    best_loss=float("inf");best=None;stale=0;history=[]
    for epoch in range(1,101):
        started=time.monotonic();model.train();order=indices[torch.randperm(len(indices),generator=generator)];total=0.
        for start in range(0,len(order),cfg["probe_batch_size"]):
            ix=order[start:start+cfg["probe_batch_size"]]
            loss=F.cross_entropy(model(x[ix]),y[ix])
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
        logits=batched_logits(model,x[val],"cpu",cfg["probe_batch_size"])
        loss=float(F.cross_entropy(logits,y[val]))
        if not np.isfinite(loss):raise RuntimeError("Nonfinite validation loss")
        improved=loss<best_loss
        if improved:
            best_loss=loss;stale=0;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        row=dict(epoch=epoch,train_cross_entropy=total/len(order),validation_cross_entropy=loss,
                 improved=improved,seconds=time.monotonic()-started)
        history.append(row);progress(epoch,row)
        if stale>=10:break
    model.load_state_dict(best)
    return model,history


def run_model(name,registration,completed,started):
    out=OUT/name;tp,ep,sp=paths(name);source=json.loads(sp.read_text());cfg=source["config"]
    if cfg.get("feature_normalization","none")!="none":raise RuntimeError("Unexpected feature normalization")
    data=torch.load(tp,map_location="cpu",weights_only=False);test=torch.load(ep,map_location="cpu",weights_only=False)
    if set(data["image_ids"])&set(test["image_ids"]):raise RuntimeError("Train/test overlap")
    y,target=data["labels"].long(),test["labels"].long()
    mask,val=deterministic_image_split(data["image_ids"],cfg["validation_fraction"],seed=997)
    assert int(mask.sum())==source["image_disjoint_probe_split"]["train_objects"]
    assert int(val.sum())==source["image_disjoint_probe_split"]["validation_objects"]
    assert len(target)==source["eval_objects"]
    classes=source["num_classes"];groups=frequency_groups(y[mask],classes)
    x,z=data["views"]["pred_mask"].float(),test["views"]["pred_mask"].float()
    protocol=dict(registration_sha256=registration,source_summary=str(sp),model=name,classes=classes,
                  train_objects=int(mask.sum()),validation_objects=int(val.sum()),eval_objects=len(target),
                  ontology_id=source["ontology_id"],mask_protocol=source["predicted_mask_protocol"],spec=SPEC)
    pf=out/"protocol.json"
    if pf.exists() and json.loads(pf.read_text())!=protocol:raise RuntimeError("Protocol changed")
    atomic_json(pf,protocol);digest=sha256(pf);runs=[];predictions={}
    for seed in SPEC["seeds"]:
        for architecture in SPEC["architectures"]:
            stem="%s_seed%d"%(architecture,seed);rf=out/(stem+".json");npz=out/(stem+"_predictions.npz")
            def progress(epoch,row):
                elapsed=time.monotonic()-started
                atomic_json(OUT/"progress.json",dict(stage="probe_training",images=completed,total=24,
                    detail="%s | %s | seed %d | epoch %d/100 (early stopping)"%(name,architecture,seed,epoch),
                    seconds=elapsed,eta_seconds=elapsed/completed*(24-completed) if completed else None))
                print(json.dumps(dict(backbone=name,architecture=architecture,seed=seed,**row)),flush=True)
            if rf.exists() and npz.exists():
                row=json.loads(rf.read_text())
                if row["protocol_sha256"]!=digest:raise RuntimeError("Stale probe result")
                with np.load(npz,allow_pickle=False) as payload:pred= torch.from_numpy(payload["predictions"].copy())
            else:
                model,history=train(x,y,mask,val,classes,architecture,seed,cfg,progress)
                temperature=fit_temperature(batched_logits(model,x[val],"cpu",512),y[val])
                logits=batched_logits(model,z,"cpu",512);pred=logits.argmax(1)
                row=dict(seed=seed,architecture=architecture,history=history,device="cpu",
                    best_epoch=min(history,key=lambda r:r["validation_cross_entropy"])["epoch"],reached_epoch_cap=len(history)==100,
                    protocol_sha256=digest,temperature=temperature,parameter_count=sum(p.numel() for p in model.parameters()),
                    metrics=evaluate_object_logits(logits,target,test["image_ids"],groups,areas=test["areas"],mask_iou=test["mask_iou"],temperature=temperature),
                    relationship_endpoints=relationship_endpoint_summary(pred,target,test["graph_records"],mask_iou=test["mask_iou"],bootstrap_seed=seed+3000))
                save_torch(ROOT/"checkpoints/R10_probe_capacity"/name/(stem+".pt"),dict(state_dict=model.state_dict(),protocol=protocol,architecture=architecture,temperature=temperature))
                temp=output_path(npz.with_suffix(".tmp"))
                with temp.open("wb") as stream:np.savez_compressed(stream,logits=logits.numpy(),predictions=pred.numpy(),labels=target.numpy(),image_ids=np.asarray(test["image_ids"]))
                temp.replace(npz);atomic_json(rf,row);del model
            runs.append(row);predictions[(architecture,seed)]=pred;completed+=1
    paired={str(seed):paired_accuracy_delta(predictions[("linear",seed)],predictions[("mlp",seed)],target,test["image_ids"],seed=seed+1000) for seed in SPEC["seeds"]}
    atomic_json(out/"summary.json",dict(status="complete",protocol=protocol,runs=runs,paired_mlp_minus_linear=paired))
    return completed


def main():
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");args=p.parse_args()
    ensure_storage();torch.set_num_threads(4)
    pf=ROOT/"manifests/R10_probe_capacity.json";contract=fingerprint()
    if args.register:
        if pf.exists() and json.loads(pf.read_text())["contract"]!=contract:raise RuntimeError("Already registered differently")
        if not pf.exists():atomic_json(pf,dict(contract=contract,registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print("[REGISTERED] Four remaining encoders, CPU, 24 paired probe fits",flush=True);return
    if json.loads(pf.read_text())["contract"]!=contract:raise RuntimeError("Source or assets changed")
    lock=(ROOT/"status/R10_probe_capacity.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status="running",pid=os.getpid(),gpu=[],command=[sys.executable]+sys.argv,
               completion=str(OUT/"summary.json"),progress_file=str(OUT/"progress.json"),log=str(ROOT/"logs/R10_probe_capacity.log"))
    atomic_json(ROOT/"status/R10_probe_capacity.json",state)
    completed=0;started=time.monotonic()
    try:
        for name in MODELS:completed=run_model(name,sha256(pf),completed,started)
        atomic_json(OUT/"summary.json",dict(status="complete",models=MODELS,completed_runs=completed,
                    previous_models=["dinov2_b","cradio_v4_so400m"],seconds=time.monotonic()-started,
                    note="Combine with R5 to cover six encoders; each pair uses the same split and device, no superiority guarantee"))
        atomic_json(ROOT/"status/R10_probe_capacity.json",dict(state,status="complete"))
    except Exception as exc:
        atomic_json(ROOT/"status/R10_probe_capacity.json",dict(state,status="failed",error=str(exc)))
        raise


if __name__=="__main__":main()
