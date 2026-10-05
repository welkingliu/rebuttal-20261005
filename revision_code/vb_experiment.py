"""Finite native V-B stages. Outputs are isolated from all old experiments."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from native_runtime import load_model, dataset, image_id, infer
from vb_protocol import (SPEC, VB, KS, ResidualHead, objective, choose_ids,
                         protocol_hash, verify_registration, summarize_rows, paired_gate)
from vb_native import ContextPatch, OfficialMetrics, compare_predictions, targets


def save_npz(path,**data):
    path=output_path(path)
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f,**data)
    temp.replace(path)


def save_state(path,value):
    path=output_path(path)
    temp=path.with_suffix(".tmp")
    torch.save(value,str(temp))
    temp.replace(path)


def cache_root(task):
    return ROOT/"cache/VB"/task


def checkpoint(task,mode,seed):
    return ROOT/"checkpoints/VB"/task/(mode+"_seed%d.pth"%seed)


def protocol(task):
    return json.loads((cache_root(task)/"protocol.json").read_text())


def setup(task,out):
    model,cfg,transform,provenance=load_model("tde_motifs",task,out)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model,cfg,transform,provenance,ContextPatch(model)


def prepare(args):
    out=cache_root(args.task)
    model,cfg,transform,provenance,patch=setup(args.task,out)
    train,validation=dataset(cfg,"train"),dataset(cfg,"val")
    tr={image_id(train,i):i for i in range(len(train))}
    va={image_id(validation,i):i for i in range(len(validation))}
    if set(tr)&set(va):
        raise RuntimeError("Native training and validation overlap")
    train_ids=choose_ids(tr,SPEC["train_images"],"VB_train:")
    val_ids=choose_ids(va,SPEC["development_images"]+SPEC["gate_images"],"VB_val:")
    dev=val_ids[:SPEC["development_images"]]
    gate=val_ids[SPEC["development_images"]:]
    if len(dev)!=500 or len(gate)!=1000 or len(train_ids)!=5000:
        raise RuntimeError("Incomplete V-B splits")
    record=dict(spec_hash=protocol_hash(),model=provenance,train=train_ids,development=dev,gate=gate)
    path=out/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=record:
        raise RuntimeError("Cache provenance changed")
    atomic_json(path,record)
    # Exercise actual native post-processing before the long cache extraction.
    errors=[]
    for iid in dev[:3]:
        patch.enabled=False
        base,_=infer(model,cfg,transform,validation,va[iid])
        width=patch.capture["features"].shape[1]
        if width!=512:
            raise RuntimeError("Registered context width is 512")
        patch.head=ResidualHead(width).cuda().eval()
        patch.enabled=True
        zero,_=infer(model,cfg,transform,validation,va[iid])
        errors.append(compare_predictions(base,zero))
    if not patch.average_calls:
        raise RuntimeError("Expected frozen TDE average-context branch was not exercised")
    atomic_json(out/"integration_check.json",dict(status="complete",zero_update_max_errors=errors,
                task=args.task,average_branch_calls=patch.average_calls,native_parameters_frozen=True))
    patch.enabled=False
    patch.head=None
    started=time.monotonic()
    for split,ds,mapping,ids in [("train",train,tr,train_ids),("development",validation,va,dev)]:
        for number,iid in enumerate(ids):
            dest=out/split/(iid+".npz")
            if dest.exists():
                with np.load(dest) as z:
                    if str(z["protocol_sha256"].item())!=sha256(path):
                        raise RuntimeError("Stale V-B cached feature")
                continue
            infer(model,cfg,transform,ds,mapping[iid])
            captured=patch.capture
            gt=ds.get_groundtruth(mapping[iid],evaluation=True)
            target=targets(captured,gt,args.task)
            save_npz(dest,features=captured["features"].float().cpu().numpy(),
                     logits=captured["base_logits"].float().cpu().numpy(),targets=target,
                     protocol_sha256=np.asarray(sha256(path)))
            if (number+1)%50==0:
                row=dict(split=split,images=number+1,total=len(ids),seconds=time.monotonic()-started)
                atomic_json(out/"progress.json",row);print(json.dumps(row),flush=True)
    patch.close()
    atomic_json(out/"summary.json",dict(status="complete",train_images=5000,development_images=500,
                gate_images=1000,protocol_sha256=sha256(path),seconds=time.monotonic()-started))


def load_features(task,split):
    p=protocol(task)
    rows=[]
    for iid in p[split]:
        with np.load(cache_root(task)/split/(iid+".npz")) as z:
            if str(z["protocol_sha256"].item())!=sha256(cache_root(task)/"protocol.json"):
                raise RuntimeError("Feature provenance mismatch")
            rows.append([torch.from_numpy(z[k].copy()) for k in ("features","logits","targets")])
    return tuple(torch.cat([r[j] for r in rows]) for j in range(3))


def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=VB/args.task/"training"/(args.mode+"_seed%d"%args.seed)
    x,z,y=load_features(args.task,"train")
    vx,vz,vy=load_features(args.task,"development")
    positive=vy>0
    head=ResidualHead(x.shape[1])
    optim=torch.optim.AdamW(head.parameters(),lr=SPEC["learning_rate"],weight_decay=SPEC["weight_decay"])
    # Check actual training gradients before committing to epochs.
    valid=(y>=0).nonzero().flatten()[:512]
    loss,_,_=objective(head,x[valid],z[valid],y[valid],SPEC["kl_weights"][args.mode])
    loss.backward()
    grad=sum(float(p.grad.square().sum()) for p in head.parameters() if p.grad is not None)**.5
    if not np.isfinite(grad) or grad==0:
        raise RuntimeError("Residual training has no finite nonzero gradient")
    optim.zero_grad(set_to_none=True)
    best=float("inf");stale=0;history=[]
    g=torch.Generator().manual_seed(args.seed)
    eligible=(y>=0).nonzero().flatten()
    for epoch in range(1,SPEC["max_epochs"]+1):
        head.train();total=0.;count=0
        order=eligible[torch.randperm(len(eligible),generator=g)]
        for i in range(0,len(order),SPEC["batch_size"]):
            ix=order[i:i+SPEC["batch_size"]]
            loss,ce,kl=objective(head,x[ix],z[ix],y[ix],SPEC["kl_weights"][args.mode])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training objective")
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            optim.step();total+=float(loss)*len(ix);count+=len(ix)
        head.eval()
        with torch.no_grad():
            v=head(vx,vz)
            val=float(F.cross_entropy(v[positive],vy[positive]))
        row=dict(epoch=epoch,loss=total/count,development_foreground_nll=val)
        history.append(row);print(json.dumps(row),flush=True)
        atomic_json(out/"progress.json",row)
        if val<best-1e-6:
            best=val;stale=0
            save_state(checkpoint(args.task,args.mode,args.seed),dict(state=head.state_dict(),
                task=args.task,mode=args.mode,seed=args.seed,epoch=epoch,spec_hash=protocol_hash(),
                model=protocol(args.task)["model"],cache_protocol_sha256=sha256(cache_root(args.task)/"protocol.json")))
        else:
            stale+=1
        if stale>=SPEC["patience"]:
            break
    atomic_json(out/"summary.json",dict(status="complete",history=history,best_development_nll=best,
        checkpoint=str(checkpoint(args.task,args.mode,args.seed)),checkpoint_sha256=sha256(checkpoint(args.task,args.mode,args.seed)),
        training_objects=len(eligible),background_objects=int((y==0).sum()),ignored_objects=int((y<0).sum()),
        first_object_gradient_norm=grad,relation_loss_used=False))
    temp_path=cache_root(args.task)/"temperature.json"
    if not temp_path.exists():
        # Same positive/background weighting as the task-matched object objective.
        q=torch.nn.Parameter(torch.zeros(()))
        op=torch.optim.LBFGS([q],lr=.1,max_iter=50,line_search_fn="strong_wolfe")
        mask=vy>=0
        weight=torch.ones(151);weight[0]=SPEC["background_weight"]
        def closure():
            op.zero_grad()
            loss=F.cross_entropy(vz[mask]/q.clamp(-3,3).exp(),vy[mask],weight=weight)
            loss.backward();return loss
        op.step(closure)
        temperature=float(q.detach().clamp(-3,3).exp())
        atomic_json(temp_path,dict(temperature=temperature,split="development",images=500,
                    representation="dense native logits",spec_hash=protocol_hash()))


def evaluate(args):
    if args.split=="test":
        if not json.loads((VB/"pilot_decision.json").read_text())["accepted"]:
            raise RuntimeError("Test access blocked by failed pilot gate")
    out=VB/args.task/args.split/(args.mode+"_seed%d"%args.seed)
    model,cfg,transform,provenance,patch=setup(args.task,out)
    if provenance["checkpoint_sha256"]!=protocol(args.task)["model"]["checkpoint_sha256"]:
        raise RuntimeError("Task-specific base checkpoint changed")
    ds=dataset(cfg,"test" if args.split=="test" else "val")
    lookup={image_id(ds,i):i for i in range(len(ds))}
    ids=list(lookup) if args.split=="test" else protocol(args.task)["gate"]
    if args.split=="test" and len(ids)!=26446:
        raise RuntimeError("Incomplete formal test split")
    if set(ids)&set(protocol(args.task)["train"]+protocol(args.task)["development"]):
        raise RuntimeError("Train/development leakage into evaluation")
    state_sha=None
    if args.mode in SPEC["kl_weights"]:
        state=torch.load(str(checkpoint(args.task,args.mode,args.seed)),map_location="cpu")
        if state["spec_hash"]!=protocol_hash() or state["task"]!=args.task or state["model"]["checkpoint_sha256"]!=provenance["checkpoint_sha256"]:
            raise RuntimeError("Incompatible residual checkpoint")
        patch.head=ResidualHead().cuda().eval();patch.head.load_state_dict(state["state"])
        state_sha=sha256(checkpoint(args.task,args.mode,args.seed))
    if args.mode=="temperature":
        patch.temperature=json.loads((cache_root(args.task)/"temperature.json").read_text())["temperature"]
    patch.enabled=args.mode!="native"
    eval_protocol=dict(spec_hash=protocol_hash(),model=provenance,split=args.split,image_ids=ids,
                       mode=args.mode,seed=args.seed,state_sha256=state_sha,temperature=patch.temperature,
                       downstream="native edge context, TDE predicate computation, class NMS and ranking rerun")
    pf=out/"protocol.json"
    if pf.exists() and json.loads(pf.read_text())!=eval_protocol:
        raise RuntimeError("Evaluation resume protocol changed")
    atomic_json(pf,eval_protocol)
    metrics=OfficialMetrics(args.task);rows=[];started=time.monotonic()
    for number,iid in enumerate(ids):
        rf=out/"images"/(iid+".json")
        prediction_path=out/"predictions"/(iid+".npz")
        if rf.exists() and prediction_path.exists():
            row=json.loads(rf.read_text())
            if row["protocol_sha256"]!=sha256(pf):
                raise RuntimeError("Stale per-image evaluation")
            rows.append(row);continue
        pred,_=infer(model,cfg,transform,ds,lookup[iid])
        gt=ds.get_groundtruth(lookup[iid],evaluation=True)
        target=targets(patch.capture,gt,args.task)
        row=metrics.row(iid,pred,gt,patch.capture,target)
        row["protocol_sha256"]=sha256(pf)
        w,h=pred.size
        save_npz(prediction_path,
            pred_boxes=pred.bbox.float().cpu().numpy()/np.asarray([w,h,w,h],dtype=np.float32),
            native_labels=pred.get_field("pred_labels").cpu().numpy(),
            native_scores=pred.get_field("pred_scores").cpu().numpy(),
            object_logits=patch.capture["logits"].float().cpu().numpy(),
            pred_rel_pairs=pred.get_field("rel_pair_idxs").cpu().numpy(),
            pred_rel_scores=pred.get_field("pred_rel_scores").cpu().numpy(),
            input_proposal_targets=target,protocol_sha256=np.asarray(sha256(pf)))
        atomic_json(rf,row);rows.append(row)
        if (number+1)%50==0:
            progress=dict(images=number+1,total=len(ids),seconds=time.monotonic()-started)
            atomic_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
    patch.close()
    atomic_json(out/"summary.json",dict(status="complete",metrics=summarize_rows(rows),
                task=args.task,mode=args.mode,seed=args.seed,split=args.split,
                protocol_sha256=sha256(pf),seconds=time.monotonic()-started))


def read_rows(task,split,mode,seed):
    out=VB/task/split/(mode+"_seed%d"%seed)
    summary=json.loads((out/"summary.json").read_text())
    if summary["status"]!="complete":
        raise RuntimeError("Evaluation incomplete")
    ids=json.loads((out/"protocol.json").read_text())["image_ids"]
    return [json.loads((out/"images"/(iid+".json")).read_text()) for iid in ids]


def decide(args):
    decisions={}
    for task in SPEC["tasks"]:
        decisions[task]=paired_gate(read_rows(task,"gate","native",17),read_rows(task,"gate","conservative",17))
    accepted=all(v["accepted"] for v in decisions.values())
    atomic_json(VB/"pilot_decision.json",dict(status="complete",accepted=accepted,tasks=decisions,
                spec_hash=protocol_hash(),next_action="three_seed_formal" if accepted else "stop_expansion_report_boundary"))
    print(json.dumps(dict(accepted=accepted,tasks=decisions)),flush=True)


def report(args):
    results={}
    for task in SPEC["tasks"]:
        base=read_rows(task,"test","native",17)
        task_rows={"native":summarize_rows(base)}
        conditions=[("temperature",17)]+[(m,s) for s in SPEC["seeds"] for m in SPEC["kl_weights"]]
        for mode,seed in conditions:
            rows=read_rows(task,"test",mode,seed)
            stats=paired_gate(base,rows)
            # The gate's sample-size check is irrelevant on the formal split.
            task_rows[mode+"_seed%d"%seed]=dict(metrics=summarize_rows(rows),delta=stats["delta"],
                paired_image_bootstrap_one_sided_95_lower=stats["paired_image_bootstrap_one_sided_95_lower"])
        results[task]=task_rows
    atomic_json(VB/"summary.json",dict(status="complete",phase="three_seed_formal",results=results,
                pilot_decision=json.loads((VB/"pilot_decision.json").read_text()),
                interpretation="All seeds retained; pilot acceptance is not a guarantee of test improvement"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","train","evaluate","decide","report"])
    p.add_argument("--task",choices=SPEC["tasks"])
    p.add_argument("--mode",choices=["native","temperature","supervised","conservative"],default="conservative")
    p.add_argument("--seed",type=int,choices=SPEC["seeds"],default=17)
    p.add_argument("--split",choices=["gate","test"],default="gate")
    args=p.parse_args();ensure_storage();verify_registration();torch.set_num_threads(4)
    if args.stage not in ("decide","report") and args.task is None:
        p.error("--task is required")
    dict(prepare=prepare,train=train,evaluate=evaluate,decide=decide,report=report)[args.stage](args)
    print("[COMPLETE] V-B",args.stage,args.task,args.mode,args.seed,flush=True)


if __name__=="__main__":
    main()
