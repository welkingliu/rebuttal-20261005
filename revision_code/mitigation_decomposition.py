"""R3: dense-logit calibration and fixed-candidate label/score decomposition.

Crossed label/score conditions use explicitly independent candidate factors,
not a probability assigned to a mismatched class. This is an offline ranking
analysis and does not claim a rerun of NMS, localization, or relation reasoning.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, ensure_storage, atomic_json, sha256
from sgg_core.audits.object_grounding import fit_temperature


KS = [1,5,10,20,50,100]


def softmax(x):
    x=x-x.max(-1,keepdims=True)
    p=np.exp(x)
    return p/p.sum(-1,keepdims=True)


def iou(a,b):
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,2:],b[None,:,2:])
    area=lambda x:np.maximum(x[:,2]-x[:,0],0)*np.maximum(x[:,3]-x[:,1],0)
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum(area(a)[:,None]+area(b)[None,:]-inter,1e-12)


def matches(z, labels, confidence):
    pairs=z["pred_rel_pairs"].astype(int)
    rel=z["pred_rel_scores"]
    predicates=rel[:,1:].argmax(1)+1
    weights=confidence[pairs[:,0]]*confidence[pairs[:,1]]*rel[:,1:].max(1)
    order=np.argsort(-weights,kind="stable")[:100]
    gt=z["gt_relations"].astype(int)
    obj_gt=z["gt_labels"].astype(int)
    overlaps=iou(z["pred_boxes"],z["gt_boxes"])
    hits=[]
    for idx in order:
        s,o=pairs[idx]
        compatible=(labels[s]==obj_gt[gt[:,0]])&(labels[o]==obj_gt[gt[:,1]])&(predicates[idx]==gt[:,2])
        compatible&=(overlaps[s,gt[:,0]]>=.5)&(overlaps[o,gt[:,1]]>=.5)
        hits.append(compatible)
    rows=[]
    union=np.zeros(len(gt),dtype=bool)
    for k in KS:
        if hits:
            union=np.logical_or.reduce(hits[:k])
        counts=np.bincount(gt[:,2],minlength=51)[1:]
        counts_hit=np.bincount(gt[union,2],minlength=51)[1:]
        rows.append((float(union.mean()),counts_hit,counts))
    return rows,order


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["sgcls","sgdet"],required=True)
    args=p.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    out=ROOT/"results/R3"/args.task
    out.mkdir(parents=True,exist_ok=True)
    val=ROOT/"cache/R3_dense/validation/sgcls"
    test=ROOT/"cache/R3_dense/test"/args.task
    for folder in [val,test]:
        if json.loads((folder/"summary.json").read_text())["status"]!="complete":
            raise RuntimeError("Incomplete dense cache")
    val_ids=json.loads((val/"protocol.json").read_text())["image_ids"]
    test_ids=json.loads((test/"protocol.json").read_text())["image_ids"]
    if set(val_ids)&set(test_ids):
        raise RuntimeError("Temperature fitting would leak test images")
    val_logits,val_gt=[],[]
    for iid in val_ids:
        with np.load(val/"images"/(iid+".npz")) as z:
            assert len(z["object_logits"])==len(z["gt_labels"])
            val_logits.append(torch.from_numpy(z["object_logits"].copy()))
            val_gt.append(torch.from_numpy(z["gt_labels"].copy()).long())
    temperature=fit_temperature(torch.cat(val_logits),torch.cat(val_gt))
    states={}
    for file in sorted((ROOT/"imported/experiment5").rglob("mitigated_state_dict.pth")):
        state=torch.load(file,map_location="cpu",weights_only=False)
        if "grounding_state_dict" in state:
            if state.get("base_checkpoint_sha256") != "467da372633dbd77720cd3e7e5cc056552b86d957dd0ef4d571757e0786fc674":
                raise RuntimeError("Historical adaptation used a different base SGCls checkpoint")
            state=state["grounding_state_dict"]
        if "state_dict" in state:
            state=state["state_dict"]
        rw,rb=state["relation_calibrator.weight"],state["relation_calibrator.bias"]
        if not torch.allclose(rw,torch.eye(51),atol=1e-7) or not torch.allclose(rb,torch.zeros(51),atol=1e-7):
            raise RuntimeError("Historical state changed relations; independent object-only decomposition invalid")
        key=file.parents[2].name+"_"+file.parent.name
        states[key]=(state["entity_calibrator.weight"].numpy(),state["entity_calibrator.bias"].numpy(),str(file),sha256(file))
    if len(states)!=6:
        raise RuntimeError("Expected all six historical states")
    names=["native","temperature_only"]+[s+"/"+v for s in states for v in ["labels_only","scores_only","joint"]]
    n=len(test_ids)
    recalls={name:np.zeros((n,len(KS))) for name in names}
    class_sums={name:np.zeros((len(KS),50)) for name in names}
    class_n=np.zeros(50)
    calibration={name:dict(correct=0,count=0,bins=np.zeros((15,3))) for name in names}
    top50_overlap={name:[] for name in names}
    protocol=dict(task=args.task,temperature=temperature,temperature_fit_images=len(val_ids),
        temperature_objective="151-way object NLL on original held-out validation images only",
        cache_sha256=sha256(test/"protocol.json"),historical_states={k:dict(path=v[2],sha256=v[3]) for k,v in states.items()},
        native_labels="post-NMS native predicted labels",updated_labels="argmax foreground of adapted dense refined logits",
        native_scores="native per-proposal confidence",updated_scores="max foreground probability of adapted dense refined logits",
        crossing="Label and score are independent per-proposal factors, NOT probabilities of crossed labels",
        fixed=["boxes","proposal IDs","relation pairs","predicate probabilities"],
        interpretation="Offline fixed-candidate decomposition; not a rerun of complete detection/NMS")
    atomic_json(out/"protocol.json",protocol)
    for index,iid in enumerate(test_ids):
        with np.load(test/"images"/(iid+".npz")) as f:
            z={k:f[k] for k in f.files}
        labels=z["native_labels"].astype(int)
        scores=z["native_scores"].astype(float)
        logits=z["object_logits"].astype(np.float32)
        pt=softmax(logits/temperature)
        variants=dict(native=(labels,scores),temperature_only=(labels,pt[np.arange(len(labels)),labels]))
        for name,(w,b,_,_) in states.items():
            prob=softmax(logits@w.T+b)
            new_labels=prob[:,1:].argmax(1)+1
            new_scores=prob[:,1:].max(1)
            variants[name+"/labels_only"]=(new_labels,scores)
            variants[name+"/scores_only"]=(labels,new_scores)
            variants[name+"/joint"]=(new_labels,new_scores)
        class_counts=np.bincount(z["gt_relations"][:,2],minlength=51)[1:]
        class_n+=(class_counts>0)
        native_top=None
        for name,(lab,conf) in variants.items():
            rows,order=matches(z,lab,conf)
            if name=="native":
                native_top=set(order[:50].tolist())
            top50_overlap[name].append(len(native_top&set(order[:50].tolist()))/max(len(native_top),1))
            for ki,(rec,hit,count) in enumerate(rows):
                recalls[name][index,ki]=rec
                class_sums[name][ki]+=np.divide(hit,count,out=np.zeros(50,dtype=float),where=count>0)
            if args.task=="sgcls":
                if len(lab)!=len(z["gt_labels"]):
                    raise RuntimeError("SGCls native proposals are not GT-aligned")
                correct=(lab==z["gt_labels"])
                record=calibration[name]
                record["correct"]+=int(correct.sum())
                record["count"]+=len(lab)
                for bi in range(15):
                    sel=np.minimum((conf*15).astype(int),14)==bi
                    record["bins"][bi]+=np.array([sel.sum(),conf[sel].sum(),correct[sel].sum()])
        if (index+1)%100==0:
            row=dict(images=index+1,total=n,task=args.task)
            print(json.dumps(row),flush=True)
            atomic_json(out/"progress.json",row)
    summary={}
    rng=np.random.default_rng(17)
    for name in names:
        values=recalls[name]
        delta=values[:,KS.index(50)]-recalls["native"][:,KS.index(50)]
        boots=[float(delta[rng.integers(0,n,n)].mean()) for _ in range(2000)]
        record=calibration[name]
        cal=None
        if record["count"]:
            bins=record["bins"]
            valid=bins[:,0]>0
            ece=float(np.abs(bins[valid,1]-bins[valid,2]).sum()/record["count"])
            cal=dict(top1=record["correct"]/record["count"],ece_15=ece,objects=record["count"],
                     score_label_crossing=name.endswith("labels_only") or name.endswith("scores_only"))
        summary[name]=dict(R={str(k):float(values[:,i].mean()) for i,k in enumerate(KS)},
            image_macro_mR={str(k):float(np.divide(class_sums[name][i],class_n,out=np.zeros(50),where=class_n>0).mean()) for i,k in enumerate(KS)},
            delta_R50=float(delta.mean()),paired_image_bootstrap_95ci=np.quantile(boots,[.025,.975]).tolist(),
            mean_top50_retained_fraction=float(np.mean(top50_overlap[name])),object_calibration=cal)
    np.savez_compressed(out/"image_recalls.npz",image_ids=np.asarray(test_ids),**{name.replace("/","__"):v for name,v in recalls.items()})
    atomic_json(out/"summary.json",dict(status="complete",protocol=protocol,images=n,conditions=summary))
    print("[COMPLETE] R3",args.task,flush=True)


if __name__=="__main__":
    main()
