"""V-U fitting, prediction-only allocation and audited native SGDet evaluation."""
import argparse
import fcntl
import time
import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT,atomic_json,ensure_storage,output_path,sha256
from evidence_completion import read,lock
from repro_experiment import save_torch
from vp_experiment import rows_for,progress,finish
from vp_math import split_fold
from vq_experiment import load_bundle as parent_bundle,image_inputs,assert_baseline
from vu_protocol import register,load,OUT,CACHE,WEIGHTS
from vu_math import (ARMS,PRIMARY,EXPERTS,view_features,fit_expert,fit_risk,
                     risk_features,allocation,apply_expert)


def prepare(protocol,reg,lane):
    start=time.monotonic(); stage=OUT/lane/"prepare"; path=CACHE/lane/"data.pt"
    if (stage/"summary.json").exists():
        info=read(stage/"summary.json")
        if info["protocol_sha256"]!=reg or info["bundle_sha256"]!=sha256(path): raise RuntimeError("Pack drift")
        return
    bundle,parent_sha=parent_bundle(protocol["vq_protocol_sha256"],lane)
    manifest={}
    for shard in [0,1]:
        value=read(OUT/lane/("extract%d"%shard)/"summary.json")
        if value["protocol_sha256"]!=reg or value["status"]!="complete": raise RuntimeError("Extraction incomplete")
        for item in value["manifest"]:
            if item["image_id"] in manifest: raise RuntimeError("Duplicate extraction ID")
            manifest[item["image_id"]]=item
    if set(manifest)!=set(protocol["image_ids"]) or bundle["image_ids"]!=protocol["image_ids"]:
        raise RuntimeError("Feature/parent image coverage differs")
    tight,context,risks=[],[],[]
    for i,entry in enumerate(protocol["inputs"]):
        iid=entry["image_id"]; p=CACHE/lane/"features"/(iid+".pt")
        if sha256(p)!=manifest[iid]["feature_sha256"] or sha256(entry["raw"])!=entry["raw_sha256"]:
            raise RuntimeError("Feature/raw checksum changed")
        feature=torch.load(str(p),map_location="cpu"); raw=torch.load(entry["raw"],map_location="cpu")
        a,b=bundle["offsets"][i:i+2]
        if (feature["protocol_sha256"]!=reg or feature["raw_sha256"]!=entry["raw_sha256"]
            or not torch.equal(raw["native_logits"].float(),bundle["native_logits"][a:b])
            or not torch.equal(raw["crop_boxes"],feature["crop_boxes"])):
            raise RuntimeError("Cross-runtime feature/box ordering mismatch")
        for key,buffer in [("siglip_tight",tight),("siglip_context",context)]:
            value=feature[key].float()
            if value.shape!=(b-a,768) or not torch.isfinite(value).all() or not torch.allclose(value.norm(dim=-1),torch.ones(b-a),atol=2e-5):
                raise RuntimeError("Invalid normalized visual feature")
            buffer.append(value)
        risks.append(risk_features(raw["native_logits"].float(),raw["crop_boxes"].float(),raw["image_size"]))
        if (i+1)%100==0 or i+1==len(protocol["inputs"]):progress(stage,"verify_pack",i+1,len(protocol["inputs"]),start)
    bundle.update(protocol_sha256=reg,parent_bundle_sha256=parent_sha,siglip_tight=torch.cat(tight),
                  siglip_context=torch.cat(context),risk_features=torch.cat(risks))
    save_torch(path,bundle)
    finish(stage,dict(status="complete",images=len(protocol["image_ids"]),proposals=len(bundle["labels"]),
        protocol_sha256=reg,bundle_sha256=sha256(path),parent_bundle_sha256=parent_sha,
        feature_manifest=manifest,elapsed_seconds=time.monotonic()-start))


def bundle_for(reg,lane):
    info=read(OUT/lane/"prepare/summary.json"); path=CACHE/lane/"data.pt"
    if info["protocol_sha256"]!=reg or sha256(path)!=info["bundle_sha256"]:raise RuntimeError("Bundle mismatch")
    value=torch.load(str(path),map_location="cpu")
    if value["protocol_sha256"]!=reg:raise RuntimeError("Unregistered bundle")
    return value,info["bundle_sha256"]


def train_fold(protocol,reg,lane,fold,bundle,digest,split):
    stage=OUT/lane/("fold%d"%fold); path=WEIGHTS/lane/("fold%d.pth"%fold)
    summary=stage/"training_summary.json"
    if summary.exists():
        info=read(summary)
        if info["protocol_sha256"]!=reg or info["bundle_sha256"]!=digest or info["checkpoint_sha256"]!=sha256(path):
            raise RuntimeError("Trained checkpoint mismatch")
        return torch.load(str(path),map_location="cpu"),info
    fit_ix=rows_for(bundle,split["fit"]); val_ix=rows_for(bundle,split["inner_validation"])
    train_ix=rows_for(bundle,split["training"]); labels=bundle["labels"]
    states={}; histories={}; epochs={}
    for name in EXPERTS:
        x=view_features(bundle,name); start=time.monotonic()
        _,selection=fit_expert(x[fit_ix],labels[fit_ix],validation=(x[val_ix],labels[val_ix]),smoke=protocol["smoke"],
            callback=lambda row,total:progress(stage,"inner_"+name,row["epoch"],total,start,detail=row))
        epochs[name]=selection["selected_epoch"]
        if not epochs[name]: raise RuntimeError("No expert epoch selected")
        start=time.monotonic()
        states[name],history=fit_expert(x[train_ix],labels[train_ix],epochs=epochs[name],
            callback=lambda row,total:progress(stage,"refit_"+name,row["epoch"],total,start,detail=row))
        histories[name]=dict(selection=selection,refit=history)
        del x
    progress(stage,"fit_identity_risk",0,20,time.monotonic())
    risk=fit_risk(bundle["risk_features"][train_ix],bundle["native_logits"][train_ix],labels[train_ix],smoke=protocol["smoke"])
    payload=dict(protocol_sha256=reg,bundle_sha256=digest,states=states,risk=risk,epochs=epochs,
                 fit_ids=split["training"],held_ids=split["held"],fold=fold)
    save_torch(path,payload)
    info=dict(protocol_sha256=reg,bundle_sha256=digest,checkpoint_sha256=sha256(path),
              epochs=epochs,histories=histories,risk_history=risk["history"],risk_positive_events=risk["positive_events"],
              fit_images=len(split["training"]),held_images=len(split["held"]))
    atomic_json(summary,info);return payload,info


def evaluate(protocol,reg,lane,fold,bundle,payload,info,split):
    from vn_train_proposals import make_post,replay
    from vb_native import OfficialMetrics
    from vc_protocol import KS,summarize_rows
    stage=OUT/lane/("fold%d"%fold); start=time.monotonic()
    _,post=make_post(stage/"runtime"); metric=OfficialMetrics("sgdet")
    records=image_inputs(protocol,bundle,split["held"])
    mapping={iid:i for i,iid in enumerate(bundle["image_ids"])}; arms={k:[] for k in ARMS}; budget_totals={k:0 for k in ARMS}
    with torch.no_grad():
        for index,rec in enumerate(records):
            iid=rec["iid"]; raw=rec["raw"]; j=mapping[iid]; a,b=bundle["offsets"][j:j+2]
            native=raw["native_logits"].float().cuda(); features={k:view_features({
                name:bundle[name][a:b] for name in ["features","siglip_tight","siglip_context"]},k).cuda() for k in EXPERTS}
            risk_x=risk_features(native,raw["crop_boxes"].float().cuda(),raw["image_size"])
            if not torch.allclose(risk_x.cpu(),bundle["risk_features"][a:b],atol=2e-5,rtol=2e-5):
                raise RuntimeError("Risk input ordering mismatch")
            per={}; selected_counts={}; expert_diagnostics={}
            for name in EXPERTS:
                state=payload["states"][name]
                pred=F.linear(features[name],state["weight"].cuda(),state["bias"].cuda()).argmax(-1).cpu().numpy()+1
                original=native[:,1:].argmax(-1).cpu().numpy()+1; target=rec["target"]; pos=target>0
                expert_diagnostics[name]=dict(positive=int(pos.sum()),correct=int(((pred==target)&pos).sum()),
                    repairs=int(((pred==target)&(original!=target)&pos).sum()),
                    damages=int(((pred!=target)&(original==target)&pos).sum()))
            for arm in ARMS:
                if arm=="native":selected=torch.zeros(b-a,dtype=torch.bool,device="cuda");logits=native
                else:
                    if arm.endswith("_all"):
                        name={"dino_all":"dino","siglip_tight_all":"siglip_tight","siglip_dual_all":"siglip_dual"}[arm]
                        selected=torch.ones(b-a,dtype=torch.bool,device="cuda")
                    else:
                        name="siglip_dual"
                        selected=allocation(arm,native,risk_x,raw["pairs"].long().cuda(),raw["relation_logits"].float().cuda(),iid,payload["risk"])
                    logits=apply_expert(native,features[name],payload["states"][name],selected)
                    if not torch.equal(logits[~selected],native[~selected]):raise RuntimeError("Unselected predictions changed")
                    if not torch.allclose(logits.softmax(-1)[:,0],native.softmax(-1)[:,0],atol=2e-6,rtol=2e-5):raise RuntimeError("Background mass changed")
                prediction=replay(post,raw,logits)
                row=metric.row(iid,prediction,rec["gt"],dict(logits=logits),rec["target"])
                if arm=="native":assert_baseline(row,rec["baseline"])
                arms[arm].append(row);per[arm]=row;selected_counts[arm]=int(selected.sum());budget_totals[arm]+=int(selected.sum())
                for k in KS:metric.result["sgdet_recall"][k].clear()
            if len({selected_counts[k] for k in [PRIMARY,"uncertainty_budget","random_budget"]})!=1:
                raise RuntimeError("Unequal allocation budget")
            atomic_json(stage/"images"/(iid+".json"),dict(protocol_sha256=reg,checkpoint_sha256=info["checkpoint_sha256"],
                image_id=iid,fold=fold,metrics=per,selected_counts=selected_counts,expert_diagnostics=expert_diagnostics))
            if (index+1)%50==0 or index+1==len(records):progress(stage,"native_SGDet_all_policies",index+1,len(records),start)
    finish(stage,dict(status="complete",protocol_sha256=reg,checkpoint_sha256=info["checkpoint_sha256"],fold=fold,
        images=len(records),primary=PRIMARY,epochs=payload["epochs"],arms={k:summarize_rows(v) for k,v in arms.items()},
        selected_counts=budget_totals,validation_accessed=False,test_accessed=False,formal_gate_accepted=False,
        independent_confirmation=False,elapsed_seconds=time.monotonic()-start))


def summarize(protocol,reg,lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]:raise ValueError("No smoke efficacy summary")
    rows=[]; choices=[];start=time.monotonic()
    for fold in range(5):
        stage=OUT/lane/("fold%d"%fold);done=read(stage/"summary.json")
        digest=sha256(WEIGHTS/lane/("fold%d.pth"%fold))
        if done["protocol_sha256"]!=reg or done["checkpoint_sha256"]!=digest:raise RuntimeError("Fold provenance drift")
        choices.append(dict(fold=fold,epochs=done["epochs"]))
        for iid in read(stage/"split.json")["held"]:
            row=read(stage/"images"/(iid+".json"))
            if row["protocol_sha256"]!=reg or row["checkpoint_sha256"]!=digest or row["fold"]!=fold:raise RuntimeError("Row provenance drift")
            rows.append(row)
    by_id={r["image_id"]:r for r in rows}
    if len(rows)!=3000 or set(by_id)!=set(protocol["image_ids"]):raise RuntimeError("Incomplete cross-fitting")
    rows=[by_id[i] for i in protocol["image_ids"]]; folds=np.array([r["fold"] for r in rows]); comparisons={}
    for reference,arm in [("native",a) for a in ARMS[1:]]+[(a,PRIMARY) for a in ["uncertainty_budget","random_budget"]]+[("siglip_tight_all","siglip_dual_all")]:
        base=[r["metrics"][reference] for r in rows];trial=[r["metrics"][arm] for r in rows]
        pos=np.array([r["positive_objects"] for r in base]);support=np.array([r["class_recalls"][4] for r in base],dtype=float)
        other=np.array([r["class_recalls"][4] for r in trial],dtype=float)
        if pos.tolist()!=[r["positive_objects"] for r in trial] or not np.array_equal(np.isfinite(support),np.isfinite(other)):
            raise RuntimeError("Comparison denominator changed")
        gain=np.array([u["post_nms_correct"]-b["post_nms_correct"] for b,u in zip(base,trial)])
        dr=np.array([u["recalls"][4]-b["recalls"][4] for b,u in zip(base,trial)])
        stats=policy_summary(np.arange(3000),np.arange(3000),gain,dr,np.nan_to_num(other-support),np.isfinite(support),pos,folds)
        stats["evaluated_images"]=stats.pop("selected_images");comparisons[arm+"_versus_"+reference]=stats
    expert_stats={k:{metric:sum(r["expert_diagnostics"][k][metric] for r in rows) for metric in ["positive","correct","repairs","damages"]} for k in EXPERTS}
    checks=comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(OUT/lane,dict(status="complete",images=3000,protocol_sha256=reg,primary=PRIMARY,choices=choices,
        arms={k:summarize_rows([r["metrics"][k] for r in rows]) for k in ARMS},comparisons=comparisons,
        expert_diagnostics=expert_stats,selected_counts={k:sum(r["selected_counts"][k] for r in rows) for k in ARMS},
        primary_training_screen_satisfied=all(checks.values()),formal_gate_accepted=False,
        independent_confirmation=False,native_sgg_trained_on_these_images=True,reused_train_images=True,
        validation_accessed=False,test_accessed=False,offline_extraction_not_measured_speedup=True,
        next_step="Stop. No control promotion, extra seeds, confirmation/test or new tuning automatically.",elapsed_seconds=time.monotonic()-start))


def main():
    from vk_gate_native import configure_backend
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["register","prepare","fold","summarize"],required=True)
    parser.add_argument("--smoke",action="store_true");parser.add_argument("--fold",type=int,choices=range(5))
    args=parser.parse_args();ensure_storage();torch.set_num_threads(2);configure_backend()
    if args.stage=="register":
        _,reg,lane=register(args.smoke);print("[registered]",lane,reg,flush=True);return
    protocol,reg,lane=load(args.smoke)
    name="fold%d"%args.fold if args.fold is not None else args.stage
    guard=output_path(OUT/lane/(name+".lock")).open("w");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=="prepare":prepare(protocol,reg,lane)
    elif args.stage=="fold":
        if args.fold is None or (args.smoke and args.fold!=0):raise ValueError("Invalid fold")
        stage=OUT/lane/("fold%d"%args.fold)
        if (stage/"summary.json").exists():
            value=read(stage/"summary.json")
            if value["protocol_sha256"]!=reg or value["checkpoint_sha256"]!=sha256(WEIGHTS/lane/("fold%d.pth"%args.fold)):
                raise RuntimeError("Completed fold changed")
            return
        bundle,digest=bundle_for(reg,lane);split=split_fold(protocol["image_ids"],protocol["folds"],args.fold)
        lock(stage/"split.json",dict(protocol_sha256=reg,**split))
        payload,info=train_fold(protocol,reg,lane,args.fold,bundle,digest,split)
        evaluate(protocol,reg,lane,args.fold,bundle,payload,info,split)
    else:summarize(protocol,reg,lane)


if __name__=="__main__":main()
