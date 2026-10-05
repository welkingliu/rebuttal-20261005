"""One foreground-eligibility routing pilot after the measured V-L background failure."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources
from repro_experiment import config, dataset
from ve_experiment import lookup
from vb_native import OfficialMetrics, targets
from vc_protocol import summarize_rows, paired_gate, KS
from vk_gate_native import configure_backend
from vk_gate_protocol import CACHE, OUT as VK
from r19_vk_decomposition import cached_prediction_check
from vl_sgdet_feasibility import RAW, all_candidates, emit_progress
from vk_gate_protocol import CHECKPOINT
from vl_router import predict_logistic, select_action


OUT=ROOT / "results/VM_localization_router"
VL=ROOT / "results/VL_proposal_router"


def localization_features(features, baseline, boxes, size):
    """No target, GT boxes, or labels may enter the localization estimator."""
    p=baseline.softmax(-1);fg=p[:,1:]; top=fg.topk(2,dim=-1).values
    wh=(boxes[:,2:]-boxes[:,:2]+1).clamp_min(1)
    center=(boxes[:,2:]+boxes[:,:2])/2
    extra=torch.stack([p[:,0],top[:,0],top[:,0]-top[:,1],
        -(p*p.clamp_min(1e-12).log()).sum(-1)/np.log(151),
        wh.prod(-1)/(size[0]*size[1]),(wh[:,0]/wh[:,1]).log().clamp(-5,5),
        center[:,0]/size[0],center[:,1]/size[1]],-1)
    x=torch.cat([features,extra],-1).numpy().astype(np.float64)
    if x.shape!=(len(features),776) or not np.isfinite(x).all():
        raise ValueError("Localization feature contract failed")
    return x


def fit_localization(x,y):
    from sklearn.linear_model import LogisticRegression
    mean=x.mean(0); scale=np.maximum(x.std(0),.001)
    estimator=LogisticRegression(C=.01,solver="lbfgs",max_iter=2000,random_state=17)
    estimator.fit((x-mean)/scale,y)
    if int(estimator.n_iter_.max())>=2000: raise RuntimeError("Localization fit did not converge")
    return dict(mean=mean.tolist(),scale=scale.tolist(),weight=estimator.coef_[0].tolist(),
                bias=float(estimator.intercept_[0]),iterations=int(estimator.n_iter_[0]))


def choose_localized(model, x, candidate_indices, frozen_choices):
    pm=predict_logistic(model,x)
    rows=[]
    for index,r in zip(candidate_indices,frozen_choices):
        if index!=r["proposal"]: raise ValueError("Candidate order changed")
        value=dict(r,localization_probability=float(pm[index]),
                   localization_weighted_priority=float(pm[index]*r["expected_signed_utility"]))
        rows.append(value)
    # The 0.5 condition is a fixed binary decision, not a searched acceptance threshold.
    eligible=[r for r in rows if r["localization_probability"]>=.5 and r["safety_probability"]>=.6 and r["expected_signed_utility"]>0]
    chosen=min(eligible,key=lambda r:(-r["localization_weighted_priority"],r["proposal"]))["proposal"] if eligible else None
    return chosen,rows,pm


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["fit","evaluate"],required=True);a=p.parse_args()
    ensure_storage();torch.set_num_threads(2);configure_backend()
    guard=output_path(OUT / "execution.lock").open("w");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=read(VL / "protocol.json")
    if sha256(CHECKPOINT)!=prior["expert_checkpoint_sha256"]:
        raise RuntimeError("Frozen visual expert changed")
    protocol=dict(version="vm_localization_eligibility_v1",train_ids=prior["train_ids"],evaluation_ids=prior["evaluation_ids"],
        hypothesis="V-L often changes unmatched background proposals; learn annotation-localization eligibility to direct its unchanged repair selector toward matched proposals",
        prior_protocol_sha256=sha256(VL / "protocol.json"),frozen_selector_sha256=sha256(VL / "selector.json"),
        data_role="Consumed validation development: same200 fit, same800 exploration; not independent confirmation",
        features="DINO768 plus 8 native confidence/geometric features; no GT at prediction",
        labels="Matched positive IoU>=0.5 versus background IoU<0.3 on fixed native proposal boxes; ignore ambiguous band",
        caveat="Annotation-localization eligibility, not proof of true background in incompletely annotated VG",
        model="one unweighted binary logistic head",C=.01,max_iterations=2000,standardization="fit200 only",
        policy="frozen V-L selector; eligibility probability>=0.5; original conditional safety>=0.6; rank by eligibility*original priority; at most1 action/image",
        alpha=.5,no_hyperparameter_search=True,no_test=True,no_extra_seeds=True,
        scientific_gate_pass=False,independent_confirmation=False,
        sources=sources(["vm_localization_router.py","vl_router.py","vl_sgdet_feasibility.py","vb_native.py","vc_protocol.py"]))
    reg=lock(OUT / "protocol.json",protocol);out=OUT/a.stage
    if (out/"summary.json").exists():
        if read(out/"summary.json")["protocol_sha256"]!=reg: raise RuntimeError("Resume drift")
        print("Already complete: "+str(out),flush=True);return
    start=time.monotonic()
    if a.stage=="fit":
        xx=[];yy=[];image_rows=[]
        for n,iid in enumerate(protocol["train_ids"],1):
            raw=torch.load(str(RAW/(iid+".pt")),map_location="cpu")
            fpath=CACHE/"sgdet/gate/features"/(iid+".pt")
            f=torch.load(str(fpath),map_location="cpu")
            expected=read(VL/"fit/images"/(iid+".json"))["input_sha256"]
            if sha256(RAW/(iid+".pt"))!=expected["raw"] or sha256(fpath)!=expected["feature"]: raise RuntimeError("Fit input changed")
            x=localization_features(f["features"],raw["native_logits"],raw["proposal_boxes"],raw["size"])
            y=raw["target"].numpy();keep=y>=0;xx.append(x[keep]);yy.append((y[keep]>0).astype(int))
            image_rows.append(dict(image_id=iid,positive=int((y>0).sum()),background=int((y==0).sum()),ignored=int((y<0).sum())))
            if n%20==0:emit_progress(out,start,n,200,0,"localization_fit_inputs")
        x=np.concatenate(xx);y=np.concatenate(yy);model=fit_localization(x,y);model["protocol_sha256"]=reg
        lock(OUT/"localization_head.json",model)
        summary=dict(status="complete",protocol_sha256=reg,images=200,training_proposals=len(y),positive=int(y.sum()),
                     per_image=image_rows,checkpoint_sha256=sha256(OUT/"localization_head.json"),elapsed_seconds=time.monotonic()-start)
    else:
        from sklearn.metrics import roc_auc_score,average_precision_score
        cfg=config("sgdet",out)
        from pysgg.modeling.roi_heads.relation_head.inference import make_roi_relation_post_processor
        from pysgg.structures.bounding_box import BoxList
        post=make_roi_relation_post_processor(cfg)
        if post.use_gt_box or post.use_relness_ranking or post.BCE_loss or post.attribute_on:raise RuntimeError("Postprocessor mismatch")
        ds=dataset(cfg,"val");mapping=lookup(ds);metric=OfficialMetrics("sgdet")
        model=read(OUT/"localization_head.json");selector=read(VL/"selector.json")
        if model["protocol_sha256"]!=reg: raise RuntimeError("Head provenance failed")
        state=torch.load(str(CHECKPOINT),map_location="cpu")
        records=[];all_y=[];all_pm=[]
        for n,iid in enumerate(protocol["evaluation_ids"],1):
            path=out/"images"/(iid+".json")
            if path.exists():
                r=read(path)
                if r["protocol_sha256"]!=reg: raise RuntimeError("Record provenance mismatch")
                records.append(r);all_y.extend(r["audit_match_targets"]);all_pm.extend(r["audit_match_probabilities"]);continue
            rpath=RAW/(iid+".pt");fpath=CACHE/"sgdet/gate/features"/(iid+".pt")
            raw=torch.load(str(rpath),map_location="cpu");f=torch.load(str(fpath),map_location="cpu")
            prior_row=read(VL/"evaluate/images"/(iid+".json"))
            if sha256(rpath)!=prior_row["input_sha256"]["raw"] or sha256(fpath)!=prior_row["input_sha256"]["feature"]:raise RuntimeError("Evaluation inputs changed")
            x=localization_features(f["features"],raw["native_logits"],raw["proposal_boxes"],raw["size"])
            cached=torch.load(str(VL/"evaluate/features"/(iid+".pt")),map_location="cpu")
            _,frozen_choices=select_action(selector,cached["x"].numpy(),cached["indices"])
            chosen,choices,pm=choose_localized(model,x,cached["indices"],frozen_choices)
            with torch.no_grad():
                candidate,eligible,_,_=all_candidates(f["features"],raw["native_logits"],state)
                if eligible.nonzero(as_tuple=False).flatten().tolist()!=cached["indices"]:raise RuntimeError("Eligibility changed")
                base=raw["native_logits"].cuda();updated=base.clone()
                if chosen is not None:updated[chosen]=candidate[chosen].cuda()
                def replay(logits):
                    box=BoxList(raw["proposal_boxes"].cuda(),raw["size"],"xyxy");box.add_field("boxes_per_cls",raw["boxes_per_cls"].cuda())
                    return post.forward(([raw["relation_logits"].cuda()],[logits]),[raw["pairs"].cuda()],[box])[0]
                native_pred=replay(base);prediction=replay(updated) if chosen is not None else native_pred
                cached_prediction_check(native_pred,VK/"sgdet/gate/native/predictions"/(iid+".npz"))
                # Only after committing the GT-free action, attach evaluation labels.
                gt=ds.get_groundtruth(mapping[iid],evaluation=True);y=targets(raw,gt,"sgdet")
                if not np.array_equal(y,raw["target"].numpy()):raise RuntimeError("Target matching changed")
                b=metric.row(iid,native_pred,gt,dict(logits=base),y);u=metric.row(iid,prediction,gt,dict(logits=updated),y)
                keep=y>=0;all_y.extend((y[keep]>0).astype(int).tolist());all_pm.extend(pm[keep].tolist())
                r=dict(image_id=iid,protocol_sha256=reg,chosen=chosen,choices=choices,baseline=b,updated=u,
                    chosen_target_for_audit=int(y[chosen]) if chosen is not None else None,
                    audit_match_targets=(y[keep]>0).astype(int).tolist(),audit_match_probabilities=pm[keep].tolist())
                atomic_json(path,r);records.append(r)
                for k in KS:metric.result["sgdet_recall"][k].clear()
            if n%20==0 or n==800:emit_progress(out,start,n,800,n,"localization_routing_evaluation")
        b=[r["baseline"] for r in records];u=[r["updated"] for r in records]
        summary=dict(status="complete",protocol_sha256=reg,images=800,baseline=summarize_rows(b),updated=summarize_rows(u),
            exploratory_numeric_criteria=paired_gate(b,u),selected_images=sum(r["chosen"] is not None for r in records),
            chosen_positive=sum(r["chosen_target_for_audit"] is not None and r["chosen_target_for_audit"]>0 for r in records),
            chosen_background=sum(r["chosen_target_for_audit"]==0 for r in records),
            chosen_ambiguous=sum(r["chosen_target_for_audit"] is not None and r["chosen_target_for_audit"]<0 for r in records),
            localization_auc=float(roc_auc_score(all_y,all_pm)),localization_average_precision=float(average_precision_score(all_y,all_pm)),
            formal_gate_accepted=False,independent_confirmation=False,test_evaluated=False,
            elapsed_seconds=time.monotonic()-start)
    atomic_json(out/"summary.json",summary);atomic_json(out/"progress.json",dict(status="complete",images=summary["images"],total=summary["images"]))
    print(json.dumps({k:v for k,v in summary.items() if k!="per_image"}),flush=True)


if __name__=="__main__":main()
