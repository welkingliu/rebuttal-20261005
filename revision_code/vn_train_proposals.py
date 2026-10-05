"""Fixed training-only proposal routing pipeline; consumed validation is exploratory."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources, selected_ids
from repro_experiment import config, dataset, save_torch
from native_runtime import infer
from vb_native import OfficialMetrics, targets, compare_predictions
from vc_protocol import summarize_rows, paired_gate, KS
from vk_gate_protocol import CHECKPOINT, PRIMARY_SHA, MANIFEST as VK_MANIFEST, CACHE as VK_CACHE, OUT as VK_OUT, ids as vk_ids
from vl_router import action_features, fit_selector, select_action, FEATURE_NAMES
from vl_policy import improves_identity, preserves_relations


OUT=ROOT/"results/VN_train_proposal_router"
CACHE=ROOT/"cache/VN_train_proposal_router"
VALID_RAW=ROOT/"cache/R19_vk_mechanism/sgdet"
SOURCE_NAMES=["vn_train_proposals.py","vl_router.py","vl_policy.py","vm_localization_router.py",
    "vl_sgdet_feasibility.py","independent_identity_expert.py","vi_native.py","ve_experiment.py",
    "repro_experiment.py","vb_native.py","vc_protocol.py","vk_gate_native.py"]


def locations(smoke):
    lane="smoke" if smoke else "train3000"
    return OUT/lane,CACHE/lane


def progress(out,start,n,total,stage):
    import os
    elapsed=time.monotonic()-start
    value=dict(status="running",stage=stage,pid=os.getpid(),images=n,total=total,seconds=elapsed,
        eta_seconds=elapsed/n*(total-n) if n else None)
    atomic_json(out/"progress.json",value);print(json.dumps(value),flush=True)
    if elapsed>4*3600:raise TimeoutError("V-N four-hour per-stage bound exceeded")


def register(out,smoke,ds):
    from ve_experiment import lookup
    from independent_identity_expert import WEIGHT
    split_path=ROOT/"results/VI_selective_fusion/splits.json"
    split=read(split_path);excluded=set(split["train"])
    if len(excluded)!=5000:raise RuntimeError("Cannot identify all visual expert training images")
    mapping=lookup(ds)
    train=selected_ids(set(mapping)-excluded,3000,"vn_native_train_proposals_v1")
    if smoke:train=train[:3]
    if set(train)&set(vk_ids()) or set(train)&excluded:raise RuntimeError("Training/evaluation/expert overlap")
    if len(train)!=(3 if smoke else 3000):raise RuntimeError("Insufficient training images")
    protocol=dict(version="vn_training_proposals_v1",smoke=smoke,training_ids=train,evaluation_ids=vk_ids(),
        data_role="Router/localization heads fit native VG train only, excluding visual-expert training images. Validation is previously consumed, exploratory, NOT independent confirmation.",
        training_caveat="Native frozen SGG was trained on VG train; only the auxiliary visual expert is out-of-fit for these images",
        excluded_expert_images=sorted(excluded),expert_split_sha256=sha256(split_path),expert_sha256=PRIMARY_SHA,
        dino_weight_sha256=sha256(WEIGHT),native_checkpoint_sha256=read(VK_MANIFEST)["baseline_assets"]["sgdet"]["sha256"],
        candidate_alpha=.5,prototype_min_objects=5,selector_C=.1,localization_C=.01,maximum_iterations=2000,
        selector="Same V-L two-stage utility model on actual post-NMS single-update outcomes",
        localization="Same V-M DINO+geometry head, IoU>=0.5 positive, <0.3 background, intermediate ignored",
        policy="Same V-M policy: localization>=0.5, conditional safety>=0.6, at most one action/image; frozen expert and native relation/NMS/ranking",
        new_model_families=0,hyperparameter_search=False,final_test_access=False,scientific_gate_pass=False,
        stop="Missing class/outcome support, parity failure or stage deadline; one fit and one consumed-validation evaluation, no automatic seeds or final test",
        sources=sources(SOURCE_NAMES))
    return protocol,lock(out/"protocol.json",protocol)


def verify(out):
    protocol=read(out/"protocol.json")
    if protocol["sources"]!=sources(SOURCE_NAMES) or sha256(CHECKPOINT)!=PRIMARY_SHA:
        raise RuntimeError("Frozen code/expert drift")
    return protocol,sha256(out/"protocol.json")


def checked_load(path,registration):
    row=torch.load(str(path),map_location="cpu")
    if row["protocol_sha256"]!=registration:raise RuntimeError("Cache registration mismatch")
    return row


def make_post(out):
    from pysgg.modeling.roi_heads.relation_head.inference import make_roi_relation_post_processor
    cfg=config("sgdet",out)
    post=make_roi_relation_post_processor(cfg)
    if post.use_gt_box or post.attribute_on or post.BCE_loss or post.use_relness_ranking:
        raise RuntimeError("Postprocessor contract drift")
    return cfg,post


def replay(post,raw,logits):
    from pysgg.structures.bounding_box import BoxList
    box=BoxList(raw["proposal_boxes"].cuda(),raw["size"],"xyxy")
    box.add_field("boxes_per_cls",raw["boxes_per_cls"].cuda())
    return post.forward(([raw["relation_logits"].cuda()],[logits]),[raw["pairs"].cuda()],[box])[0]


def ground_truth(raw):
    from pysgg.structures.bounding_box import BoxList
    gt=BoxList(raw["gt_boxes"],raw["image_size"],"xyxy")
    gt.add_field("labels",raw["gt_labels"]);gt.add_field("relation_tuple",raw["gt_relations"])
    return gt


def export(out,cache,smoke):
    from ve_experiment import lookup
    from vi_native import setup
    from vk_gate_native import crop_geometry
    if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError("At least 12 GiB free required before export")
    stage=out/"export";start=time.monotonic()
    model,cfg,transform,provenance,patch=setup("sgdet",stage)
    ds=dataset(cfg,"train");mapping=lookup(ds)
    protocol,reg=register(out,smoke,ds)
    if provenance["checkpoint_sha256"]!=protocol["native_checkpoint_sha256"]:raise RuntimeError("Wrong native checkpoint")
    if (stage/"summary.json").exists():
        if read(stage/"summary.json")["protocol_sha256"]!=reg:raise RuntimeError("Result drift")
        patch.close();return
    post=model.roi_heads.relation.post_processor
    rows=[]
    progress(stage,start,0,len(protocol["training_ids"]),"native_training_proposals")
    for n,iid in enumerate(protocol["training_ids"],1):
        path=cache/"raw"/(iid+".pt");meta=stage/"images"/(iid+".json")
        if path.exists() and meta.exists():
            record=read(meta)
            if record["protocol_sha256"]!=reg or record["raw_sha256"]!=sha256(path):raise RuntimeError("Resume drift")
            rows.append(record)
        else:
            patch.enabled=False
            with torch.no_grad():
                prediction,_=infer(model,cfg,transform,ds,mapping[iid]);c=patch.capture
                if not c.get("route_checked"):raise RuntimeError("Native routing contract unverified")
                gt=ds.get_groundtruth(mapping[iid],evaluation=True);y=targets(c,gt,"sgdet")
                raw=dict(image_id=iid,protocol_sha256=reg,native_logits=c["baseline"].cpu(),
                    proposal_boxes=c["proposal_boxes"].cpu(),size=c["size"],boxes_per_cls=c["boxes_per_cls"].cpu(),
                    pairs=c["pairs"].cpu(),relation_logits=c["relation_logits"].cpu(),target=torch.from_numpy(y),
                    image=str(Path(ds.filenames[mapping[iid]]).resolve()),image_size=gt.size,
                    crop_boxes=crop_geometry(c,gt,"sgdet"),gt_boxes=gt.bbox.cpu(),
                    gt_labels=gt.get_field("labels").cpu(),gt_relations=gt.get_field("relation_tuple").cpu())
                replayed=replay(post,raw,c["baseline"])
                error=compare_predictions(prediction,replayed)
                save_torch(path,raw)
                record=dict(image_id=iid,protocol_sha256=reg,raw_sha256=sha256(path),
                    proposals=len(y),positive=int((y>0).sum()),background=int((y==0).sum()),
                    baseline_replay_max_error=error,positive_labels=np.unique(y[y>0]).tolist())
                atomic_json(meta,record);rows.append(record)
        if n%10==0 or n==len(protocol["training_ids"]):progress(stage,start,n,len(protocol["training_ids"]),"native_training_proposals")
    patch.close()
    summary=dict(status="complete",protocol_sha256=reg,images=len(rows),proposals=sum(r["proposals"] for r in rows),
        positives=sum(r["positive"] for r in rows),classes=sorted(set(v for r in rows for v in r["positive_labels"])),
        baseline_replay_max_error=max(r["baseline_replay_max_error"] for r in rows),elapsed_seconds=time.monotonic()-start)
    finish(stage,summary)


def features(out,cache):
    from PIL import Image
    from independent_identity_expert import REPO,WEIGHT,crop_tensor
    protocol,reg=verify(out);require_stage(out,"export",reg);stage=out/"features";start=time.monotonic()
    if (stage/"summary.json").exists():return
    if sha256(WEIGHT)!=protocol["dino_weight_sha256"]:raise RuntimeError("DINO weights changed")
    encoder=torch.hub.load(str(REPO),"dinov2_vitb14",source="local",pretrained=False)
    encoder.load_state_dict(torch.load(str(WEIGHT),map_location="cpu",weights_only=True),strict=True)
    encoder.cuda().eval().requires_grad_(False)
    # Match the V-K visual feature backend, including its separate runtime.
    torch.manual_seed(17);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    progress(stage,start,0,len(protocol["training_ids"]),"detected_box_dino")
    for n,iid in enumerate(protocol["training_ids"],1):
        path=cache/"features"/(iid+".pt")
        rawpath=cache/"raw"/(iid+".pt")
        if path.exists():
            saved=checked_load(path,reg)
            if saved["raw_sha256"]!=sha256(rawpath):raise RuntimeError("Feature source changed")
        else:
            raw=checked_load(rawpath,reg)
            with Image.open(raw["image"]) as source:image=source.convert("RGB")
            if tuple(image.size)!=tuple(raw["image_size"]):raise RuntimeError("Image coordinates differ")
            with torch.inference_mode():
                parts=[]
                for chunk in raw["crop_boxes"].split(32):
                    crops=torch.stack([crop_tensor(image,b.tolist()) for b in chunk]).cuda()
                    parts.append(encoder(crops).float().cpu())
                x=F.normalize(torch.cat(parts),dim=-1)
            save_torch(path,dict(image_id=iid,protocol_sha256=reg,raw_sha256=sha256(rawpath),features=x,crop_boxes=raw["crop_boxes"]))
        if n%10==0 or n==len(protocol["training_ids"]):progress(stage,start,n,len(protocol["training_ids"]),"detected_box_dino")
    finish(stage,dict(status="complete",protocol_sha256=reg,images=len(protocol["training_ids"]),elapsed_seconds=time.monotonic()-start))


def training_outcomes(out,cache):
    from vl_sgdet_feasibility import all_candidates
    protocol,reg=verify(out);require_stage(out,"features",reg);stage=out/"outcomes";start=time.monotonic()
    if (stage/"summary.json").exists():return
    _,post=make_post(stage);metric=OfficialMetrics("sgdet")
    state=torch.load(str(CHECKPOINT),map_location="cpu");k=KS.index(50)
    counts=dict(actions=0,effective=0,safe=0)
    for n,iid in enumerate(protocol["training_ids"],1):
        path=cache/"outcomes"/(iid+".pt")
        if path.exists():
            record=checked_load(path,reg)
        else:
            raw=checked_load(cache/"raw"/(iid+".pt"),reg);feature=checked_load(cache/"features"/(iid+".pt"),reg)
            if not torch.equal(feature["crop_boxes"],raw["crop_boxes"]):raise RuntimeError("Proposal feature order changed")
            gt=ground_truth(raw);y=targets(raw,gt,"sgdet")
            if not np.array_equal(y,raw["target"].numpy()):raise RuntimeError("Training target drift")
            with torch.no_grad():
                baseline=raw["native_logits"].cuda();base=replay(post,raw,baseline)
                base_row=metric.row(iid,base,gt,dict(logits=baseline),y)
                candidate,eligible,_,legacy=all_candidates(feature["features"],raw["native_logits"],state)
                expert=F.linear(feature["features"],state["expert_head"]["weight"],state["expert_head"]["bias"]).softmax(-1)
                indices=eligible.nonzero(as_tuple=False).flatten().tolist();xx=[];ee=[];ss=[]
                for index in indices:
                    logits=baseline.clone();logits[index]=candidate[index].cuda();prediction=replay(post,raw,logits)
                    xx.append(action_features(index,legacy,raw["native_logits"],feature["features"],expert,state["centers"],base,prediction))
                    row=metric.row(iid,prediction,gt,dict(logits=logits),y)
                    preserved=preserves_relations(base_row,row,k)
                    ee.append(row["post_nms_correct"]!=base_row["post_nms_correct"] or not preserved)
                    ss.append(improves_identity(base_row,row) and preserved)
                record=dict(image_id=iid,protocol_sha256=reg,x=torch.from_numpy(np.asarray(xx,dtype=np.float64).reshape(-1,len(FEATURE_NAMES))),
                    indices=indices,effect=torch.tensor(ee,dtype=torch.bool),safe=torch.tensor(ss,dtype=torch.bool))
                save_torch(path,record)
                for key in KS:metric.result["sgdet_recall"][key].clear()
        counts["actions"]+=len(record["indices"]);counts["effective"]+=int(record["effect"].sum());counts["safe"]+=int(record["safe"].sum())
        if n%10==0 or n==len(protocol["training_ids"]):progress(stage,start,n,len(protocol["training_ids"]),"training_post_nms_outcomes")
    finish(stage,dict(status="complete",protocol_sha256=reg,images=len(protocol["training_ids"]),elapsed_seconds=time.monotonic()-start,**counts))


def fit(out,cache):
    from vm_localization_router import localization_features, fit_localization
    protocol,reg=verify(out);require_stage(out,"outcomes",reg);stage=out/"fit";start=time.monotonic()
    if (stage/"summary.json").exists():return
    xx=[];ee=[];ss=[];lx=[];ly=[]
    for n,iid in enumerate(protocol["training_ids"],1):
        row=checked_load(cache/"outcomes"/(iid+".pt"),reg)
        xx.append(row["x"].numpy());ee.extend(row["effect"].tolist());ss.extend(row["safe"].tolist())
        raw=checked_load(cache/"raw"/(iid+".pt"),reg);f=checked_load(cache/"features"/(iid+".pt"),reg)
        y=raw["target"].numpy();keep=y>=0
        lx.append(localization_features(f["features"],raw["native_logits"],raw["proposal_boxes"],raw["size"])[keep]);ly.append((y[keep]>0).astype(int))
        if n%100==0 or n==len(protocol["training_ids"]):progress(stage,start,n,len(protocol["training_ids"]),"fit_input_assembly")
    selector=fit_selector(np.concatenate(xx),ee,ss);selector["protocol_sha256"]=reg
    localizer=fit_localization(np.concatenate(lx),np.concatenate(ly));localizer["protocol_sha256"]=reg
    lock(out/"selector.json",selector);lock(out/"localization_head.json",localizer)
    finish(stage,dict(status="complete",protocol_sha256=reg,images=len(protocol["training_ids"]),effective_actions=sum(ee),safe_actions=sum(ss),
        selector_sha256=sha256(out/"selector.json"),localization_sha256=sha256(out/"localization_head.json"),elapsed_seconds=time.monotonic()-start))


def evaluate(out,cache):
    from ve_experiment import lookup
    from vl_sgdet_feasibility import all_candidates
    from vm_localization_router import localization_features, choose_localized
    from r19_vk_decomposition import cached_prediction_check
    protocol,reg=verify(out);require_stage(out,"fit",reg);stage=out/"evaluate";start=time.monotonic()
    if (stage/"summary.json").exists():return
    cfg,post=make_post(stage);ds=dataset(cfg,"val");mapping=lookup(ds);metric=OfficialMetrics("sgdet")
    selector=read(out/"selector.json");localizer=read(out/"localization_head.json")
    if selector["protocol_sha256"]!=reg or localizer["protocol_sha256"]!=reg:raise RuntimeError("Fitted head provenance drift")
    state=torch.load(str(CHECKPOINT),map_location="cpu");records=[]
    for n,iid in enumerate(protocol["evaluation_ids"],1):
        path=stage/"images"/(iid+".json")
        if path.exists():
            record=read(path)
            if record["protocol_sha256"]!=reg:raise RuntimeError("Resume mismatch")
            records.append(record);continue
        raw=torch.load(str(VALID_RAW/(iid+".pt")),map_location="cpu")
        f=torch.load(str(VK_CACHE/"sgdet/gate/features"/(iid+".pt")),map_location="cpu")
        original=torch.load(str(VK_CACHE/"sgdet/gate/export"/(iid+".pt")),map_location="cpu")
        if f["protocol_sha256"]!=sha256(VK_MANIFEST) or not torch.equal(raw["native_logits"],original["baseline"]) or not torch.equal(f["crop_boxes"],original["crop_boxes"]):raise RuntimeError("Consumed-validation cache drift")
        with torch.no_grad():
            baseline=raw["native_logits"].cuda();base=replay(post,raw,baseline)
            cached_prediction_check(base,VK_OUT/"sgdet/gate/native/predictions"/(iid+".npz"))
            candidate,eligible,_,legacy=all_candidates(f["features"],raw["native_logits"],state)
            expert=F.linear(f["features"],state["expert_head"]["weight"],state["expert_head"]["bias"]).softmax(-1)
            indices=eligible.nonzero(as_tuple=False).flatten().tolist();xx=[]
            for index in indices:
                logits=baseline.clone();logits[index]=candidate[index].cuda();trial=replay(post,raw,logits)
                xx.append(action_features(index,legacy,raw["native_logits"],f["features"],expert,state["centers"],base,trial))
            x=np.asarray(xx,dtype=np.float64).reshape(-1,len(FEATURE_NAMES));_,choices=select_action(selector,x,indices)
            local_x=localization_features(f["features"],raw["native_logits"],raw["proposal_boxes"],raw["size"])
            chosen,choices,_=choose_localized(localizer,local_x,indices,choices)
            updated=baseline.clone()
            if chosen is not None:updated[chosen]=candidate[chosen].cuda()
            prediction=replay(post,raw,updated) if chosen is not None else base
            gt=ds.get_groundtruth(mapping[iid],evaluation=True);y=targets(raw,gt,"sgdet")
            b=metric.row(iid,base,gt,dict(logits=baseline),y);u=metric.row(iid,prediction,gt,dict(logits=updated),y)
            record=dict(image_id=iid,protocol_sha256=reg,baseline=b,updated=u,chosen=chosen,choices=choices,
                chosen_target_for_audit=int(y[chosen]) if chosen is not None else None)
            atomic_json(path,record);records.append(record)
            for key in KS:metric.result["sgdet_recall"][key].clear()
        if n%10==0 or n==1000:progress(stage,start,n,1000,"frozen_policy_exploratory_evaluation")
    b=[r["baseline"] for r in records];u=[r["updated"] for r in records]
    finish(stage,dict(status="complete",protocol_sha256=reg,images=len(records),baseline=summarize_rows(b),updated=summarize_rows(u),
        exploratory_numeric_criteria=paired_gate(b,u),selected_images=sum(r["chosen"] is not None for r in records),
        chosen_positive=sum(r["chosen_target_for_audit"] is not None and r["chosen_target_for_audit"]>0 for r in records),
        chosen_background=sum(r["chosen_target_for_audit"]==0 for r in records),
        independent_confirmation=False,formal_gate_accepted=False,test_evaluated=False,
        next_step="No automatic expansion. Inspect joint criteria and untouched-data availability before any confirmation.",elapsed_seconds=time.monotonic()-start))


def require_stage(out,name,reg):
    row=read(out/name/"summary.json")
    if row["status"]!="complete" or row["protocol_sha256"]!=reg:raise RuntimeError("Incomplete prerequisite: "+name)


def finish(stage,summary):
    atomic_json(stage/"summary.json",summary)
    atomic_json(stage/"progress.json",dict(status="complete",images=summary["images"],total=summary["images"]))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["export","features","outcomes","fit","evaluate"],required=True)
    p.add_argument("--smoke",action="store_true");a=p.parse_args()
    ensure_storage();torch.set_num_threads(2)
    if a.stage!="features":
        from vk_gate_native import configure_backend
        configure_backend()
    out,cache=locations(a.smoke)
    guard=output_path(out/"execution.lock").open("w");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.smoke and a.stage in ["fit","evaluate"]:raise ValueError("Smoke only tests export/features/outcomes, not scientific efficacy")
    if a.stage=="export":export(out,cache,a.smoke)
    elif a.stage=="features":features(out,cache)
    elif a.stage=="outcomes":training_outcomes(out,cache)
    elif a.stage=="fit":fit(out,cache)
    else:evaluate(out,cache)


if __name__=="__main__":main()
