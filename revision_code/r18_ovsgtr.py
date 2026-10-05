"""Bounded open-vocabulary SGG audit on the author's GLIP-unseen split."""
import argparse
from functools import reduce
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

from common import ROOT, OLD, atomic_json, output_path, sha256, ensure_storage

REPO=OLD/"external/official_repos/OvSGTR"
sys.path[:0]=[str(REPO),str(REPO/"GroundingDINO")]
ASSETS=ROOT/"external/R18_ovsgtr"
KS=[1,5,10,20,50,100]


def bootstrap(values):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    if not len(x):return dict(n_images=0,mean=None,ci95=None)
    rng=np.random.default_rng(17)
    means=[float(x[rng.integers(len(x),size=len(x))].mean()) for _ in range(2000)]
    return dict(n_images=len(x),mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())


def score(graph,gt,base_objects,novel_predicates):
    from datasets.sgg_metrics import SGRecall
    from util.box_ops import box_iou
    boxes,labels,rels=gt
    rels=np.asarray(rels,dtype=int)
    labels=np.asarray(labels)
    graph={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in graph.items()}
    pb=graph["pred_boxes"];pl=graph["pred_boxes_class"].numpy()
    pairs=graph["all_node_pairs"].numpy();probs=graph["all_relation"].numpy()
    collector={};ev=SGRecall(collector);ev.register_container("sgdet")
    collector["sgdet_recall"]={k:[] for k in KS}
    local=dict(pred_rel_inds=pairs,rel_scores=probs,gt_rels=rels,gt_classes=labels,
        gt_boxes=np.asarray(boxes),pred_boxes=pb.numpy(),pred_classes=pl,
        obj_scores=graph["pred_boxes_score"].numpy())
    matches=ev.calculate_recall(dict(iou_thres=.5),local,"sgdet")["pred_to_gt"] if len(pairs) else []
    strata={"all":np.ones(len(rels),dtype=bool),
            "novel_object_endpoint":np.array([labels[s] not in base_objects or labels[o] not in base_objects for s,o,p in rels]),
            "novel_predicate":np.isin(rels[:,2],list(novel_predicates))}
    strata["base_objects_and_predicate"]=~(strata["novel_object_endpoint"]|strata["novel_predicate"])
    result={}
    for name,keep in strata.items():
        ids=set(np.flatnonzero(keep));values={}
        for k in KS:
            hit=set(reduce(np.union1d,matches[:k],np.array([],dtype=int)).astype(int))
            values[str(k)]=len(hit&ids)/len(ids) if ids else None
        result[name]=dict(ground_truth_relations=len(ids),R=values)
    count=np.bincount(rels[:,2],minlength=51)[1:]
    hit=reduce(np.union1d,matches[:50],np.array([],dtype=int)).astype(int)
    correct=np.bincount(rels[hit,2],minlength=51)[1:]
    result["class_recall50"]=[float(a/b) if b else None for a,b in zip(correct,count)]
    ious=box_iou(torch.as_tensor(boxes),pb)[0]
    if len(pb):
        best,idx=ious.max(1);idx=idx.numpy();localized=best.numpy()>=.5
        lookup={tuple(p):i for i,p in enumerate(pairs.tolist())}
        evidence=[]
        for s,o,p in rels:
            if not (localized[s] and localized[o]) or idx[s]==idx[o]:continue
            j=lookup.get((idx[s],idx[o]))
            if j is None:continue
            evidence.append(dict(identity_correct=bool(pl[idx[s]]==labels[s] and pl[idx[o]]==labels[o]),
                predicate_correct=bool(1+probs[j,1:].argmax()==p),
                novel_endpoint=bool(labels[s] not in base_objects or labels[o] not in base_objects)))
        result["identity_audit"]=dict(objects=len(labels),localized_objects=int(localized.sum()),
            identity_correct_objects=int(((pl[idx]==labels)&localized).sum()),
            localized_pairs=len(evidence),source_relations=len(rels),pairs=evidence,
            matching="class-agnostic maximum IoU >=0.5 per GT object; reused matches permitted; distinct endpoints required")
    else:result["identity_audit"]=dict(objects=len(labels),localized_objects=0,identity_correct_objects=0,
                                     localized_pairs=0,source_relations=len(rels),pairs=[])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");args=p.parse_args()
    ensure_storage();torch.set_num_threads(4)
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    out=ROOT/"results/R18_ovsgtr"/("smoke" if args.smoke else "formal")
    from util.slconfig import SLConfig
    from groundingdino.models.GroundingDINO import ms_deform_attn
    # The upstream differentiable PyTorch implementation supports GPU tensors.
    # Use it explicitly when the optional compiled CUDA extension is absent.
    ms_deform_attn.MultiScaleDeformableAttnFunction.apply=lambda value,shapes,start,loc,weights,step: (
        ms_deform_attn.multi_scale_deformable_attn_pytorch(value,shapes,loc,weights))
    from models.GroundingDINO.groundingdino import build_groundingdino
    from models.GroundingDINO import ms_deform_attn as relation_deform_attn
    relation_deform_attn.MultiScaleDeformableAttnFunction.apply=lambda value,shapes,start,loc,weights,step: (
        relation_deform_attn.multi_scale_deformable_attn_pytorch(value,shapes,loc,weights))
    from datasets.vg import build_vg,VG150_BASE_OBJ_CATEGORIES,VG150_NOVEL_PREDICATE
    import datasets.vg as vg
    from r16_motifs import image_paths
    vg.load_image_filenames=image_paths
    cfg=SLConfig.fromfile(str(REPO/"config/GroundingDINO_SwinT_OGC_ovdr.py"))
    config=argparse.Namespace(**cfg._cfg_dict.to_dict())
    config.device="cuda";config.data_path=str(ROOT/"data/R18_ovsgtr")
    config.text_encoder_type=str(ASSETS/"bert-base-uncased")
    config.fix_size=False;config.eval=True
    ds=build_vg("test",config,disable_transforms=True)
    from datasets.coco import make_coco_transforms
    transform=make_coco_transforms("val",args=config)
    eligible=[i for i in range(len(ds)) if len(ds.annotations[i]["labels"])<=100]
    selected=np.random.default_rng(17).permutation(eligible)[:(2 if args.smoke else 1000)].tolist()
    weight=OLD/"checkpoints/sgg/weights/ovsgtr/vg/vg-ovdr-swint.pth"
    protocol=dict(model="OvSGTR OvD+R Swin-T",checkpoint=str(weight),sha256=sha256(weight),
        source_sha256={str(f.relative_to(REPO)):sha256(f) for d in [REPO/"models",REPO/"datasets",REPO/"GroundingDINO/groundingdino"] for f in d.rglob("*.py")},
        code_sha256=sha256(Path(__file__)),split="author split_GLIPunseen test",seed=17,
        dataset_sha256=sha256(ROOT/"data/R18_ovsgtr/visual_genome/stanford_filtered/VG-SGG.h5"),
        image_ids=[int(ds.ids[i]) for i in selected],candidate_images=len(ds),excluded_over_100_objects=len(ds)-len(eligible),
        control_rule="first 200 eligible images in fixed selected order; key vs degree-zero, area ratio 0.5-2, strengths0.5/1",
        scope="One open-vocabulary detector/relation model; not evidence for all VLMs/MLLMs; subset, not official full reproduction",
        deformation_operator="upstream PyTorch reference on CUDA",input="image plus full fixed 150 object/50 predicate vocabulary; no GT supplied to model")
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text())!=protocol:
        raise RuntimeError("R18 protocol drift")
    atomic_json(out/"protocol.json",protocol)
    model,criterion,post=build_groundingdino(config)
    payload=torch.load(str(weight),map_location="cpu")
    state={k[7:] if k.startswith("module.") else k:v for k,v in payload["model"].items()}
    incompatible=model.load_state_dict(state,strict=False)
    # BERT position_ids changed from persistent to nonpersistent across versions.
    missing=[k for k in incompatible.missing_keys if not k.endswith("position_ids")]
    unexpected=[k for k in incompatible.unexpected_keys if not k.endswith("position_ids")]
    if missing or unexpected:raise RuntimeError("Checkpoint coverage mismatch: "+str((missing,unexpected)))
    atomic_json(out/"loading.json",dict(missing=incompatible.missing_keys,unexpected=incompatible.unexpected_keys,
                                      parameters=sum(p.numel() for p in model.parameters())))
    model=model.cuda().eval();post=post["bbox"].cuda().eval()
    for name in ["rln_proj","rln_classifier","rln_freq_bias"]:setattr(post,name,getattr(model,name,None))
    post.name2classes=ds.name2classes;post.name2predicates=ds.name2predicates
    assert not post.use_gt_box
    from util.misc import nested_tensor_from_tensor_list
    from paired_visual_control import selected_nodes,mask
    base_objects={ds.name2classes[n] for n in VG150_BASE_OBJ_CATEGORIES if n!="__background__"}
    novel_predicates={ds.name2predicates[n] for n in VG150_NOVEL_PREDICATE}
    @torch.no_grad()
    def infer(image,target):
        x,t=transform(image,dict(target))
        sample=nested_tensor_from_tensor_list([x.cuda()])
        prediction=model(sample,captions=[target["caption"]],rel_captions=[target["rel_caption"]])
        return post(prediction,target["orig_size"][None].cuda())[0]["graph"]
    rows=[];controls=0;started=time.monotonic()
    for n,i in enumerate(selected):
        iid=str(ds.ids[i]);dest=out/"images"/(iid+".json")
        if dest.exists():
            row=json.loads(dest.read_text());rows.append(row);controls+=int(bool(row["controls"]));continue
        image,target=ds[i];gt=ds.get_groundtruth(i)
        graph=infer(image,target);clean=score(graph,gt,base_objects,novel_predicates)
        tensors={k:v.cpu().numpy() for k,v in graph.items() if isinstance(v,torch.Tensor)}
        np.savez_compressed(output_path(out/"predictions"/(iid+".npz")),**tensors)
        row=dict(image_id=iid,clean=clean,controls={},exclusion=None)
        nodes,reason=selected_nodes(np.asarray(gt[0]),np.asarray(gt[2])[:,:2])
        if nodes and controls<200:
            for mode,node in [("key",nodes[0]),("unrelated",nodes[1])]:
                for strength in [.5,1.]:
                    modified=mask(image,np.asarray(gt[0])[node],strength)
                    if np.array_equal(np.asarray(image),np.asarray(modified)):raise RuntimeError("Ineffective image intervention")
                    row["controls"][mode+"_"+str(strength)]=score(infer(modified,target),gt,base_objects,novel_predicates)
            controls+=1
        else:row["exclusion"]=reason or "control_budget_reached"
        atomic_json(dest,row);rows.append(row)
        progress=dict(images=n+1,total=len(selected),controlled_images=controls,seconds=time.monotonic()-started,
                      eta_seconds=(time.monotonic()-started)/(n+1)*(len(selected)-n-1))
        atomic_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
    results={}
    for stratum in ["all","novel_object_endpoint","novel_predicate","base_objects_and_predicate"]:
        values=[r["clean"][stratum]["R"]["50"] for r in rows]
        results[stratum]=bootstrap([v for v in values if v is not None])
        results[stratum]["ground_truth_relations"]=sum(r["clean"][stratum]["ground_truth_relations"] for r in rows)
    contrasts={}
    for strength in [.5,1.]:
        subset=[r for r in rows if r["controls"]]
        contrasts[str(strength)]=bootstrap([r["controls"]["key_"+str(strength)]["all"]["R"]["50"]-
             r["controls"]["unrelated_"+str(strength)]["all"]["R"]["50"] for r in subset])
    c=np.asarray([r["clean"]["class_recall50"] for r in rows],dtype=float)
    support=np.isfinite(c).sum(0)
    macro=np.divide(np.nansum(c,0),support,out=np.zeros(50),where=support>0)
    audits=[r["clean"]["identity_audit"] for r in rows]
    pairs=[p for a in audits for p in a["pairs"]]
    identity={k:sum(a[k] for a in audits) for k in ["objects","localized_objects","identity_correct_objects","localized_pairs","source_relations"]}
    identity["conditional_object_accuracy"]=identity["identity_correct_objects"]/identity["localized_objects"] if identity["localized_objects"] else None
    identity["pair_coverage"]=identity["localized_pairs"]/identity["source_relations"] if identity["source_relations"] else None
    identity["endpoint_error_rate"]=sum(not p["identity_correct"] for p in pairs)/len(pairs) if pairs else None
    identity["predicate_hit_by_identity"]={}
    for correct in [True,False]:
        group=[p for p in pairs if p["identity_correct"]==correct]
        identity["predicate_hit_by_identity"][str(correct)]=dict(pairs=len(group),
            accuracy=sum(p["predicate_correct"] for p in group)/len(group) if group else None)
    atomic_json(out/"summary.json",dict(status="complete",images=len(rows),controlled_images=controls,
              R50=results,mR50_all_50=float(macro.mean()),predicate_image_support=support.tolist(),identity=identity,
              paired_key_minus_unrelated_R50=contrasts,protocol_sha256=sha256(out/"protocol.json")))


if __name__=="__main__":main()
