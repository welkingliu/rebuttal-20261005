"""R4 replacement audit: identical image/pair denominators for visual controls.

One highest-degree node versus an annotated degree-zero node matched within a
factor of two in box area. Masking strength varies, node count remains one.
This intentionally declares a corrected protocol, not a reconstruction of the
unavailable historical per-image intervention records.
"""
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image
import torch

from common import ROOT, ensure_storage, atomic_json, output_path, sha256
from native_runtime import load_model, dataset, image_id


def selected_nodes(boxes, pairs):
    degree=np.bincount(pairs.reshape(-1),minlength=len(boxes))
    key=int(degree.argmax())
    candidates=np.flatnonzero(degree==0)
    if not len(candidates):
        return None,"no_annotated_unrelated_node"
    areas=np.maximum(boxes[:,2]-boxes[:,0],0)*np.maximum(boxes[:,3]-boxes[:,1],0)
    if areas[key]<=0:
        return None,"degenerate_key_box"
    ratios=np.maximum(areas[candidates],1e-8)/areas[key]
    choice=int(np.argmin(np.abs(np.log(ratios))))
    ratio=float(ratios[choice])
    if not .5<=ratio<=2.:
        return None,"unrelated_area_not_matched"
    return (key,int(candidates[choice]),ratio),None


def mask(image,box,strength):
    values=np.asarray(image).copy()
    x1,y1,x2,y2=np.round(box).astype(int)
    h,w=values.shape[:2]
    x1,x2=np.clip([x1,x2],0,w)
    y1,y2=np.clip([y1,y2],0,h)
    if x2<=x1 or y2<=y1:
        raise ValueError("Empty masking region")
    fill=values.mean((0,1),keepdims=True)
    region=values[y1:y2,x1:x2]
    values[y1:y2,x1:x2]=np.clip((1-strength)*region+strength*fill,0,255).round().astype(np.uint8)
    return Image.fromarray(values)


def infer_pairs(model,cfg,transform,image,target):
    from pysgg.structures.image_list import to_image_list
    gt=target.get_field("relation_tuple").long()
    pairs=torch.unique(gt[:,:2],dim=0)
    relation=model.roi_heads["relation"]
    previous=relation.samp_processor.prepare_test_pairs
    relation.samp_processor.prepare_test_pairs=lambda device,proposals:[pairs.to(device)]
    result={}
    def hook(module,args,output):
        result["scores"]=output[1][0].detach().cpu().numpy()
    handle=relation.predictor.register_forward_hook(hook)
    try:
        tensor,resized=transform(image,target)
        inputs=to_image_list([tensor.cuda()],size_divisible=int(cfg.DATALOADER.SIZE_DIVISIBILITY))
        with torch.no_grad():
            model(inputs,[resized.to("cuda")],logger=None)
    finally:
        handle.remove()
        relation.samp_processor.prepare_test_pairs=previous
    lookup={tuple(p):i for i,p in enumerate(pairs.tolist())}
    return result["scores"][[lookup[tuple(p)] for p in gt[:,:2].tolist()]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family",choices=["transformer","tde_motifs"],required=True)
    args=p.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    out=ROOT/"results/R4_paired"/args.family
    model,cfg,transform,provenance=load_model(args.family,"sgcls",out)
    ds=dataset(cfg,"test")
    selected=json.loads((ROOT/"results/R2"/args.family/"test/protocol.json").read_text())["image_ids"]
    by_id={image_id(ds,i):i for i in range(len(ds))}
    protocol=dict(model=provenance,candidate_images=selected,strengths=[.25,.5,1.],
        node_count=1,reference="highest annotated degree vs degree-zero matched box area within factor 2",
        interpretation="Spatial visual-evidence control, not identity-specific; degree-zero is annotation-relative",
        background="Report both all-class and foreground-only top-1 on identical annotated pairs",
        historical_reconstruction=False)
    atomic_json(out/"protocol.json",protocol)
    records,excluded=[],[]
    start=time.monotonic()
    for n,iid in enumerate(selected):
        record=out/"images"/(iid+".json")
        path=out/"images"/(iid+".npz")
        if record.exists() and path.exists():
            row=json.loads(record.read_text())
            if row["protocol_sha256"]!=sha256(out/"protocol.json"):
                raise RuntimeError("Paired-control resume protocol mismatch")
            records.append(row)
            continue
        index=by_id[iid]
        target=ds.get_groundtruth(index,evaluation=True)
        gt=target.get_field("relation_tuple").numpy()
        boxes=target.bbox.numpy()
        nodes,reason=selected_nodes(boxes,gt[:,:2])
        if nodes is None:
            excluded.append(dict(image_id=iid,reason=reason))
            continue
        image,_,_=ds[index]
        clean=infer_pairs(model,cfg,transform,image,target)
        labels=gt[:,2]
        payload=dict(gt=labels,pairs=gt[:,:2],clean_logits=clean)
        stats={}
        for mode,node in [("key",nodes[0]),("unrelated",nodes[1])]:
            for strength in (.25,.5,1.):
                modified=mask(image,boxes[node],strength)
                if np.array_equal(np.asarray(image),np.asarray(modified)):
                    raise RuntimeError("Mask did not change consumed image")
                scores=infer_pairs(model,cfg,transform,modified,target)
                name="%s_%.2f"%(mode,strength)
                payload[name]=scores
                stats[name]={scope:float((pred==labels).mean()) for scope,pred in
                    [("foreground",scores[:,1:].argmax(1)+1),("all_classes",scores.argmax(1))]}
        row=dict(image_id=iid,relations=len(labels),key_node=nodes[0],control_node=nodes[1],area_ratio=nodes[2],
            protocol_sha256=sha256(out/"protocol.json"),clean={scope:float((pred==labels).mean()) for scope,pred in
                [("foreground",clean[:,1:].argmax(1)+1),("all_classes",clean.argmax(1))]},conditions=stats)
        np.savez_compressed(output_path(path),**payload)
        atomic_json(record,row)
        records.append(row)
        if (n+1)%25==0:
            progress=dict(candidate_images=n+1,total=len(selected),paired_images=len(records),excluded=len(excluded),seconds=time.monotonic()-start)
            atomic_json(out/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    if len(records)<100:
        raise RuntimeError("Insufficient matched images: %d"%len(records))
    summary={}
    rng=np.random.default_rng(17)
    for strength in (.25,.5,1.):
        for scope in ("foreground","all_classes"):
            values=np.array([[r["clean"][scope],r["conditions"]["key_%.2f"%strength][scope],
                              r["conditions"]["unrelated_%.2f"%strength][scope]] for r in records])
            delta=values[:,1]-values[:,2]
            ci=np.quantile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)],[.025,.975])
            summary["%s_%.2f"%(scope,strength)]=dict(clean=float(values[:,0].mean()),key=float(values[:,1].mean()),
                unrelated=float(values[:,2].mean()),key_minus_unrelated=float(delta.mean()),paired_95ci=ci.tolist(),
                estimand="macro image predicate top1, same image/pair set in all conditions")
    atomic_json(out/"summary.json",dict(status="complete",protocol=protocol,results=summary,
        paired_images=len(records),paired_relations=sum(r["relations"] for r in records),excluded=excluded))
    print("[COMPLETE] R4 paired",args.family,flush=True)


if __name__=="__main__":
    main()
