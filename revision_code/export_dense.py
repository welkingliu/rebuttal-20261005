"""Export frozen native logits for R3 without changing any historical cache."""
import argparse
import json
import time

import numpy as np
import torch

from common import ROOT, ensure_storage, atomic_json, output_path, sha256
from native_runtime import load_model, dataset, infer, image_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["sgcls","sgdet"],required=True)
    p.add_argument("--split",choices=["validation","test"],required=True)
    args=p.parse_args()
    ensure_storage()
    torch.set_num_threads(4)
    out=ROOT/"cache/R3_dense"/args.split/args.task
    model,cfg,transform,provenance=load_model("tde_motifs",args.task,out)
    ds=dataset(cfg,"train" if args.split=="validation" else "test",all_training=args.split=="validation")
    ids={image_id(ds,i):i for i in range(len(ds))}
    if args.split=="validation":
        requested=json.loads((ROOT/"manifests/R3_validation_ids.json").read_text())["validation_image_ids"]
        missing=set(requested)-set(ids)
        if missing:
            raise RuntimeError("Native data omitted original validation IDs: %s"%sorted(missing))
    else:
        requested=list(ids)
        if len(requested)!=26446:
            raise RuntimeError("Full native VG test split must contain 26446 images")
    protocol=dict(model=provenance,image_ids=requested,split=args.split,task=args.task,
                  output_contract="dense refined logits plus native final labels, scores, boxes and pairs",
                  proposal_policy="native proposals preserved; class-specific NMS not rerun for offline label/score swaps")
    protocol_path=out/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise RuntimeError("Dense export resume protocol changed")
    atomic_json(protocol_path,protocol)
    captured={}
    def hook(module,inputs):
        _,logits=inputs[0]
        if module.attribute_on:
            logits=logits[0]
        captured["logits"]=logits[0].detach()
    handle=model.roi_heads["relation"].post_processor.register_forward_pre_hook(hook)
    started=time.monotonic()
    try:
        for n,iid in enumerate(requested):
            dest=out/"images"/(iid+".npz")
            if dest.exists():
                with np.load(dest) as check:
                    assert "object_logits" in check and check["object_logits"].shape[1]==151
                continue
            captured.clear()
            pred,_=infer(model,cfg,transform,ds,ids[iid])
            gt=ds.get_groundtruth(ids[iid],evaluation=True)
            logits=captured["logits"].float().cpu().numpy()
            if len(logits)!=len(pred):
                raise RuntimeError("Native logits not proposal-aligned")
            w,h=pred.size
            gw,gh=gt.size
            values=dict(object_logits=logits,
                        native_labels=pred.get_field("pred_labels").cpu().numpy(),
                        native_scores=pred.get_field("pred_scores").cpu().numpy(),
                        pred_boxes=pred.bbox.float().cpu().numpy()/np.array([w,h,w,h],dtype=np.float32),
                        pred_rel_pairs=pred.get_field("rel_pair_idxs").cpu().numpy(),
                        pred_rel_scores=pred.get_field("pred_rel_scores").cpu().numpy(),
                        gt_boxes=gt.bbox.numpy()/np.array([gw,gh,gw,gh],dtype=np.float32),
                        gt_labels=gt.get_field("labels").numpy(),gt_relations=gt.get_field("relation_tuple").numpy())
            temporary=dest.with_suffix(".tmp")
            with output_path(temporary).open("wb") as f:
                np.savez_compressed(f,**values)
            temporary.replace(output_path(dest))
            if (n+1)%50==0:
                row=dict(images=n+1,total=len(requested),seconds=time.monotonic()-started)
                atomic_json(out/"progress.json",row)
                print(json.dumps(row),flush=True)
    finally:
        handle.remove()
    atomic_json(out/"summary.json",dict(status="complete",images=len(requested),
        protocol_sha256=sha256(protocol_path),elapsed_seconds=time.monotonic()-started))
    print("[COMPLETE] R3 dense",args.split,args.task,flush=True)


if __name__=="__main__":
    main()
