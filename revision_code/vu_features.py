"""Modern-runtime SigLIP2 extraction on original predicted boxes, without labels."""
import argparse
import fcntl
import json
import os
import time

import torch
from torch.nn import functional as F
from PIL import Image

from common import atomic_json,ensure_storage,output_path,sha256,canonical_path
from vu_protocol import OUT,CACHE,load
from vu_math import crop_bounds


def main():
    import transformers
    from transformers import AutoModel,AutoImageProcessor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,choices=[0,1],required=True)
    parser.add_argument("--smoke",action="store_true"); args=parser.parse_args()
    ensure_storage(); torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    protocol,reg,lane=load(args.smoke,check_assets=True)
    stage=OUT/lane/("extract%d"%args.shard)
    guard=output_path(stage/"run.lock").open("w");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=protocol["inputs"][args.shard::2]; start=time.monotonic()
    parent=AutoModel.from_pretrained(protocol["model_dir"],local_files_only=True)
    model=getattr(parent,"vision_model",None)
    if model is None: raise RuntimeError("FixRes checkpoint lacks vision_model")
    model=model.float().cuda().eval().requires_grad_(False); del parent
    processor=AutoImageProcessor.from_pretrained(protocol["model_dir"],local_files_only=True,use_fast=False)
    manifest=[]; crop_count=0; parity=[]
    for index,entry in enumerate(rows):
        if time.monotonic()-start>6*3600: raise TimeoutError("Extraction stage budget")
        if sha256(entry["raw"])!=entry["raw_sha256"]: raise RuntimeError("Raw input drift")
        raw=torch.load(entry["raw"],map_location="cpu",weights_only=False)
        iid=entry["image_id"]
        if raw["image_id"]!=iid or raw["protocol_sha256"]!=protocol["vn_protocol_sha256"]:
            raise RuntimeError("Proposal order/provenance mismatch")
        image_path=canonical_path(raw["image"]); image_sha=sha256(image_path)
        path=CACHE/lane/"features"/(iid+".pt")
        if path.exists():
            existing=torch.load(str(path),map_location="cpu",weights_only=False)
            if (existing["protocol_sha256"]!=reg or existing["raw_sha256"]!=entry["raw_sha256"]
                or existing["image_sha256"]!=image_sha or not torch.equal(existing["crop_boxes"],raw["crop_boxes"])):
                raise RuntimeError("Resumed feature cache changed")
        else:
            with Image.open(image_path) as source: image=source.convert("RGB")
            if tuple(image.size)!=tuple(raw["image_size"]): raise RuntimeError("Image coordinate system mismatch")
            # Only predicted crop geometry and pixels enter this encoder.
            crops=[image.crop(crop_bounds(b.tolist(),image.size,scale))
                   for scale in [1.,1.5] for b in raw["crop_boxes"]]
            parts=[]
            with torch.inference_mode():
                for offset in range(0,len(crops),16):
                    pixels=processor(images=crops[offset:offset+16],return_tensors="pt")["pixel_values"].cuda()
                    pooled=model(pixel_values=pixels).pooler_output
                    if pooled.ndim!=2 or pooled.shape[1]!=768 or not torch.isfinite(pooled).all():
                        raise RuntimeError("Unexpected SigLIP pooler output")
                    parts.append(F.normalize(pooled.float(),dim=-1).cpu())
                features=torch.cat(parts); n=len(raw["crop_boxes"])
                if args.smoke and index==0:
                    single=F.normalize(model(pixel_values=processor(images=[crops[0]],return_tensors="pt")["pixel_values"].cuda()).pooler_output.float(),dim=-1).cpu()
                    error=float((single-features[:1]).abs().max())
                    if not torch.allclose(single,features[:1],atol=2e-5,rtol=2e-4): raise RuntimeError("Crop batch parity failed")
                    parity.append(dict(image_id=iid,max_feature_error=error))
            payload=dict(image_id=iid,protocol_sha256=reg,raw_sha256=entry["raw_sha256"],
                image=str(image_path),image_sha256=image_sha,crop_boxes=raw["crop_boxes"],
                siglip_tight=features[:n],siglip_context=features[n:],
                torch_version=str(torch.__version__),transformers_version=transformers.__version__)
            target=output_path(path); temp=target.with_suffix(".tmp")
            torch.save(payload,str(temp),pickle_protocol=2);os.replace(temp,target)
        crop_count+=2*len(raw["crop_boxes"])
        manifest.append(dict(image_id=iid,feature_sha256=sha256(path),image_sha256=image_sha))
        if (index+1)%10==0 or index+1==len(rows):
            elapsed=time.monotonic()-start
            status=dict(status="running",pid=os.getpid(),stage="siglip2_dual_view",images=index+1,total=len(rows),
                seconds=elapsed,eta_seconds=elapsed/(index+1)*(len(rows)-index-1),crops=crop_count)
            atomic_json(stage/"progress.json",status);print(json.dumps(status),flush=True)
    atomic_json(stage/"summary.json",dict(status="complete",images=len(rows),protocol_sha256=reg,
        manifest=manifest,crops=crop_count,smoke_crop_parity=parity,elapsed_seconds=time.monotonic()-start,
        offline_exhaustive_extraction=True,GT_used_by_encoder=False))
    atomic_json(stage/"progress.json",dict(status="complete",images=len(rows),total=len(rows)))


if __name__=="__main__":main()
