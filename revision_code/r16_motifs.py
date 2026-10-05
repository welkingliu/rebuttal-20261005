"""Plain MotifPredictor reproduction using the upstream benchmark implementation."""
import argparse
from functools import reduce
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, OLD, atomic_json, output_path, sha256, ensure_storage
from repro_experiment import DeterministicImages, save_torch

REPO = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch"
sys.path.insert(0,str(REPO))
RUN = ROOT / "results/R16_plain_motifs"
KS = [1,5,10,20,50,100]
ZERO = None


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("Protocol mismatch: "+str(path))
    atomic_json(path,value)


def image_paths(img_dir, image_file):
    info=json.loads(Path(image_file).read_text())
    roots=[OLD/"data/vg/VG_100K",OLD/"data/vg/VG_100K_2",OLD/"data/vg/VG_100K/VG_100K"]
    files=[]; meta=[]
    for row in info:
        name=str(row["image_id"])+".jpg"
        if name in {"1592.jpg","1722.jpg","4616.jpg","4617.jpg"}:
            continue
        matches=[p/name for p in roots if (p/name).is_file()]
        if not matches:
            raise FileNotFoundError(name)
        files.append(str(matches[0]));meta.append(row)
    assert len(files)==108073
    return files,meta


def dataset(cfg,split):
    import maskrcnn_benchmark.data.datasets.visual_genome as vg
    vg.load_image_filenames=image_paths
    return vg.VGDataset(split,str(OLD/"data/vg/VG_100K"),str(OLD/"data/vg/VG-SGG-with-attri.h5"),
        str(OLD/"data/vg/VG-SGG-dicts-with-attri.json"),str(OLD/"data/vg/image_data.json"),
        num_val_im=5000,filter_duplicate_rels=split=="train",filter_non_overlap=False,flip_aug=False)


def configure(task,out):
    from maskrcnn_benchmark.config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(REPO/"configs/e2e_relation_X_101_32_8_FPN_1x.yaml"))
    cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR="MotifPredictor"
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX=task!="sgdet"
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL=task=="predcls"
    cfg.MODEL.ATTRIBUTE_ON=False
    cfg.MODEL.WEIGHT=""
    cfg.GLOVE_DIR=str(OLD/"data/derived/glove")
    cfg.OUTPUT_DIR=str(out/"runtime")
    cfg.DATALOADER.NUM_WORKERS=0
    cfg.SOLVER.IMS_PER_BATCH=8
    cfg.SOLVER.BASE_LR=.01
    cfg.SOLVER.MAX_ITER=75000
    cfg.SOLVER.VAL_PERIOD=3000
    cfg.SOLVER.CHECKPOINT_PERIOD=3000
    cfg.TEST.IMS_PER_BATCH=1
    cfg.DTYPE="float32"
    cfg.freeze()
    output_path(out/"runtime/config.yml").write_text(cfg.dump())
    os.chdir(str(out/"runtime"))
    return cfg


def build(task,out,weight=None):
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    cfg=configure(task,out)
    stat="VG_stanford_filtered_with_attribute_train_statistics.cache"
    source=ROOT/"data/R16_plain_motifs/upstream_statistics.pth"
    audit=json.loads((RUN/"statistics_audit/summary.json").read_text())
    if sha256(source)!=audit["corrected_sha256"]:
        raise RuntimeError("Upstream statistics hash mismatch")
    stats=torch.load(str(source),map_location="cpu")
    vocab=json.loads((OLD/"data/vg/VG-SGG-dicts-with-attri.json").read_text())
    names=["__background__"]+[k for k,v in sorted(vocab["label_to_idx"].items(),key=lambda x:x[1]) if v>0]
    predicates=["__background__"]+[k for k,v in sorted(vocab["predicate_to_idx"].items(),key=lambda x:x[1]) if v>0]
    assert list(stats["obj_classes"])==names and list(stats["rel_classes"])==predicates
    save_torch(Path(cfg.OUTPUT_DIR)/stat,stats)
    model=build_detection_model(cfg)
    detector=OLD/"checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"
    if weight:
        payload=torch.load(str(weight),map_location="cpu")
        model.load_state_dict(payload["model"],strict=True)
    else:
        payload=torch.load(str(detector),map_location="cpu")
        marker=json.loads(detector.with_suffix(".pth.json").read_text())
        assert sha256(detector)==marker["output_sha256"]
        state=payload["model"]
        mapping={"roi_heads.relation.box_feature_extractor":"roi_heads.box.feature_extractor",
                 "roi_heads.relation.union_feature_extractor.feature_extractor":"roi_heads.box.feature_extractor"}
        DetectronCheckpointer(cfg,model)._load_model(payload,mapping)
        checked=[];new=[]
        for key,value in model.state_dict().items():
            other=key
            for dst,src in mapping.items():
                if key.startswith(dst+"."):
                    other=src+key[len(dst):]
            if not key.startswith("roi_heads.relation.") or other!=key:
                if other not in state and ".pooler.reduce_channel." in key:
                    new.append(key);continue
                if other not in state or not torch.equal(value,state[other]):
                    raise RuntimeError("Initialization mismatch: "+key)
                checked.append(key)
        atomic_json(out/"initialization.json",dict(detector=str(detector),sha256=sha256(detector),
                      verified_tensors=len(checked),new_union_parameters=new))
    return model.cuda(),cfg


def recall_row(pred,gt,task):
    global ZERO
    from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall
    pred=pred.resize(gt.size).to("cpu")
    rel=gt.get_field("relation_tuple").numpy().astype(int)
    local=dict(pred_rel_inds=pred.get_field("rel_pair_idxs").numpy()[:100],
        rel_scores=pred.get_field("pred_rel_scores").numpy()[:100],gt_rels=rel,
        gt_classes=gt.get_field("labels").numpy(),gt_boxes=gt.bbox.numpy(),pred_boxes=pred.bbox.numpy(),
        pred_classes=pred.get_field("pred_labels").numpy(),obj_scores=pred.get_field("pred_scores").numpy())
    collector={};evaluator=SGRecall(collector);evaluator.register_container(task)
    collector[task+"_recall"]={k:[] for k in KS}
    matches=evaluator.calculate_recall({"iou_thres":.5},local,task)["pred_to_gt"] if len(local["pred_rel_inds"]) else []
    count=np.bincount(rel[:,2],minlength=51)[1:]
    if ZERO is None:
        zpath=REPO/"maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch"
        ZERO={tuple(map(int,x)) for x in torch.load(str(zpath),map_location="cpu").tolist()}
    classes=local["gt_classes"]
    zero={i for i,(s,o,p) in enumerate(rel) if (int(classes[s]),int(classes[o]),int(p)) in ZERO}
    r=[];mr=[];zr=[]
    for k in KS:
        hit=reduce(np.union1d,matches[:k],np.array([],dtype=int)).astype(int)
        correct=np.bincount(rel[hit,2],minlength=51)[1:]
        r.append(len(hit)/len(rel));mr.append([float(h/c) if c else None for h,c in zip(correct,count)])
        zr.append(len(set(hit)&zero)/len(zero) if zero else None)
    return dict(R=r,class_recall=mr,relations=len(rel),zR=zr,zero_shot_relations=len(zero))


@torch.no_grad()
def evaluate(model,cfg,ds,task,out,limit=0):
    from maskrcnn_benchmark.data.transforms import build_transforms
    from maskrcnn_benchmark.structures.image_list import to_image_list
    from reproduction_pair_audit import ChunkUnion
    transform=build_transforms(cfg,is_train=False)
    model.eval()
    chunk=ChunkUnion(model.roi_heads.relation.union_feature_extractor,256)
    rows=[];start=time.monotonic()
    total=min(limit,len(ds)) if limit else len(ds)
    try:
        for i in range(total):
            image,_,_=ds[i]
            gt=ds.get_groundtruth(i,evaluation=True)
            tensor,target=transform(image,gt)
            images=to_image_list([tensor.cuda()],size_divisible=32)
            pred=model(images,[target.to("cuda")])[0]
            row=recall_row(pred,gt,task)
            iid=Path(ds.filenames[i]).stem
            row["image_id"]=iid
            if len(ds)==26446:
                from export_pysgg_vg_task import convert_prediction
                dest=output_path(out/"predictions"/(iid+".npz"))
                with dest.with_suffix(".tmp").open("wb") as f:
                    np.savez_compressed(f,**convert_prediction(pred.to("cpu")))
                dest.with_suffix(".tmp").replace(dest)
            atomic_json(out/"images"/(iid+".json"),row)
            rows.append(row)
            if (i+1)%100==0 or i==0:
                progress=dict(stage="evaluation",images=i+1,total=total,seconds=time.monotonic()-start)
                atomic_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
    finally:
        chunk.close()
    r=np.asarray([x["R"] for x in rows]).mean(0)
    c=np.asarray([x["class_recall"] for x in rows],dtype=float)
    n=np.isfinite(c).sum(0)
    mr=np.divide(np.nansum(c,0),n,out=np.zeros_like(n,dtype=float),where=n>0).mean(1)
    summary=dict(status="complete",images=total,R=dict(zip(map(str,KS),r.tolist())),mR=dict(zip(map(str,KS),mr.tolist())))
    z=np.asarray([x["zR"] for x in rows],dtype=float)
    summary["zR"]={str(k):float(np.nanmean(z[:,j])) if np.isfinite(z[:,j]).any() else None for j,k in enumerate(KS)}
    summary["zero_shot_relations"]=sum(x["zero_shot_relations"] for x in rows)
    if not limit and len(ds)==26446:
        refs={"predcls":(.6518,.1479),"sgcls":(.3892,.0828),"sgdet":(.3278,.0675)}[task]
        checks={key:dict(actual=summary[key]["50"],reference=ref,delta=summary[key]["50"]-ref,
                    passed=abs(summary[key]["50"]-ref)<=.02) for key,ref in zip(["R","mR"],refs)}
        summary.update(reference_checks=checks,reproduction_passed=all(x["passed"] for x in checks.values()))
    atomic_json(out/"summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["predcls","sgcls","sgdet"],required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();ensure_storage();torch.set_num_threads(4)
    random.seed(666);np.random.seed(666);torch.manual_seed(666);torch.cuda.manual_seed_all(666)
    out=RUN/args.task/("smoke" if args.smoke else "formal")
    steps=2 if args.smoke else 75000
    protocol=dict(task=args.task,predictor="MotifPredictor",architecture="plain non-causal Motifs",
        source="KaihuaTang/Scene-Graph-Benchmark.pytorch",task_flags=dict(gt_box=args.task!="sgdet",gt_labels=args.task=="predcls"),
        seed=666,true_batch=8,steps=steps,base_lr=.01,effective_optimizer_lr=.08,val_period=3000,
        scheduler="upstream plateau: patience2, factor0.1, max3 decays",selection="best validation R@100",
        deviation="Single GPU batch8 vs example two-GPU global batch12; iterations scaled to equal image budget; FP32",
        validation_images=5000,test_images=26446,reference_absolute_tolerance=.02,
        sources={str(f.relative_to(REPO)):sha256(f) for f in (REPO/"maskrcnn_benchmark").rglob("*.py")},
        driver_sha256=sha256(Path(__file__)))
    protocol["statistics_sha256"]=sha256(ROOT/"data/R16_plain_motifs/upstream_statistics.pth")
    protocol["statistics_semantics"]="upstream log probabilities; recomputed with upstream overlap policy"
    immutable(out/"protocol.json",protocol)
    model,cfg=build(args.task,out)
    for module in (model.backbone,model.rpn,model.roi_heads.box):
        for parameter in module.parameters():parameter.requires_grad_(False)
    from maskrcnn_benchmark.solver import make_optimizer,make_lr_scheduler
    from maskrcnn_benchmark.data.transforms import build_transforms
    from maskrcnn_benchmark.data.collate_batch import BatchCollator
    optimizer=make_optimizer(cfg,model,logging.getLogger("r16"),slow_heads=[],rl_factor=8.)
    scheduler=make_lr_scheduler(cfg,optimizer,logging.getLogger("r16"))
    train=dataset(cfg,"train");val=dataset(cfg,"val")
    assert len(val)==5000
    train.transforms=build_transforms(cfg,is_train=True)
    resume=out/"latest.pth";start=0;best=-1.
    if resume.exists():
        state=torch.load(str(resume),map_location="cpu")
        assert state["protocol"]==protocol
        model.load_state_dict(state["model"],strict=True)
        optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        start=state["iteration"];best=state["best"]
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state(state["cuda_rng"])
    sampler=DeterministicImages(len(train),start*8,steps*8,666)
    loader=DataLoader(train,batch_size=8,sampler=sampler,num_workers=0,collate_fn=BatchCollator(32),
                      generator=torch.Generator().manual_seed(667))
    started=time.monotonic()
    for step,(images,targets,_) in enumerate(loader,start+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses=model(images.to("cuda"),[t.to("cuda") for t in targets])
        loss=sum(losses.values())
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss; no update")
        if step==start+1 and args.task!="predcls":
            parameters=[p for n,p in model.named_parameters() if "context_layer.decoder_rnn.out_obj" in n]
            if not parameters:raise RuntimeError("Expected Motifs object decoder absent")
            g=torch.autograd.grad(losses["loss_refine_obj"],parameters,retain_graph=True,allow_unused=True)
            norm=sum(float(x.square().sum()) for x in g if x is not None)**.5
            if not np.isfinite(norm) or norm<=0:raise RuntimeError("Object gradient disconnected")
            atomic_json(out/"object_gradient.json",dict(norm=norm,loss_terms=list(losses)))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
        if not torch.isfinite(norm):raise RuntimeError("Nonfinite gradient; no update")
        optimizer.step()
        value=None
        if step%3000==0 or step==steps:
            save_torch(out/"pre_validation.pth",dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),iteration=step,best=best,protocol=protocol,
                pending_validation=True,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            result=evaluate(model,cfg,val,args.task,out/("validation_%06d"%step),2 if args.smoke else 0)
            value=result["R"]["100"]
            if value>best:
                best=value;save_torch(out/"best.pth",dict(model=model.state_dict(),iteration=step,protocol=protocol))
        scheduler.step(value,epoch=step)
        if step%200==0 or step==steps:
            save_torch(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                iteration=step,best=best,protocol=protocol,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
        elapsed=time.monotonic()-started
        row=dict(stage="training",task=args.task,iteration=step,total=steps,seconds=elapsed,
            eta_seconds=elapsed/(step-start)*(steps-step),loss={k:float(v.detach()) for k,v in losses.items()},
            lr=optimizer.param_groups[-1]["lr"],true_image_batch=8,memory_mb=torch.cuda.max_memory_allocated()/1024**2)
        atomic_json(out/"progress.json",row)
        if step<=start+2 or step%20==0:print(json.dumps(row),flush=True)
        if scheduler.stage_count>=3:break
    payload=torch.load(str(out/"best.pth"),map_location="cpu")
    model.load_state_dict(payload["model"],strict=True)
    result=evaluate(model,cfg,dataset(cfg,"test"),args.task,out/"test",2 if args.smoke else 0)
    atomic_json(out/"summary.json",dict(status="complete",selected_iteration=payload["iteration"],test=result,
        interpretation="Independent plain-Motifs reimplementation; TDE results cannot substitute"))


if __name__=="__main__":
    main()
