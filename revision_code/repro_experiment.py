"""Task-specific repair: supervised object head, real resume, isolated evaluation."""
import argparse
from contextlib import nullcontext
from functools import reduce
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from common import ROOT, OLD, atomic_json, output_path, sha256, ensure_storage
from repro_protocol import RUN, SPEC, verify, immutable
from native_runtime import assets, image_id, infer

KS = [1, 5, 10, 20, 50, 100]


def dataset(cfg, split, all_training=False):
    from pysgg.config.paths_catalog import DatasetCatalog
    from pysgg.data.datasets.visual_genome import VGDataset
    key = getattr(cfg.DATASETS, {"train":"TRAIN", "val":"VAL", "test":"TEST"}[split])[0]
    kwargs = DatasetCatalog.get(key, cfg)["args"]
    # Upstream dynamically reloads the catalog during model construction.
    for field in ("img_dir", "roidb_file", "dict_file", "image_file"):
        path = Path(kwargs[field])
        if not path.is_absolute():
            path = OLD / "external/official_repos/PySGG" / path
        kwargs[field] = str(path.resolve())
    kwargs.update(filter_duplicate_rels=False, filter_non_overlap=False)
    if all_training:
        kwargs["num_val_im"] = 0
    ds = VGDataset(**kwargs)
    assert len(ds.ind_to_classes) == 151 and len(ds.ind_to_predicates) == 51
    return ds


class DeterministicImages(Sampler):
    """Image order resumes at an optimizer boundary without replaying old batches."""
    def __init__(self, n, start, stop, seed=666, rank=0, world_size=1):
        self.n, self.start, self.stop, self.seed = n, start, stop, seed
        self.rank, self.world_size = rank, world_size

    def __len__(self):
        return len(range(self.start+self.rank, self.stop, self.world_size))

    def __iter__(self):
        epoch = -1; order = None
        for offset in range(self.start+self.rank, self.stop, self.world_size):
            current, position = divmod(offset, self.n)
            if current != epoch:
                epoch = current
                order = np.random.default_rng(self.seed + epoch).permutation(self.n)
            yield int(order[position])


def save_torch(path, value):
    path = output_path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, str(temp)); temp.replace(path)


def config(task, out, training=False):
    from pysgg.config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(OLD / "configs/pysgg_vg_tritask" / ("transformer_" + task + ".yaml")))
    if task == "sgcls" and not training:
        saved = assets("transformer", task)[1].parent / "config.yml"
        cfg.merge_from_file(str(saved))
    cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE = True
    cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS = True
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = task == "sgcls"
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    cfg.MODEL.ROI_RELATION_HEAD.FIX_FEATURE = False
    cfg.MODEL.WEIGHT = ""
    cfg.MODEL.PRETRAINED_DETECTOR_CKPT = str(OLD / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth")
    cfg.PATHS_CATALOG = str(OLD / "external/official_repos/PySGG/pysgg/config/paths_catalog.py")
    cfg.GLOVE_DIR = str(OLD / "data/derived/glove")
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.TEST.ALLOW_LOAD_FROM_CACHE = False
    cfg.TEST.RELATION.SYNC_GATHER = False
    cfg.SOLVER.BASE_LR = .001
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.MAX_ITER = 16000
    cfg.SOLVER.STEPS = (10000, 16000)
    cfg.SOLVER.WEIGHT_DECAY = .0001
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.
    cfg.SOLVER.BIAS_LR_FACTOR = 1
    cfg.SOLVER.MOMENTUM = .9
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = .1
    cfg.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"
    cfg.SOLVER.PRE_VAL = False
    cfg.SOLVER.TO_VAL = False
    cfg.DTYPE = "float32"
    cfg.OUTPUT_DIR = str(out / "runtime")
    output_path(out / "runtime/.placeholder").parent.mkdir(parents=True, exist_ok=True)
    cfg.freeze()
    # Some upstream statistics helpers use the module-global cfg and cwd.
    os.chdir(str(out / "runtime"))
    return cfg


def build(task, out, weights=None, fresh=False):
    from pysgg.modeling.detector import build_detection_model
    from pysgg.utils.checkpoint import DetectronCheckpointer
    from pysgg.data.transforms import build_transforms
    cfg = config(task, out, training=fresh or weights is not None)
    # Reuse exactly the same ontology statistics, verified by their class names.
    stat_name = "VG_stanford_filtered_with_attribute_train_statistics.cache"
    source = assets("transformer", "sgdet")[1].parent / stat_name
    stats = torch.load(str(source), map_location="cpu")
    vocab = json.loads((OLD / "data/vg/v1.4/VG-SGG-dicts.json").read_text())
    names = ["__background__"] + [k for k, v in sorted(vocab["label_to_idx"].items(), key=lambda p: p[1]) if v > 0]
    predicates = ["__background__"] + [k for k, v in sorted(vocab["predicate_to_idx"].items(), key=lambda p: p[1]) if v > 0]
    if list(stats["obj_classes"]) != names or list(stats["rel_classes"]) != predicates:
        raise RuntimeError("Statistics ontology mismatch")
    dest = Path(cfg.OUTPUT_DIR) / stat_name
    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    if not dest.exists() and rank == 0:
        save_torch(dest, stats)
    if distributed:
        torch.distributed.barrier()
    if sha256(dest) != sha256(source):
        existing = torch.load(str(dest), map_location="cpu")
        if not torch.equal(existing["fg_matrix"], stats["fg_matrix"]):
            raise RuntimeError("Statistics cache drift")
    model = build_detection_model(cfg)
    new_union_parameters = []
    if fresh:
        source_weight = Path(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
        marker = json.loads(source_weight.with_suffix(".pth.json").read_text())
        if sha256(source_weight) != marker["output_sha256"]:
            raise RuntimeError("Detector SHA mismatch")
        payload = torch.load(str(source_weight), map_location="cpu")
        detector = payload["model"]
        current = model.state_dict()
        detector_keys = {k for k in current if not k.startswith("roi_heads.relation.")}
        if not detector_keys <= set(detector) or any(current[k].shape != detector[k].shape for k in detector_keys):
            raise RuntimeError("Detector architecture mismatch")
        mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                   "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}
        DetectronCheckpointer(cfg, model)._load_model(payload, mapping)
        # Verify actual assigned weights, not merely successful permissive loading.
        for name in detector_keys:
            if not torch.equal(model.state_dict()[name], detector[name]):
                raise RuntimeError("Detector initialization failed: " + name)
        for dst, src in mapping.items():
            for name, value in model.state_dict().items():
                if name.startswith(dst + "."):
                    counterpart = src + name[len(dst):]
                    if counterpart not in detector and name.startswith(
                            "roi_heads.relation.union_feature_extractor.feature_extractor.pooler.reduce_channel."):
                        # The relation-only all-level pooling projection has no detector counterpart.
                        new_union_parameters.append(name)
                        continue
                    if counterpart not in detector or not torch.equal(value, detector[counterpart]):
                        raise RuntimeError("ROI initialization failed: " + name)
    else:
        source_weight = Path(weights) if weights else assets("transformer", task)[1]
        if weights is None and sha256(source_weight) != assets("transformer", task)[2]:
            raise RuntimeError("Existing checkpoint SHA mismatch")
        payload = torch.load(str(source_weight), map_location="cpu")
        state = {k.removeprefix("module.") if hasattr(k, "removeprefix") else (k[7:] if k.startswith("module.") else k): v
                 for k, v in payload["model"].items()}
        model.load_state_dict(state, strict=True)
    provenance = dict(source_checkpoint=str(source_weight), checkpoint_sha256=sha256(source_weight),
                      flags=SPEC["flags"], config=cfg.dump(), statistics_source=str(source), statistics_sha256=sha256(source),
                      new_union_projection_parameters=new_union_parameters)
    if rank == 0:
        immutable(out / "model_provenance.json", provenance)
    return model.cuda(), cfg, build_transforms(cfg, is_train=False), provenance


def train(args):
    from pysgg.data.transforms import build_transforms
    from pysgg.data.collate_batch import BatchCollator
    from pysgg.solver import make_optimizer, make_lr_scheduler
    task = args.task; smoke = args.smoke_steps > 0
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if world > 1 else 0
    out = RUN / task / ("training_smoke" if smoke else "training")
    seed = SPEC["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, cfg, _, provenance = build(task, out, fresh=True)
    for module in [model.backbone, model.rpn, model.roi_heads.box]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if not model.roi_heads.relation.object_cls_refine or not model.roi_heads.relation.pass_obj_recls_loss:
        raise RuntimeError("Corrected classifier training/output flags not enabled")
    ds = dataset(cfg, "train")
    # Match native training: one sampled predicate for duplicate endpoint pairs.
    ds.filter_duplicate_rels = True
    ds.transforms = build_transforms(cfg, is_train=True)
    logger = logging.getLogger("reproduction")
    optimizer = make_optimizer(cfg, model, logger, slow_heads=[], rl_factor=16.)
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    steps = args.smoke_steps if smoke else SPEC["training"]["optimizer_steps"]
    global_batch = SPEC["training"]["global_image_microsteps"]
    if global_batch % world:
        raise RuntimeError("World size must divide effective batch")
    accumulation = global_batch // world
    registration_sha = sha256(ROOT / "manifests/R9_reproduction.json")
    contract = dict(task=task, registration_sha256=registration_sha, optimizer_steps=steps,
                    image_accumulation=accumulation, world_size=world, train_images=len(ds), initialization=provenance["checkpoint_sha256"], smoke=smoke)
    if rank == 0:
        immutable(out / "protocol.json", contract)
    resume = out / "latest.pth"; start = 0
    if resume.exists():
        state = torch.load(str(resume), map_location="cpu")
        if state["protocol"] != contract:
            raise RuntimeError("Resume protocol mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start = state["iteration"]
        rng = state["rng_by_rank"][rank]
        random.setstate(rng["random_state"]); np.random.set_state(rng["numpy_state"])
        torch.set_rng_state(rng["torch_state"]); torch.cuda.set_rng_state(rng["cuda_state"])
    sampler = DeterministicImages(len(ds), start * global_batch, steps * global_batch, seed, rank, world)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0,
                        collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY),
                        generator=torch.Generator().manual_seed(seed + 1))
    model.train(); optimizer.zero_grad(set_to_none=True)
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(core,device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,find_unused_parameters=True)
    started = time.monotonic(); totals = {}; gradient_checked = False
    for micro, (images, targets, _) in enumerate(loader, start * accumulation + 1):
        if any(len(t) == 0 for t in targets):
            raise RuntimeError("Empty object target; no silent image skipping")
        images = images.to("cuda"); targets = [t.to("cuda") for t in targets]
        context = model.no_sync() if world > 1 and micro % accumulation else nullcontext()
        with context:
            losses = model(images, targets, logger=None)
            if "loss_refine_obj" not in losses or not losses["loss_refine_obj"].requires_grad:
                raise RuntimeError("Object classification objective is missing/disconnected")
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; last checkpoint retained")
            (loss / accumulation).backward()
        if not gradient_checked:
            parameters = [p for n,p in core.named_parameters() if "context_layer.out_obj" in n and p.requires_grad]
            norm = sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** .5
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError("Object-head gradient is not finite and nonzero")
            if rank == 0:
                atomic_json(out / "gradient_check.json", dict(status="complete", object_gradient_norm=norm,
                            loss_terms=list(losses), gradient_source="joint supervised loss; explicit object loss present"))
            gradient_checked = True
        for k, value in losses.items():
            totals[k] = totals.get(k, 0.) + float(value.detach()) / accumulation
        step, remainder = divmod(micro, accumulation)
        if remainder:
            continue
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient; no optimizer update")
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        elapsed = time.monotonic() - started
        value = dict(stage="training", task=task, iteration=step, total=steps,
                     seconds=elapsed, completed_this_session=step-start,
                     eta_seconds=elapsed / max(step-start, 1) * (steps-step),
                     loss=totals, optimizer_lr=optimizer.param_groups[-1]["lr"],
                     images_seen=step*global_batch, image_batch_per_gpu=1, accumulation_per_gpu=accumulation, world_size=world,
                     memory_mb=torch.cuda.max_memory_allocated()/1024**2)
        if rank == 0:
            atomic_json(out / "progress.json", value)
        if rank == 0 and (step % 10 == 0 or step <= start+2 or step == steps):
            print(json.dumps(value), flush=True)
        totals = {}
        if step % 200 == 0 or step == steps:
            rng = dict(random_state=random.getstate(),numpy_state=np.random.get_state(),
                       torch_state=torch.get_rng_state(),cuda_state=torch.cuda.get_rng_state())
            rngs = [None] * world
            if world > 1:
                torch.distributed.all_gather_object(rngs, rng)
            else:
                rngs = [rng]
            if rank == 0:
                state = dict(model=core.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             iteration=step, protocol=contract, rng_by_rank=rngs)
                save_torch(resume, state)
                if step % 1000 == 0:
                    save_torch(out / ("model_%07d.pth" % step), dict(model=core.state_dict(), iteration=step, protocol=contract))
            if world > 1:
                torch.distributed.barrier()
    final = out / "model_final.pth"
    if rank == 0:
        save_torch(final, dict(model=core.state_dict(), iteration=steps, protocol=contract))
        atomic_json(out / "summary.json", dict(status="complete", training_completed=True, smoke=smoke,
                    iterations=steps, checkpoint=str(final), checkpoint_sha256=sha256(final),
                    reproduction_status="not_yet_evaluated", elapsed_seconds=time.monotonic()-started))


def recall_row(pred, gt, task, seen):
    from pysgg.data.datasets.evaluation.vg.sgg_eval import SGRecall
    pred = pred.resize(gt.size).to("cpu")
    rel = gt.get_field("relation_tuple").cpu().numpy().astype(int)
    classes = gt.get_field("labels").cpu().numpy()
    local = dict(pred_rel_inds=pred.get_field("rel_pair_idxs").numpy()[:100],
                 rel_scores=pred.get_field("pred_rel_scores").numpy()[:100],
                 gt_rels=rel, gt_classes=classes, gt_boxes=gt.bbox.cpu().numpy(),
                 pred_boxes=pred.bbox.numpy(), pred_classes=pred.get_field("pred_labels").numpy(),
                 obj_scores=pred.get_field("pred_scores").numpy())
    collector = {}; recall = SGRecall(collector); recall.register_container(task)
    collector[task+"_recall"] = {k: [] for k in KS}
    matches = recall.calculate_recall({"iou_thres": .5}, local, task)["pred_to_gt"] if len(local["pred_rel_inds"]) else []
    count = np.bincount(rel[:,2], minlength=51)[1:]
    zero = {i for i, (s,o,p) in enumerate(rel) if (int(classes[s]), int(p), int(classes[o])) not in seen}
    rr, mr, zr = [], [], []
    for k in KS:
        hit = reduce(np.union1d, matches[:k], np.array([], dtype=int)).astype(int)
        correct = np.bincount(rel[hit,2], minlength=51)[1:]
        rr.append(len(hit)/len(rel)); mr.append([float(h/c) if c else None for h,c in zip(correct,count)])
        zr.append(len(set(hit)&zero)/len(zero) if zero else None)
    return dict(R=rr, class_recall=mr, zR=zr, zero_shot_relations=len(zero), relations=len(rel))


def evaluate(args):
    out = RUN / args.task / ("eval_" + args.variant) / (args.split + ("_smoke" if args.limit else ""))
    weight = RUN / args.task / "training/model_final.pth" if args.variant == "retrained" else None
    model, cfg, transform, provenance = build(args.task, out, weights=weight)
    model.eval()
    ds = dataset(cfg, args.split)
    # Recover seen triplets directly from native training annotations, no held-out labels.
    train_ds = dataset(cfg, "train", all_training=True)
    seen = set()
    for labels, triples in zip(train_ds.gt_classes, train_ds.relationships):
        for s,o,p in triples:
            seen.add((int(labels[s]),int(p),int(labels[o])))
    del train_ds
    ids = [image_id(ds,i) for i in range(len(ds))]
    if args.limit:
        ids = ids[:args.limit]
    pf = out / "protocol.json"
    immutable(pf, dict(provenance=provenance, image_ids=ids, split=args.split, task=args.task,
                       seen_triplets=len(seen), registration_sha256=sha256(ROOT / "manifests/R9_reproduction.json")))
    rows = []; start = time.monotonic(); computed = 0
    protocol_sha = sha256(pf)
    from export_pysgg_vg_task import convert_prediction
    captured = {}
    def post_hook(module, inputs):
        captured["logits"] = inputs[0][1][0].detach().cpu().numpy()
    handle = model.roi_heads.relation.post_processor.register_forward_pre_hook(post_hook)
    for i,iid in enumerate(ids):
        path = out / "images" / (iid + ".json")
        prediction = out / "predictions" / (iid + ".npz")
        if path.exists() and prediction.exists():
            row = json.loads(path.read_text())
            if row["protocol_sha256"] != protocol_sha:
                raise RuntimeError("Stale evaluation image")
            rows.append(row); continue
        pred, _ = infer(model,cfg,transform,ds,i)
        gt = ds.get_groundtruth(i,evaluation=True)
        row = recall_row(pred,gt,args.task,seen)
        row.update(image_id=iid,protocol_sha256=protocol_sha)
        cache = convert_prediction(pred.to("cpu"))
        cache["dense_postprocessor_object_logits"] = captured["logits"]
        temp = output_path(prediction.with_suffix(".tmp"))
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **cache)
        temp.replace(prediction); atomic_json(path,row); rows.append(row)
        computed += 1
        if (i+1)%25==0 or computed==1 or i+1==len(ids):
            value = dict(stage="evaluation", images=i+1, total=len(ids), seconds=time.monotonic()-start,
                         eta_seconds=(time.monotonic()-start)/computed*(len(ids)-i-1),
                         completed_this_session=computed)
            atomic_json(out/"progress.json",value); print(json.dumps(value),flush=True)
    handle.remove()
    r = np.array([x["R"] for x in rows]).mean(0)
    perclass = np.array([x["class_recall"] for x in rows],dtype=float)
    count = np.isfinite(perclass).sum(0)
    mr = np.divide(np.nansum(perclass,axis=0),count,out=np.zeros_like(count,dtype=float),where=count>0).mean(1)
    zeros = np.array([x["zR"] for x in rows],dtype=float)
    zcount = np.isfinite(zeros).sum(0)
    zr = [float(np.nansum(zeros[:,j])/zcount[j]) if zcount[j] else None for j in range(len(KS))]
    manifest = assets("transformer",args.task)[0]
    checks = {}
    for name,ref in manifest["reference_metrics"].items():
        if name.lower().startswith(args.task+"/"):
            metric = name.split("/")[1]; index = KS.index(int(metric.split("@")[1]))
            actual = float(mr[index] if metric.startswith("mR") else r[index])
            checks[name] = dict(actual=actual,reference=ref,delta=actual-ref,passes=abs(actual-ref)<=.02)
    full_test = args.split=="test" and len(rows)==26446 and not args.limit
    if not checks:
        raise RuntimeError("No matching reference metrics; refuse vacuous reproduction pass")
    passed = all(x["passes"] for x in checks.values()) if full_test else None
    atomic_json(out/"summary.json",dict(status="complete",images=len(rows),task=args.task,split=args.split,
                R=dict(zip(map(str,KS),r.tolist())),mR=dict(zip(map(str,KS),mr.tolist())),zR=dict(zip(map(str,KS),zr)),
                zero_shot_relations=sum(x["zero_shot_relations"] for x in rows),reference_checks=checks,
                full_test=full_test,reproduction_passed=passed,
                reproduction_status=("passed_registered_tolerance" if passed else "reference_gap") if full_test else "not_a_full_test",
                protocol_sha256=sha256(pf)))


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["train","evaluate"])
    p.add_argument("--task",choices=["sgcls","sgdet"],required=True)
    p.add_argument("--variant",choices=["corrected","retrained"],default="corrected")
    p.add_argument("--split",choices=["val","test"],default="test")
    p.add_argument("--limit",type=int,default=0);p.add_argument("--smoke_steps",type=int,default=0)
    p.add_argument("--local_rank",type=int,default=0)
    args=p.parse_args();ensure_storage();verify();torch.set_num_threads(4)
    if int(os.environ.get("WORLD_SIZE","1"))>1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",args.local_rank)))
        torch.distributed.init_process_group("nccl",init_method="env://")
    if args.stage=="train": train(args)
    else: evaluate(args)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__=="__main__": main()
