"""V-V: inner-selected conditional repair with native SGDet channel attribution."""
import argparse
import fcntl
import time
import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import read, lock
from repro_experiment import save_torch
from vp_experiment import rows_for, progress, finish
from vp_math import split_fold
from vq_experiment import image_inputs, assert_baseline
from vu_experiment import bundle_for
from vu_math import view_features
from vv_protocol import register, load, OUT, WEIGHTS
from vv_math import (PRIMARY, TRAINED, ARMS, SPEC, fit, choose_epoch, make_head,
                     predict, project_channels)


def records_for(protocol, bundle, ids):
    records = image_inputs(protocol, bundle, ids)
    mapping = {iid:i for i,iid in enumerate(bundle["image_ids"])}
    for rec in records:
        i = mapping[rec["iid"]]
        a,b = bundle["offsets"][i:i+2]
        rec["visual"] = view_features({k:bundle[k][a:b] for k in ["siglip_tight","siglip_context"]}, "siglip_dual")
    return records


def channel_diagnostics(native, updated, target):
    nbg, ubg = native.softmax(-1)[:,0].cpu().numpy(), updated.softmax(-1)[:,0].cpu().numpy()
    result = {}
    for group, keep in [("positive",target>0),("background",target==0),("ambiguous",target<0)]:
        result[group] = dict(count=int(keep.sum()),native_bg_sum=float(nbg[keep].sum()),updated_bg_sum=float(ubg[keep].sum()))
    result["foreground_argmax_changes"] = int((native[:,1:].argmax(-1)!=updated[:,1:].argmax(-1)).sum())
    result["proposals"] = len(native)
    return result


def score(records, head, name, post, metric, stage, label, no_op=False, channel=None):
    from vn_train_proposals import replay
    from vc_protocol import KS
    start = time.monotonic()
    rows, diagnostics = [], []
    if head is not None:
        head.eval()
    with torch.no_grad():
        for i,rec in enumerate(records):
            raw = rec["raw"]
            native = raw["native_logits"].float().cuda()
            updated = native if head is None else predict(rec["visual"].cuda(),native,head,name)
            if channel:
                updated = project_channels(native,updated,channel)
            if not torch.isfinite(updated).all():
                raise RuntimeError("Nonfinite repaired scores")
            if no_op and not torch.equal(updated,native):
                raise RuntimeError("Epoch-zero logits must be bit-exact")
            if name=="foreground_visual" or channel=="foreground":
                if not torch.allclose(updated.softmax(-1)[:,0],native.softmax(-1)[:,0],atol=2e-6,rtol=2e-5):
                    raise RuntimeError("Foreground-only projection changed BG mass")
            if channel=="background":
                if not torch.allclose(updated[:,1:].softmax(-1),native[:,1:].softmax(-1),atol=2e-6,rtol=2e-5):
                    raise RuntimeError("Background-only projection changed FG conditional")
            prediction = replay(post,raw,updated)
            row = metric.row(rec["iid"],prediction,rec["gt"],dict(logits=updated),rec["target"])
            if no_op or head is None:
                assert_baseline(row,rec["baseline"])
            rows.append(row)
            diagnostics.append(channel_diagnostics(native,updated,rec["target"]))
            for k in KS:
                metric.result["sgdet_recall"][k].clear()
            if (i+1)%50==0 or i+1==len(records):
                progress(stage,label,i+1,len(records),start)
    return rows,diagnostics


def select(protocol,reg,bundle,fold,split,name,post,metric):
    from vc_protocol import summarize_rows
    stage = OUT/("smoke" if protocol["smoke"] else "train3000")/("fold%d"%fold)
    path = stage/("selection_"+name+".json")
    if path.exists():
        info = read(path)
        if info["protocol_sha256"]!=reg:
            raise RuntimeError("Selector drift")
        return info
    ix = rows_for(bundle,split["fit"])
    records = records_for(protocol,bundle,split["inner_validation"])
    visual = view_features({k:bundle[k][ix] for k in ["siglip_tight","siglip_context"]},"siglip_dual")
    candidates = [0,1,2] if protocol["smoke"] else SPEC["epochs"]
    summaries = []
    start = time.monotonic()

    def callback(head,row):
        epoch = row["epoch"]
        if epoch in candidates:
            values,_ = score(records,head,name,post,metric,stage,"inner_%s_epoch%d"%(name,epoch),no_op=epoch==0)
            metrics = summarize_rows(values)
            summaries.append(dict(epoch=epoch,object=metrics["post_nms_object_top1"],
                R50=metrics["R"]["50"],mR50=metrics["mR"]["50"],metrics=metrics))
            atomic_json(stage/("candidate_progress_"+name+".json"),dict(protocol_sha256=reg,candidates=summaries))
        progress(stage,"fit_inner_"+name,epoch,max(candidates),start,detail=row)

    state,history = fit(visual,bundle["native_logits"][ix],bundle["labels"][ix],name,max(candidates),callback)
    change = max(float(t.abs().max()) for t in state.values())
    if not change>0:
        raise RuntimeError("Training did not update residual")
    info = dict(protocol_sha256=reg,candidates=summaries,history=history,
        trained_parameter_max_change=change,**choose_epoch(summaries))
    atomic_json(path,info)
    return info


def train(protocol,reg,lane,fold,bundle,split,post,metric):
    stage = OUT/lane/("fold%d"%fold)
    path = WEIGHTS/lane/("fold%d.pth"%fold)
    info_path = stage/"training_summary.json"
    if info_path.exists():
        info = read(info_path)
        if info["protocol_sha256"]!=reg or info["checkpoint_sha256"]!=sha256(path):
            raise RuntimeError("Fitted fold drift")
        return torch.load(str(path),map_location="cpu"),info
    selections = {name:select(protocol,reg,bundle,fold,split,name,post,metric) for name in TRAINED}
    ix = rows_for(bundle,split["training"])
    visual = view_features({k:bundle[k][ix] for k in ["siglip_tight","siglip_context"]},"siglip_dual")
    states,histories,epochs = {},{},{}
    for name in TRAINED:
        epochs[name] = selections[name]["selected_epoch"]
        start = time.monotonic()
        states[name],histories[name] = fit(visual,bundle["native_logits"][ix],bundle["labels"][ix],name,epochs[name],
            callback=lambda head,row:progress(stage,"refit_"+name,row["epoch"],epochs[name],start,detail=row))
    payload = dict(protocol_sha256=reg,bundle_sha256=protocol["bundle_sha256"],states=states,epochs=epochs,
        fold=fold,fit_ids=split["training"],held_ids=split["held"])
    save_torch(path,payload)
    info = dict(protocol_sha256=reg,checkpoint_sha256=sha256(path),epochs=epochs,histories=histories,
        fit_images=len(split["training"]),held_images=len(split["held"]),
        selection_parameter_changes={k:v["trained_parameter_max_change"] for k,v in selections.items()})
    atomic_json(info_path,info)
    return payload,info


def evaluate(protocol,reg,lane,fold,bundle,split,payload,info,post,metric):
    from vc_protocol import summarize_rows
    stage = OUT/lane/("fold%d"%fold)
    records = records_for(protocol,bundle,split["held"])
    arms,diagnostics = {},{}
    start = time.monotonic()
    for arm in ARMS:
        name = PRIMARY if arm.startswith("joint_") else arm
        channel = "foreground" if arm=="joint_fg_projection" else ("background" if arm=="joint_bg_projection" else None)
        if arm=="native":
            head=None;no_op=True
        else:
            head=make_head("cuda")
            head.load_state_dict(payload["states"][name],strict=True)
            no_op=payload["epochs"][name]==0
        arms[arm],diagnostics[arm] = score(records,head,name,post,metric,stage,"outer_"+arm,no_op=no_op,channel=channel)
    for i,iid in enumerate(split["held"]):
        atomic_json(stage/"images"/(iid+".json"),dict(protocol_sha256=reg,checkpoint_sha256=info["checkpoint_sha256"],
            image_id=iid,fold=fold,metrics={k:arms[k][i] for k in ARMS},
            channels={k:diagnostics[k][i] for k in ARMS}))
    finish(stage,dict(status="complete",protocol_sha256=reg,checkpoint_sha256=info["checkpoint_sha256"],
        fold=fold,images=len(records),primary=PRIMARY,epochs=payload["epochs"],
        arms={k:summarize_rows(v) for k,v in arms.items()},
        smoke_gradients_exercised=all(v>0 for v in info["selection_parameter_changes"].values()),
        validation_accessed=False,test_accessed=False,formal_gate_accepted=False,
        independent_confirmation=False,elapsed_seconds=time.monotonic()-start))


def summarize(protocol,reg,lane):
    from vc_protocol import summarize_rows
    from vo_math import policy_summary
    if protocol["smoke"]:
        raise ValueError("Smoke is not efficacy evidence")
    rows,choices = [],[]
    for fold in range(5):
        stage = OUT/lane/("fold%d"%fold)
        done = read(stage/"summary.json")
        digest = sha256(WEIGHTS/lane/("fold%d.pth"%fold))
        if done["protocol_sha256"]!=reg or done["checkpoint_sha256"]!=digest:
            raise RuntimeError("Completed fold changed")
        choices.append(dict(fold=fold,epochs=done["epochs"]))
        for iid in read(stage/"split.json")["held"]:
            row = read(stage/"images"/(iid+".json"))
            if row["protocol_sha256"]!=reg or row["checkpoint_sha256"]!=digest or row["fold"]!=fold or row["image_id"]!=iid:
                raise RuntimeError("Per-image provenance mismatch")
            rows.append(row)
    by_id = {r["image_id"]:r for r in rows}
    if len(rows)!=3000 or set(by_id)!=set(protocol["image_ids"]):
        raise RuntimeError("Incomplete or repeated outer images")
    rows = [by_id[i] for i in protocol["image_ids"]]
    folds = np.array([r["fold"] for r in rows])
    comparisons = {}
    for reference,arm in [("native",a) for a in ARMS[1:]]+[(a,PRIMARY) for a in TRAINED[1:]]:
        base = [r["metrics"][reference] for r in rows]
        trial = [r["metrics"][arm] for r in rows]
        pos = np.array([r["positive_objects"] for r in base])
        support = np.array([r["class_recalls"][4] for r in base],dtype=float)
        other = np.array([r["class_recalls"][4] for r in trial],dtype=float)
        if pos.tolist()!=[r["positive_objects"] for r in trial] or not np.array_equal(np.isfinite(support),np.isfinite(other)):
            raise RuntimeError("Denominator/support drift")
        gain = np.array([u["post_nms_correct"]-b["post_nms_correct"] for b,u in zip(base,trial)])
        dr = np.array([u["recalls"][4]-b["recalls"][4] for b,u in zip(base,trial)])
        stats = policy_summary(np.arange(3000),np.arange(3000),gain,dr,np.nan_to_num(other-support),np.isfinite(support),pos,folds)
        stats["evaluated_images"] = stats.pop("selected_images")
        comparisons[arm+"_versus_"+reference] = stats
    channels = {}
    for arm in ARMS:
        channels[arm] = {key:sum(r["channels"][arm][key] for r in rows) for key in ["proposals","foreground_argmax_changes"]}
        for group in ["positive","background","ambiguous"]:
            values = {key:sum(r["channels"][arm][group][key] for r in rows) for key in ["count","native_bg_sum","updated_bg_sum"]}
            values.update(native_bg_mean=values["native_bg_sum"]/values["count"] if values["count"] else None,
                updated_bg_mean=values["updated_bg_sum"]/values["count"] if values["count"] else None)
            channels[arm][group] = values
    checks = comparisons[PRIMARY+"_versus_native"]["descriptive_original_numerical_checks"]
    finish(OUT/lane,dict(status="complete",protocol_sha256=reg,images=3000,primary=PRIMARY,choices=choices,
        arms={k:summarize_rows([r["metrics"][k] for r in rows]) for k in ARMS},comparisons=comparisons,
        channel_diagnostics=channels,primary_training_screen_satisfied=all(checks.values()),
        formal_gate_accepted=False,independent_confirmation=False,native_sgg_trained_on_these_images=True,
        reused_train_images=True,validation_accessed=False,test_accessed=False,
        next_step="Stop. No control promotion, extra seeds, original validation/test or automatic tuning."))


def main():
    from vk_gate_native import configure_backend
    from vn_train_proposals import make_post
    from vb_native import OfficialMetrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["register","fold","summarize"],required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--fold",type=int,choices=range(5))
    args = parser.parse_args()
    ensure_storage();torch.set_num_threads(2);configure_backend()
    if args.stage=="register":
        _,reg,lane=register(args.smoke)
        print("[registered]",lane,reg,flush=True)
        return
    protocol,reg,lane = load(args.smoke)
    name = "fold%d"%args.fold if args.fold is not None else args.stage
    guard=output_path(OUT/lane/(name+".lock")).open("w")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=="summarize":
        summarize(protocol,reg,lane)
        return
    if args.fold is None or (args.smoke and args.fold!=0):
        raise ValueError("Invalid fold")
    stage=OUT/lane/name
    if (stage/"summary.json").exists():
        value=read(stage/"summary.json")
        if value["protocol_sha256"]!=reg or value["checkpoint_sha256"]!=sha256(WEIGHTS/lane/(name+".pth")):
            raise RuntimeError("Completed fold drift")
        return
    bundle,digest=bundle_for(protocol["vu_protocol_sha256"],lane)
    if digest!=protocol["bundle_sha256"] or bundle["image_ids"]!=protocol["image_ids"]:
        raise RuntimeError("Source bundle changed")
    split=split_fold(protocol["image_ids"],protocol["folds"],args.fold)
    lock(stage/"split.json",dict(protocol_sha256=reg,**split))
    _,post=make_post(stage/"runtime");metric=OfficialMetrics("sgdet")
    payload,info=train(protocol,reg,lane,args.fold,bundle,split,post,metric)
    evaluate(protocol,reg,lane,args.fold,bundle,split,payload,info,post,metric)


if __name__=="__main__":
    main()
