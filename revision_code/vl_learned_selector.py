"""One locked proposal-aware router, fitted on 200 and explored on 800 consumed images."""
import argparse
import fcntl
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources, selected_ids, paired_ratio
from repro_experiment import config, dataset, save_torch
from ve_experiment import lookup
from vb_native import OfficialMetrics, targets
from vc_protocol import summarize_rows, paired_gate, KS
from vk_gate_protocol import CHECKPOINT, PRIMARY_SHA, MANIFEST, CACHE, OUT as VK, ids
from vk_gate_native import configure_backend, tensor_digest
from r19_vk_decomposition import cached_prediction_check
from vl_sgdet_feasibility import RAW, all_candidates, emit_progress
from vl_router import action_features, fit_selector, select_action, FEATURE_NAMES
from vl_policy import improves_identity, preserves_relations


OUT=ROOT / "results/VL_proposal_router"
AUDIT=ROOT / "results/VL_sgdet_feasibility/development200"
K50=KS.index(50)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["fit","evaluate"],required=True)
    a=p.parse_args()
    ensure_storage(); torch.set_num_threads(2); configure_backend()
    guard=output_path(OUT / "execution.lock").open("w")
    fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
    train=selected_ids(ids(),200,"vl_sgdet_capacity_v1")
    evaluation=[i for i in ids() if i not in set(train)]
    if len(train)!=200 or len(evaluation)!=800 or set(train)&set(evaluation):
        raise RuntimeError("Image-disjoint development split failed")
    audit=read(AUDIT / "summary.json")
    if audit["status"]!="complete" or audit["images"]!=200 or audit["privileged_joint_identity_gain"]<.005:
        raise RuntimeError("No sufficient privileged capacity to justify routing exploration")
    if sha256(CHECKPOINT)!=PRIMARY_SHA: raise RuntimeError("Expert changed")
    protocol=dict(version="vl_proposal_router_v1",train_ids=train,evaluation_ids=evaluation,
        data_role="Both partitions are consumed V-K validation, reclassified as development. Image-disjoint fit/evaluation does not restore independence from earlier method design.",
        exploratory=True,confirmatory_claim=False,scientific_gate_pass=False,
        expert_checkpoint_sha256=PRIMARY_SHA,old_gate_sha256=sha256(MANIFEST),capacity_audit_sha256=sha256(AUDIT / "summary.json"),
        model="Two-stage linear logistic selector: nonzero postprocessing utility, then safe-gain vs damage conditional on nonzero utility",
        features=FEATURE_NAMES,training_labels="Single-update post-NMS identity gain and per-image/per-predicate R50 non-degradation; GT only in development supervision",
        effective_outcome="post-NMS correctness changes, or any image/per-predicate R50 decreases",
        safe_outcome="strict post-NMS identity gain with no image/per-predicate R50 loss",
        candidate_alpha=.5,prototype_min_objects=5,C=.1,max_iterations=2000,standardization="training-only",
        selection="At most one proposal per image; maximum P(effect)*(2*P(safe|effect)-1), P(safe|effect)>=0.6; ties by index; no tuning",
        safety_reason="One action matches the supervised single-update intervention and avoids untrained multi-update interactions",
        stop="One fitting/evaluation, no hyperparameter/threshold sweep, no test-set or extra seeds. Formal gate remains unchanged.",
        sources=sources(["vl_learned_selector.py","vl_router.py","vl_sgdet_feasibility.py","vl_policy.py",
                         "vb_native.py","vc_protocol.py","repro_experiment.py"]))
    reg=lock(OUT / "protocol.json",protocol)
    stage_out=OUT / a.stage
    if (stage_out / "summary.json").exists():
        existing=read(stage_out / "summary.json")
        if existing["protocol_sha256"]!=reg: raise RuntimeError("Result protocol mismatch")
        print("Already complete: "+str(stage_out),flush=True); return
    cfg=config("sgdet",stage_out)
    from pysgg.modeling.roi_heads.relation_head.inference import make_roi_relation_post_processor
    from pysgg.structures.bounding_box import BoxList
    post=make_roi_relation_post_processor(cfg)
    if post.use_gt_box or post.use_relness_ranking or post.BCE_loss or post.attribute_on:
        raise RuntimeError("Unsupported native postprocessor")
    ds=dataset(cfg,"val"); mapping=lookup(ds); metric=OfficialMetrics("sgdet")
    state=torch.load(str(CHECKPOINT),map_location="cpu")
    model=read(OUT / "selector.json") if a.stage=="evaluate" else None
    if model is not None and model["protocol_sha256"]!=reg: raise RuntimeError("Selector provenance failed")
    selected=train if a.stage=="fit" else evaluation
    records=[]; x_rows=[]; effects=[]; safe_rows=[]; start=time.monotonic(); trials=0
    for n,iid in enumerate(selected,1):
        rowpath=stage_out / "images" / (iid+".json")
        featurepath=stage_out / "features" / (iid+".pt")
        if rowpath.exists():
            record=read(rowpath)
            if record["protocol_sha256"]!=reg: raise RuntimeError("Resume provenance mismatch")
            records.append(record)
            if a.stage=="fit":
                cached=torch.load(str(featurepath),map_location="cpu")
                x_rows.extend(cached["x"].numpy());effects.extend(record["effects"]);safe_rows.extend(record["safe"])
            continue
        rawpath=RAW / (iid+".pt"); fpath=CACHE / "sgdet/gate/features" / (iid+".pt")
        raw=torch.load(str(rawpath),map_location="cpu")
        feature=torch.load(str(fpath),map_location="cpu")
        export=torch.load(str(CACHE / "sgdet/gate/export" / (iid+".pt")),map_location="cpu")
        if (feature["protocol_sha256"]!=protocol["old_gate_sha256"] or export["protocol_sha256"]!=feature["protocol_sha256"]
            or not torch.equal(raw["native_logits"],export["baseline"])
            or not torch.equal(feature["crop_boxes"],export["crop_boxes"])
            or tensor_digest(raw["pairs"])!=export["pair_sha256"]
            or tensor_digest(raw["relation_logits"])!=export["predicate_sha256"]):
            raise RuntimeError("Feature or native-output provenance mismatch")
        values={k:raw[k].cuda() for k in ["native_logits","proposal_boxes","boxes_per_cls","pairs","relation_logits"]}
        def replay(logits):
            box=BoxList(values["proposal_boxes"].clone(),raw["size"],"xyxy")
            box.add_field("boxes_per_cls",values["boxes_per_cls"].clone())
            return post.forward(([values["relation_logits"]],[logits]),[values["pairs"]],[box])[0]
        with torch.no_grad():
            base=replay(values["native_logits"])
            cached_prediction_check(base,VK / "sgdet/gate/native/predictions" / (iid+".npz"))
            candidate,eligible,_,legacy=all_candidates(feature["features"],raw["native_logits"],state)
            indices=eligible.nonzero(as_tuple=False).flatten().tolist()
            expert=F.linear(feature["features"],state["expert_head"]["weight"],state["expert_head"]["bias"]).softmax(-1)
            feature_rows=[]
            for index in indices:
                logits=values["native_logits"].clone(); logits[index]=candidate[index].cuda()
                trial=replay(logits);trials+=1
                feature_rows.append(action_features(index,legacy,raw["native_logits"],feature["features"],expert,state["centers"],base,trial))
            x=np.asarray(feature_rows,dtype=np.float64).reshape(-1,len(FEATURE_NAMES))
            # Evaluation policy has no access to GT or metric rows.
            chosen,choices=select_action(model,x,indices) if model else (None,[])
            updated=values["native_logits"].clone()
            if chosen is not None: updated[chosen]=candidate[chosen].cuda()
            prediction=replay(updated) if chosen is not None else base
            gt=ds.get_groundtruth(mapping[iid],evaluation=True)
            y=targets(raw,gt,"sgdet")
            if not np.array_equal(y,raw["target"].numpy()): raise RuntimeError("Target matching changed")
            base_row=metric.row(iid,base,gt,dict(logits=values["native_logits"]),y)
            if a.stage=="fit":
                audited=read(AUDIT / "images" / (iid+".json"))
                outcomes={r["proposal"]:r for r in audited["single_updates"]}
                if set(outcomes)!=set(indices): raise RuntimeError("Candidate support changed")
                ee=[];ss=[]
                for index in indices:
                    r=outcomes[index]
                    relation_damage=(r["R50_delta"] < -1e-12 or any(v is not None and v < -1e-12 for v in r["class_R50_delta"]))
                    ee.append(r["post_identity_delta"]!=0 or relation_damage)
                    ss.append(r["joint_single_utility"])
                effects.extend(ee); safe_rows.extend(ss); x_rows.extend(x)
                record=dict(image_id=iid,protocol_sha256=reg,candidates=len(indices),effects=ee,safe=ss,
                    input_sha256=dict(raw=sha256(rawpath),feature=sha256(fpath)),baseline=base_row)
            else:
                updated_row=metric.row(iid,prediction,gt,dict(logits=updated),y)
                record=dict(image_id=iid,protocol_sha256=reg,candidates=len(indices),chosen=chosen,
                    choices=choices,baseline=base_row,updated=updated_row,
                    input_sha256=dict(raw=sha256(rawpath),feature=sha256(fpath)),selector_sha256=sha256(OUT / "selector.json"))
            save_torch(featurepath,dict(image_id=iid,protocol_sha256=reg,x=torch.from_numpy(x),indices=indices))
            atomic_json(rowpath,record);records.append(record)
            for k in KS: metric.result["sgdet_recall"][k].clear()
        if n%10==0 or n==len(selected): emit_progress(stage_out,start,n,len(selected),trials,a.stage)
    if a.stage=="fit":
        learned=fit_selector(np.asarray(x_rows),effects,safe_rows)
        learned["protocol_sha256"]=reg
        lock(OUT / "selector.json",learned)
        summary=dict(status="complete",protocol_sha256=reg,images=len(records),actions=len(x_rows),
            effective_actions=int(np.sum(effects)),safe_actions=int(np.sum(safe_rows)),
            learned_selector_sha256=sha256(OUT / "selector.json"),sources=protocol["sources"])
    else:
        b=[r["baseline"] for r in records];u=[r["updated"] for r in records]
        numeric=paired_gate(b,u)
        summary=dict(status="complete",protocol_sha256=reg,images=len(records),selected_images=sum(r["chosen"] is not None for r in records),
            baseline=summarize_rows(b),updated=summarize_rows(u),
            exploratory_numeric_criteria=numeric,
            descriptive_identity=paired_ratio([r["post_nms_correct"] for r in u],[r["post_nms_correct"] for r in b],[r["positive_objects"] for r in b]),
            original_gate_pass=False,scientific_gate_pass=False,independent_confirmation=False,
            interpretation="Image-disjoint exploratory evaluation on 800 reused validation images; numerical criteria are diagnostic, not the registered 1000-image two-task gate",
            selector_sha256=sha256(OUT / "selector.json"))
    summary["elapsed_seconds"]=time.monotonic()-start
    atomic_json(stage_out / "summary.json",summary)
    atomic_json(stage_out / "progress.json",dict(status="complete",images=len(records),total=len(selected)))
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
