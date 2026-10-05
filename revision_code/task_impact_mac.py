"""Task-conditioned selection over supplied regions, not physical robot execution."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import vj_mac as util


RELATIONS = ["left", "right", "above", "below"]
ARMS = ["hard_geometry", "soft_identity", "soft_frequency", "soft_geometry",
        "vk_oof_geometry", "mismatched_geometry", "oracle_identity_geometry"]
SPEC = dict(version="T1_supplied_region_task_impact_v1", relations=RELATIONS,
    normalized_center_gap=.05, maximum_queries_per_stratum=2,
    calibration_images=500, screening_images=500, negative_min_train_class_support=50,
    calibration_false_answer_max=.05, strict_threshold=True, bootstrap_samples=2000,
    seed=17029, image_salt="T1_task_calibration:", query_salt="T1_query:",
    arms=ARMS, primary_contrast=["soft_geometry", "soft_frequency"],
    secondary_contrast=["vk_oof_geometry", "soft_geometry"],
    metric="positive ordered-pair success, and negative false-answer rate; image-cluster bootstrap",
    scope="GT-supplied region universe, image-plane center relations; NOT physical affordances or embodied execution",
    identity="frozen native logits and VK image-OOF probabilities; no full-calibration VK inference on its fitting pool",
    screening_reuse="old500 explored calibration images; pilot only, not independent confirmation",
    gate_policy="native VK criteria unchanged; task screening cannot certify SGG mitigation",
    stop="one protocol; no tuning/arm switching after screening; independent confirmation requires native export")


def hash_key(text):
    return hashlib.sha256(text.encode()).digest()


def geometry(boxes, size):
    b = np.asarray(boxes, dtype=np.float64)
    if (b.ndim != 2 or b.shape[1] != 4 or len(size) != 2 or min(size) <= 0
            or not np.isfinite(b).all() or (b[:, 2:] < b[:, :2]).any()):
        raise ValueError("Invalid supplied box geometry")
    c = (b[:, :2] + b[:, 2:]) / (2 * np.asarray(size))
    dx = c[None, :, 0] - c[:, None, 0]
    dy = c[None, :, 1] - c[:, None, 1]
    gap = SPEC["normalized_center_gap"]
    out = np.stack([dx > gap, dx < -gap, dy > gap, dy < -gap])
    for r in out:
        np.fill_diagonal(r, False)
    return out


def possible_pairs(labels, target, anchor, relation):
    labels = np.asarray(labels)
    return np.argwhere((labels[:, None] == target) & (labels[None, :] == anchor) & relation).tolist()


def query_id(image_id, query):
    return "%s:%d:%d:%s" % (image_id, query["target"], query["anchor"], query["relation"])


def make_queries(item, supported):
    labels = np.asarray(item["labels"], dtype=np.int64)
    rel = geometry(item["boxes"], item["size"])
    classes = sorted(set(labels.tolist()))
    buckets = dict(positive=[], category_absent=[], relation_absent=[])
    for t in classes:
        for a in classes:
            if t == a:
                continue
            for r, name in enumerate(RELATIONS):
                q = dict(target=t, anchor=a, relation=name)
                answers = possible_pairs(labels, t, a, rel[r])
                bucket = "positive" if answers else "relation_absent"
                buckets[bucket].append(dict(query=q, answers=answers, stratum=bucket))
    absent = sorted(set(supported) - set(classes))
    # Queries depend on annotations and training support, never model predictions.
    for t in absent:
        for a in classes:
            for name in RELATIONS:
                buckets["category_absent"].append(dict(query=dict(target=t, anchor=a, relation=name),
                                                       answers=[], stratum="category_absent"))
    records = []
    coverage = {}
    for stratum, values in buckets.items():
        coverage[stratum] = len(values)
        chosen = sorted(values, key=lambda v: hash_key(SPEC["query_salt"] + query_id(item["image_id"], v["query"])))[:SPEC["maximum_queries_per_stratum"]]
        for value in chosen:
            value.update(image_id=item["image_id"], id=query_id(item["image_id"], value["query"]))
        records += chosen
    return records, coverage


def pair_score(query, probabilities, spatial, frequencies, mode):
    """The inference interface intentionally has no labels, answers or stratum."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 151 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid VG-150 probability input")
    t, a = int(query["target"]), int(query["anchor"])
    if not (1 <= t <= 150 and 1 <= a <= 150 and t != a):
        raise ValueError("Invalid query vocabulary")
    ri = RELATIONS.index(query["relation"])
    n = len(p)
    if n < 2:
        return None, 0.
    score = p[:, t, None] * p[None, :, a]
    if mode in ["hard_geometry", "soft_geometry"]:
        score = score * spatial[ri]
    if mode == "hard_geometry":
        hard = p[:, 1:].argmax(-1) + 1
        score *= (hard[:, None] == t) & (hard[None, :] == a)
    elif mode == "soft_frequency":
        score *= frequencies[ri, t-1, a-1]
    elif mode not in ["soft_geometry", "soft_identity"]:
        raise ValueError("Unknown decision rule")
    np.fill_diagonal(score, 0.)
    index = int(score.argmax()); i, j = divmod(index, n)
    value = float(score[i, j])
    return ([i, j], value) if value > 0 else (None, 0.)


def fit_threshold(negative_scores):
    scores = np.sort(np.asarray(negative_scores, dtype=float))
    if len(scores) < 100 or not np.isfinite(scores).all():
        raise ValueError("Insufficient calibration negatives")
    permitted = int(np.floor(SPEC["calibration_false_answer_max"] * len(scores)))
    threshold = float(scores[len(scores) - permitted - 1])
    empirical = float((scores > threshold).mean())
    if empirical > SPEC["calibration_false_answer_max"] + 1e-12:
        raise RuntimeError("Calibration rejection rule failed")
    return dict(threshold=threshold, negatives=len(scores), false_answers=int((scores > threshold).sum()),
                empirical_false_answer_rate=empirical, decision="strict score > threshold")


def returned_pair(pair, score, threshold):
    return pair if pair is not None and score > threshold else None


def decision_record(task, pair, labels, spatial):
    positive = task["stratum"] == "positive"
    answer = {tuple(x) for x in task["answers"]}
    full = pair is not None and tuple(pair) in answer
    target = pair is not None and pair[0] in {x[0] for x in answer}
    geometry_ok = pair is not None and bool(spatial[RELATIONS.index(task["query"]["relation"]), pair[0], pair[1]])
    label_ok = pair is not None and (labels[pair[0]] == task["query"]["target"] and labels[pair[1]] == task["query"]["anchor"])
    return dict(pair=pair, emitted=pair is not None, correct=full if positive else pair is None,
        positive_pair_success=bool(full), positive_target_success=bool(target),
        false_answer=bool(not positive and pair is not None),
        wrong_identity_when_emitted=bool(pair is not None and not label_ok),
        wrong_geometry_when_emitted=bool(pair is not None and not geometry_ok))


COUNT_KEYS = ["positive", "positive_correct", "positive_target_correct", "negative", "false_answer",
              "category_absent", "category_false_answer", "relation_absent", "relation_false_answer", "emitted", "correct_emitted"]


def metric(values):
    v = dict(zip(COUNT_KEYS, np.asarray(values).tolist()))
    rate = lambda a,b: v[a]/v[b] if v[b] else None
    pos, far = rate("positive_correct","positive"), rate("false_answer","negative")
    return dict(**{k:int(x) for k,x in v.items()}, positive_pair_success=pos,
        positive_target_success=rate("positive_target_correct","positive"), negative_false_answer_rate=far,
        category_absent_false_answer_rate=rate("category_false_answer","category_absent"),
        relation_absent_false_answer_rate=rate("relation_false_answer","relation_absent"),
        balanced_success=(pos+1-far)/2 if pos is not None and far is not None else None,
        emitted_answer_precision=rate("correct_emitted","emitted"),
        answer_coverage=v["emitted"]/(v["positive"]+v["negative"]) if v["positive"]+v["negative"] else None)


def accumulate(image_ids, rows):
    index = {iid:i for i,iid in enumerate(image_ids)}
    counts = {name:np.zeros((len(image_ids),len(COUNT_KEYS)),dtype=np.int64) for name in ARMS}
    for row in rows:
        positive = row["stratum"] == "positive"
        category = row["stratum"] == "category_absent"
        relational = row["stratum"] == "relation_absent"
        for name, value in row["decisions"].items():
            vals = [positive, positive and value["positive_pair_success"], positive and value["positive_target_success"],
                not positive, value["false_answer"], category, category and value["false_answer"],
                relational, relational and value["false_answer"], value["emitted"], positive and value["positive_pair_success"]]
            counts[name][index[row["image_id"]]] += np.asarray(vals,dtype=np.int64)
    return counts


def paired_intervals(counts):
    rng = np.random.default_rng(SPEC["seed"]); n = len(next(iter(counts.values())))
    estimates = {name:[] for name in counts}
    differences = {"geometry_minus_frequency":[], "vk_minus_native_geometry":[], "geometry_minus_mismatched":[]}
    comparisons = dict(geometry_minus_frequency=("soft_geometry","soft_frequency"),
        vk_minus_native_geometry=("vk_oof_geometry","soft_geometry"),
        geometry_minus_mismatched=("soft_geometry","mismatched_geometry"))
    for _ in range(SPEC["bootstrap_samples"]):
        ix = rng.integers(0,n,n)
        batch = {name:metric(c[ix].sum(0)) for name,c in counts.items()}
        for name,m in batch.items():
            estimates[name].append([m["positive_pair_success"],m["negative_false_answer_rate"]])
        for name,(a,b) in comparisons.items():
            differences[name].append([batch[a]["positive_pair_success"]-batch[b]["positive_pair_success"],
                                      batch[a]["negative_false_answer_rate"]-batch[b]["negative_false_answer_rate"]])
    arms = {name:dict(positive_pair_success_ci95=np.quantile(x,[.025,.975],axis=0)[:,0].tolist(),
        negative_false_answer_rate_ci95=np.quantile(x,[.025,.975],axis=0)[:,1].tolist()) for name,x in estimates.items()}
    contrasts = {}
    point = {name:metric(c.sum(0)) for name,c in counts.items()}
    for name,x in differences.items():
        a,b = comparisons[name]
        contrasts[name]=dict(positive_pair_success_delta=point[a]["positive_pair_success"]-point[b]["positive_pair_success"],
            negative_false_answer_delta=point[a]["negative_false_answer_rate"]-point[b]["negative_false_answer_rate"],
            positive_delta_ci95=np.quantile(x,[.025,.975],axis=0)[:,0].tolist(),
            negative_false_answer_delta_ci95=np.quantile(x,[.025,.975],axis=0)[:,1].tolist(),
            positive_delta_one_sided95_lower=float(np.quantile(x,.05,axis=0)[0]))
    return arms,contrasts


def native_prob(data):
    return data["baseline"].softmax(-1).numpy()


def validate_inputs(bundle, annotations, vocabulary, vk_dir):
    _, auxiliary_checks = util.validate_bundle(bundle)
    if sorted(vocabulary["label_to_idx"].values()) != list(range(1,151)):
        raise ValueError("Not the standard VG150 foreground ontology")
    offsets = {}
    for split in ["train","development"]:
        d = bundle["data"][split]; by_id = {x["image_id"]:x for x in annotations[split]}
        if len(by_id)!=len(annotations[split]) or set(by_id)!=set(d["image_ids"]):
            raise ValueError("Annotation image coverage differs")
        offsets[split]={}
        for iid,a,b in zip(d["image_ids"],d["offsets"],d["offsets"][1:]):
            item=by_id[iid]
            if item["labels"]!=d["labels"][a:b].tolist() or len(item["boxes"])!=b-a:
                raise ValueError("Region ordering mismatch: "+iid)
            geometry(item["boxes"],item["size"])
            offsets[split][iid]=(a,b)
    vk=json.loads((vk_dir/"summary.json").read_text())
    protocol=json.loads((vk_dir/"protocol.json").read_text())
    if (util.digest(vk_dir/"protocol.json")!=vk["protocol_sha256"]
            or util.digest(vk_dir/"selected.pth")!=vk["full_checkpoint_sha256"]
            or util.digest(Path(util.__file__))!=protocol["helper_sha256"]):
        raise ValueError("Frozen VK artifacts changed")
    oof=torch.load(str(vk_dir/"oof_predictions.pt"),map_location="cpu",weights_only=True)
    d=bundle["data"]["development"]
    if (oof["protocol_sha256"]!=vk["protocol_sha256"] or oof["image_ids"]!=d["image_ids"] or oof["offsets"]!=d["offsets"]
            or oof["probabilities"].shape!=d["baseline"].shape):
        raise ValueError("VK OOF provenance/order mismatch")
    probabilities=oof["probabilities"]
    if (not torch.isfinite(probabilities).all() or (probabilities<0).any()
            or not torch.allclose(probabilities.sum(-1),torch.ones(len(probabilities)),atol=1e-5)):
        raise ValueError("Invalid repaired probabilities")
    top1=float(((probabilities[:,1:].argmax(-1)+1)==d["labels"]).double().mean())
    if abs(top1-vk["results"]["primary_oof"]["top1"])>1e-12:
        raise ValueError("VK OOF replay mismatch")
    selected=torch.load(str(vk_dir/"selected.pth"),map_location="cpu",weights_only=True)
    expert=torch.nn.functional.linear(d["features"],selected["expert_head"]["weight"],
        selected["expert_head"]["bias"]).softmax(-1)
    native,foreground,candidate=util.probabilities(d["baseline"],expert)
    visual,eligible=util.visual_features(d["features"],foreground,candidate,selected["centers"],selected["counts"])
    features=torch.cat([util.confidence_features(foreground,expert),visual],-1)
    crossfit=json.loads((vk_dir/"crossfit.json").read_text())
    assigned=util.group_folds(d,protocol["calibration_folds"])
    covered=set(); fold_checks=[]
    for fold in crossfit["folds"]:
        fit=set(fold["fit_image_ids"]); hold=set(fold["prediction_image_ids"])
        if (len(fit)!=400 or len(hold)!=100 or fit&hold or covered&hold
                or fit|hold!=set(d["image_ids"])):
            raise ValueError("VK calibration fold leakage")
        index=fold["fold"]; mask=assigned==index
        if hold!={iid for iid,f in protocol["calibration_folds"].items() if f==index}:
            raise ValueError("VK fold manifest mismatch")
        state=torch.load(str(vk_dir/("fold%d.pth"%index)),map_location="cpu",weights_only=True)
        if state["protocol_sha256"]!=vk["protocol_sha256"]:
            raise ValueError("VK fold checkpoint protocol mismatch")
        predicted,_,_=util.route(d["baseline"][mask],expert[mask],features[mask],state["primary"],eligible[mask])
        error=float((predicted-probabilities[mask]).abs().max())
        if error>1e-6:
            raise ValueError("VK probabilities do not reproduce excluded-fold checkpoint")
        covered|=hold; fold_checks.append(dict(fold=index,objects=int(mask.sum()),max_error=error))
    if covered!=set(d["image_ids"]) or not torch.allclose(native[:,0],probabilities[:,0],atol=1e-7):
        raise ValueError("Incomplete VK OOF coverage or changed background mass")
    return offsets,probabilities.numpy(),dict(auxiliary_checks=auxiliary_checks,vk_oof_checks=fold_checks)


def run(args):
    out=args.output; start=time.monotonic()
    paths=dict(bundle=args.bundle,annotations=out/"inputs/annotations.json",vocabulary=out/"inputs/VG-SGG-dicts.json",
        vk_summary=args.vk/"summary.json",vk_oof=args.vk/"oof_predictions.pt",vk_protocol=args.vk/"protocol.json")
    sidecar=json.loads(args.bundle.with_suffix(".json").read_text())
    if util.digest(args.bundle)!=sidecar["bundle_sha256"]:
        raise RuntimeError("Bundle transfer mismatch")
    protocol=dict(spec=SPEC,input_hashes={k:util.digest(p) for k,p in paths.items()},
        source_sha256=util.digest(Path(__file__)),helper_sha256=util.digest(Path(util.__file__)),
        plan_sha256=util.digest(Path(__file__).with_name("T1_MAC_PLAN.md")),
        reserved_confirmation_ids=sidecar["reserved_gate_ids"],
        native_joint_gate_not_replaced=True)
    util.seal(out/"protocol.json",protocol)
    if (out/"summary.json").exists() or (out/"thresholds.json").exists():
        raise RuntimeError("Preserve completed or frozen T1 run")
    util.progress(out,"verify_task_inputs",0,5500,start)
    bundle=torch.load(str(args.bundle),map_location="cpu",weights_only=True)
    annotations=json.loads(paths["annotations"].read_text()); vocab=json.loads(paths["vocabulary"].read_text())
    offsets,repaired,input_audit=validate_inputs(bundle,annotations,vocab,args.vk)
    util.write(out/"input_audit.json",input_audit)
    meta={s:{x["image_id"]:x for x in xs} for s,xs in annotations.items()}
    tr_ids=bundle["data"]["train"]["image_ids"]
    screen_ids=bundle["data"]["development"]["image_ids"]
    cal_ids=sorted(tr_ids,key=lambda i:hash_key(SPEC["image_salt"]+i))[:500]
    if set(cal_ids)&set(screen_ids) or set(cal_ids+screen_ids)&set(sidecar["reserved_gate_ids"]):
        raise RuntimeError("Task split leakage")
    util.write(out/"splits.json",dict(calibration=cal_ids,screening=screen_ids,
        confirmation=sidecar["reserved_gate_ids"],screening_is_exploratory=True))
    labels=bundle["data"]["train"]["labels"].numpy()
    counts=np.bincount(labels,minlength=151)[1:]
    supported=(np.flatnonzero(counts>=SPEC["negative_min_train_class_support"])+1).tolist()
    total=np.zeros((150,150),dtype=np.int64); numer=np.zeros((4,150,150),dtype=np.int64)
    for n,iid in enumerate(tr_ids,1):
        item=meta["train"][iid]; y=np.asarray(item["labels"])-1
        spatial=geometry(item["boxes"],item["size"])
        i,j=np.where(~np.eye(len(y),dtype=bool))
        np.add.at(total,(y[i],y[j]),1)
        for r in range(4):
            a,b=np.where(spatial[r]); np.add.at(numer[r],(y[a],y[b]),1)
        if n%1000==0: util.progress(out,"training_only_relation_frequency",n,5000,start)
    freq=(numer+1)/(total[None]+2)
    np.savez_compressed(out/"frequency_train_only.npz",total=total,relation_counts=numer,probabilities=freq)
    probs={s:native_prob(d) for s,d in bundle["data"].items()}
    tasks={}; coverage={}; negative_scores={name:[] for name in ARMS[:4]}
    for split,ids in [("calibration",cal_ids),("screening",screen_ids)]:
        source="train" if split=="calibration" else "development"
        tasks[split]=[]; coverage[split]={}
        for n,iid in enumerate(ids,1):
            item=meta[source][iid]; queries,cv=make_queries(item,supported)
            tasks[split]+=queries; coverage[split][iid]=cv
            if split=="calibration":
                a,b=offsets[source][iid]; p=probs[source][a:b]
                spatial=geometry(item["boxes"],item["size"])
                for q in queries:
                    if q["stratum"]=="positive": continue
                    for name in negative_scores:
                        _,score=pair_score(q["query"],p,spatial,freq,name)
                        negative_scores[name].append(score)
            if n%100==0: util.progress(out,"construct_"+split+"_queries",n,len(ids),start)
    thresholds={name:fit_threshold(values) for name,values in negative_scores.items()}
    thresholds["vk_oof_geometry"]=dict(thresholds["soft_geometry"],source="native geometry unchanged")
    thresholds["mismatched_geometry"]=dict(thresholds["soft_geometry"],source="native geometry unchanged")
    thresholds["oracle_identity_geometry"]=dict(threshold=.5,source="oracle upper bound")
    util.write(out/"thresholds.json",thresholds)
    threshold_hash=util.digest(out/"thresholds.json")
    util.write(out/"tasks.json",tasks); util.write(out/"task_construction_coverage.json",coverage)
    util.write(out/"class_support.json",dict(training_counts=counts.tolist(),negative_eligible_classes=supported,
        names=vocab["idx_to_label"]))
    rows=[]
    for n,task in enumerate(tasks["screening"],1):
        iid=task["image_id"]; item=meta["development"][iid]; a,b=offsets["development"][iid]
        p=probs["development"][a:b]; y=np.asarray(item["labels"]); spatial=geometry(item["boxes"],item["size"])
        if len(y)>1:
            shift=1+int.from_bytes(hash_key("T1_mismatch:"+iid)[:4],"big")%(len(y)-1)
            perm=np.roll(np.arange(len(y)),shift)
            shuffled=spatial[:,perm][:,:,perm]
        else: shuffled=spatial
        oracle=np.zeros_like(p); oracle[np.arange(len(y)),y]=1.
        decisions={}
        for name in ARMS:
            probabilities=oracle if name=="oracle_identity_geometry" else repaired[a:b] if name=="vk_oof_geometry" else p
            support=shuffled if name=="mismatched_geometry" else spatial
            mode="soft_geometry" if name in ARMS[4:] else name
            pair,score=pair_score(task["query"],probabilities,support,freq,mode)
            pair=returned_pair(pair,score,thresholds[name]["threshold"])
            decisions[name]=dict(decision_record(task,pair,y,spatial),score=score)
        native_labels=p[:,1:].argmax(-1)+1
        endpoints=sorted({i for pair in task["answers"] for i in pair})
        relevant_error=bool(endpoints and (native_labels[endpoints]!=y[endpoints]).any())
        rows.append(dict(**task,decisions=decisions,any_answer_endpoint_identity_error=relevant_error))
        if n%500==0: util.progress(out,"task_screen_all_arms",n,len(tasks["screening"]),start)
    if util.digest(out/"thresholds.json")!=threshold_hash:
        raise RuntimeError("Threshold changed during screening")
    if not all(r["decisions"]["oracle_identity_geometry"]["correct"] for r in rows):
        raise RuntimeError("Oracle inconsistency in task construction or evaluator")
    util.write(out/"decisions.json",rows)
    image_counts=accumulate(screen_ids,rows)
    np.savez_compressed(out/"image_counts.npz",**image_counts)
    ci,contrasts=paired_intervals(image_counts)
    results={name:dict(metric(c.sum(0)),**ci[name]) for name,c in image_counts.items()}
    util.write(out/"screening_coverage.json",dict(
        by_relation={r:sum(q["query"]["relation"]==r for q in rows) for r in RELATIONS},
        target_classes=sorted({q["query"]["target"] for q in rows}),
        anchor_classes=sorted({q["query"]["anchor"] for q in rows}),
        images_without_stratum={s:sum(v[s]==0 for v in coverage["screening"].values()) for s in
            ["positive","category_absent","relation_absent"]},
        emitted_error_audit={name:dict(
            identity_errors=sum(r["decisions"][name]["wrong_identity_when_emitted"] for r in rows),
            geometry_errors=sum(r["decisions"][name]["wrong_geometry_when_emitted"] for r in rows)) for name in ARMS}))
    strata={}
    for flag in [False,True]:
        selected=[r for r in rows if r["stratum"]=="positive" and r["any_answer_endpoint_identity_error"]==flag]
        strata[str(flag)]=dict(queries=len(selected),arms={name:dict(
            success=sum(r["decisions"][name]["positive_pair_success"] for r in selected),
            repairs_vs_hard=sum(not r["decisions"]["hard_geometry"]["correct"] and r["decisions"][name]["correct"] for r in selected),
            damages_vs_hard=sum(r["decisions"]["hard_geometry"]["correct"] and not r["decisions"][name]["correct"] for r in selected)
        ) for name in ARMS})
    util.write(out/"identity_error_strata.json",strata)
    primary=contrasts["geometry_minus_frequency"]
    checks=dict(positive_gain_lower_above_zero=primary["positive_delta_one_sided95_lower"]>0,
        negative_false_answer_increase_at_most_half_point=primary["negative_false_answer_delta"]<=.005)
    summary=dict(status="complete",results=results,contrasts=contrasts,pilot_checks=checks,
        screening_images=len(screen_ids),screening_queries=len(rows),calibration_queries=len(tasks["calibration"]),
        threshold_sha256=threshold_hash,protocol_sha256=util.digest(out/"protocol.json"),
        negative_scope="absence within supplied annotated regions, not verified whole-image absence",
        geometry_source="GT supplied box centers; no physical affordance validation",
        independent_confirmation_evaluated=False,native_sgg_gate_evaluated=False,robot_execution_evaluated=False,
        extra_model_training=False,cuda_used=False,seconds=time.monotonic()-start,
        next_step="Review all controls; retain frozen rules for independent native export. Do not promote pilot to robot or SGG mitigation success.")
    util.write(out/"summary.json",summary)
    util.progress(out,"complete_exploratory_screen",len(rows),len(rows),start,pilot_checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True); p.add_argument("--vk",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(args)
        except Exception as exc:
            util.write(args.output/"failure.json",dict(status="failed",error=type(exc).__name__+": "+str(exc)))
            raise


if __name__=="__main__": main()
