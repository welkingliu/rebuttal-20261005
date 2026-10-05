"""Count- and ranking-aware sensitivity audit of the frozen V-K proposal metric."""
import argparse
import json
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common import ROOT, atomic_json, ensure_storage
from evidence_completion import lock, read, sources, paired_ratio
from repro_experiment import config, dataset
from ve_experiment import lookup
from vc_protocol import iou_pixel
from vk_gate_protocol import OUT as VK, ids


def unique_support(overlap, threshold=.5):
    """Maximize valid-match cardinality, then IoU, without consulting labels."""
    overlap = np.asarray(overlap)
    if overlap.ndim != 2: raise ValueError("Expected proposal-by-GT overlaps")
    if not overlap.size: return np.array([], dtype=int), np.array([], dtype=int)
    priority = min(overlap.shape) + 1
    proposal, gt = linear_sum_assignment(-((overlap >= threshold) * priority + overlap))
    valid = overlap[proposal, gt] >= threshold
    return proposal[valid], gt[valid]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--task",choices=["sgcls","sgdet"],required=True);a=p.parse_args()
    ensure_storage();torch.set_num_threads(2)
    out=ROOT/"results/R19_proposal_sensitivity"/a.task
    cfg=config(a.task,out);ds=dataset(cfg,"val");mapping=lookup(ds);selected=ids()
    protocol=dict(version="r19_unique_input_support_v1",task=a.task,image_ids=selected,
        scope="Post-hoc metric sensitivity, not a replacement acceptance gate or detection AP",
        matching="Class-agnostic fixed input proposals; Hungarian maximum IoU>=0.5 match cardinality, then total IoU; no GT identity in assignment",
        native_box_counting="Original positive proposal metric retained separately; final class-specific boxes are NOT substituted in this support audit",
        rank_scope="Incident in the frozen native top50 relation pairs; classification transitions evaluated on the same input proposal indices",
        sources=sources(["r19_proposal_sensitivity.py","evidence_completion.py"]))
    reg=lock(out/"protocol.json",protocol);records=[];start=time.monotonic()
    for n,iid in enumerate(selected,1):
        dest=out/"images"/(iid+".json")
        if dest.exists():
            row=read(dest)
            if row["protocol_sha256"]!=reg:raise RuntimeError("Resume mismatch")
            records.append(row);continue
        raw=torch.load(str(ROOT/"cache/R19_vk_mechanism"/a.task/(iid+".pt")),map_location="cpu")
        gt=ds.get_groundtruth(mapping[iid],evaluation=True).resize(raw["size"])
        truth=gt.get_field("labels").numpy()
        proposal,matched=unique_support(iou_pixel(raw["proposal_boxes"].numpy(),gt.bbox.numpy()))
        labels={"pre_native":raw["native_logits"][:,1:].argmax(-1).numpy()+1,
                "pre_repaired":raw["repaired_logits"][:,1:].argmax(-1).numpy()+1}
        with np.load(VK/a.task/"gate/native/predictions"/(iid+".npz")) as z:
            labels["post_native"]=z["pred_entity_scores"][:,1:].argmax(1)+1
            ranked=set(map(int,z["pred_rel_pairs"][:50].reshape(-1)))
        with np.load(VK/a.task/"gate/selective/predictions"/(iid+".npz")) as z:
            labels["post_repaired"]=z["pred_entity_scores"][:,1:].argmax(1)+1
        if len({len(x) for x in labels.values()})!=1:raise RuntimeError("Proposal order mismatch")
        metric={key:int((value[proposal]==truth[matched]).sum()) for key,value in labels.items()}
        y=raw["target"].numpy(); valid=y>0; native=labels["post_native"]==y; repaired=labels["post_repaired"]==y
        incident=np.array([i in ranked for i in range(len(y))])
        strata={}
        for name,keep in [("native_top50_endpoint",incident&valid),("not_native_top50_endpoint",~incident&valid)]:
            strata[name]=dict(objects=int(keep.sum()),correct_native=int(native[keep].sum()),correct_repaired=int(repaired[keep].sum()),
                corrections=int((~native&repaired&keep).sum()),regressions=int((native&~repaired&keep).sum()))
        row=dict(image_id=iid,protocol_sha256=reg,gt_objects=len(truth),matched_unique_objects=len(proposal),
            input_proposals=len(y),positive_proposals=int(valid.sum()),unique_correct=metric,rank_strata=strata,
            proposal_indices=proposal.tolist(),gt_indices=matched.tolist())
        atomic_json(dest,row);records.append(row)
        if n%100==0:
            elapsed=time.monotonic()-start
            value=dict(stage="one_to_one_support_sensitivity",images=n,total=len(selected),seconds=elapsed,eta_seconds=elapsed/n*(len(selected)-n))
            atomic_json(out/"progress.json",value);print(json.dumps(value),flush=True)
    result={}
    for prefix in ("pre","post"):
        result[prefix]=paired_ratio([r["unique_correct"][prefix+"_repaired"] for r in records],
            [r["unique_correct"][prefix+"_native"] for r in records],[r["matched_unique_objects"] for r in records])
    strata={}
    for name in records[0]["rank_strata"]:
        rows=[r["rank_strata"][name] for r in records]
        strata[name]=dict(counts={key:sum(r[key] for r in rows) for key in rows[0]},
            paired=paired_ratio([r["correct_repaired"] for r in rows],[r["correct_native"] for r in rows],[r["objects"] for r in rows]))
    atomic_json(out/"summary.json",dict(status="complete",images=len(records),protocol_sha256=reg,
        unique_support_results=result,rank_strata=strata,total_gt_objects=sum(r["gt_objects"] for r in records),
        matched_unique_objects=sum(r["matched_unique_objects"] for r in records),
        original_gate_unchanged=True,new_gate_claim=False))


if __name__=="__main__":main()
