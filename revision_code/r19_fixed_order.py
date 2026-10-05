"""Exact native-order factorial replay: eliminate second-sort tie permutations."""
import argparse
import itertools
import json
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage
from evidence_completion import read, lock, sources, paired_ratio
from repro_experiment import config, dataset
from ve_experiment import lookup
from vb_native import OfficialMetrics
from vc_protocol import summarize_rows, KS
from vk_gate_protocol import OUT as VK, ids


def load_prediction(path, size):
    from pysgg.structures.bounding_box import BoxList
    with np.load(path) as z:
        boxes=torch.from_numpy(z["pred_boxes"].copy())*torch.tensor([size[0],size[1],size[0],size[1]])
        pred=BoxList(boxes,size,"xyxy")
        score,label=torch.from_numpy(z["pred_entity_scores"][:,1:].copy()).max(-1)
        pred.add_field("pred_labels",label+1);pred.add_field("pred_scores",score)
        pred.add_field("rel_pair_idxs",torch.from_numpy(z["pred_rel_pairs"].copy()).long())
        pred.add_field("pred_rel_scores",torch.from_numpy(z["pred_rel_scores"].copy()))
    return pred


def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["sgcls","sgdet"],required=True);a=p.parse_args()
    ensure_storage();torch.set_num_threads(2)
    out=ROOT/"results/R19_vk_fixed_order"/a.task
    cfg=config(a.task,out);ds=dataset(cfg,"val");mapping=lookup(ds);selected=ids()
    protocol=dict(version="r19_exact_native_rank_factorial_v1",task=a.task,image_ids=selected,
        fixed_order="Use the exact native or repaired cached pair order, chosen only by score factor; never sort an already-sorted list",
        source=sources(["r19_fixed_order.py","evidence_completion.py"]),
        scope="Descriptive output-channel decomposition; no new method/gate/threshold selection")
    reg=lock(out/"protocol.json",protocol);metric=OfficialMetrics(a.task);records=[];start=time.monotonic()
    from pysgg.structures.bounding_box import BoxList
    for n,iid in enumerate(selected,1):
        dest=out/"images"/(iid+".json")
        if dest.exists():
            row=read(dest)
            if row["protocol_sha256"]!=reg: raise RuntimeError("Resume mismatch")
            records.append(row);continue
        gt=ds.get_groundtruth(mapping[iid],evaluation=True)
        raw=torch.load(str(ROOT/"cache/R19_vk_mechanism"/a.task/(iid+".pt")),map_location="cpu")
        base=load_prediction(VK/a.task/"gate/native/predictions"/(iid+".npz"),gt.size)
        changed=load_prediction(VK/a.task/"gate/selective/predictions"/(iid+".npz"),gt.size)
        bp=base.get_field("rel_pair_idxs").numpy(); cp=changed.get_field("rel_pair_idxs").numpy()
        bi=np.lexsort((bp[:,1],bp[:,0]));ci=np.lexsort((cp[:,1],cp[:,0]))
        if not np.array_equal(bp[bi],cp[ci]) or not torch.equal(base.get_field("pred_rel_scores")[bi],changed.get_field("pred_rel_scores")[ci]):
            raise RuntimeError("Pair/predicate probability invariance lost")
        arms={}
        for l,s,b in itertools.product([0,1],repeat=3):
            name="L%d_S%d_B%d"%(l,s,b)
            pred=BoxList((changed if b else base).bbox.clone(),gt.size,"xyxy")
            pred.add_field("pred_labels",(changed if l else base).get_field("pred_labels"))
            for field in ("pred_scores","rel_pair_idxs","pred_rel_scores"):
                pred.add_field(field,(changed if s else base).get_field(field))
            arms[name]=metric.row(iid,pred,gt,dict(logits=raw["repaired_logits"] if s else raw["native_logits"]),raw["target"].numpy())
        for name,source in [("L0_S0_B0","native"),("L1_S1_B1","selective")]:
            expected=read(VK/a.task/"gate"/source/"images"/(iid+".json"))
            for field in ("recalls","post_nms_correct","positive_correct"):
                if arms[name][field]!=expected[field]: raise RuntimeError("Frozen endpoint metric changed: "+field)
        row=dict(image_id=iid,protocol_sha256=reg,arms=arms);atomic_json(dest,row);records.append(row)
        if n%50==0:
            elapsed=time.monotonic()-start
            progress=dict(stage="exact_native_order_factorial",images=n,total=len(selected),seconds=elapsed,eta_seconds=elapsed/n*(len(selected)-n))
            atomic_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
    results={};ref=[r["arms"]["L0_S0_B0"] for r in records]
    for name in records[0]["arms"]:
        rows=[r["arms"][name] for r in records]
        results[name]=dict(metrics=summarize_rows(rows),R50=paired_ratio([r["recalls"][KS.index(50)] for r in rows],
            [r["recalls"][KS.index(50)] for r in ref],np.ones(len(rows))))
    atomic_json(out/"summary.json",dict(status="complete",protocol_sha256=reg,images=len(records),results=results,
        native_endpoint_metrics_exact=True,confirmatory_claim=False,selection_performed=False))


if __name__=="__main__":main()
