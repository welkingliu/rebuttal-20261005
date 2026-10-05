"""R13: expand historical GT-box transfer to all eligible VG-test/GQA-val images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from common import ROOT, OLD, ensure_storage, output_path, atomic_json, sha256
from external_overlap_audit import correctness, summarize

OUT = ROOT/"results/R13_gqa_disjoint"
CACHE = OLD/"data/derived/predictions/experiment_5_external/gqa/pysgg_tde_motifs_vg_live"
ANN = OLD/"data/gqa/val_sceneGraphs.json"
SPEC = dict(dataset="GQA validation intersected with original VG test image IDs",
    task="GT-box exact shared-label diagnostic, not native GQA SGDet",
    model="existing native TDE-Motifs SGCls checkpoint", mapping="normalized exact labels, no synonyms",
    optimization="none; six unchanged historical calibration states applied offline",
    smoke="Reproduce three image outputs from the old GQA cache before expanding the sample",
    missing_images="Explicitly count/exclude before protocol registration; no inference-time silent skip",
    interpretation="Unseen to VG task training; foundation pretraining exposure is not certified")


def dependencies():
    from sgg_core.data import gqa_psg_data_utils as data
    from sgg_core.data import shared_vg_ontology as mapping
    from sgg_core.data import data_utils
    from sgg_core.models.adapters import pysgg_live
    import pysgg_live_worker
    return [Path(module.__file__) for module in (data,mapping,data_utils,pysgg_live,pysgg_live_worker)]


def fingerprint():
    files = [Path(__file__).resolve(), ROOT/"code/common.py", ROOT/"code/native_runtime.py", ROOT/"code/external_overlap_audit.py",
             ANN, OLD/"data/vg/image_data.json", OLD/"data/vg/VG-SGG-with-attri.h5",
             OLD/"data/vg/v1.4/VG-SGG-dicts.json", CACHE/"metadata.json", CACHE/"predictions.npz"]
    files += dependencies()
    files += sorted((ROOT/"imported/experiment5").rglob("mitigated_state_dict.pth"))
    if len(list((ROOT/"imported/experiment5").rglob("mitigated_state_dict.pth"))) != 6:
        raise RuntimeError("Six historical adaptation states required")
    return dict(spec=SPEC,sources={str(p):sha256(p) for p in files})


def loader():
    from sgg_core.data.gqa_psg_data_utils import GQASceneGraphDataset
    from sgg_core.data.shared_vg_ontology import load_vg_ontology,build_exact_mapping
    ds=GQASceneGraphDataset(str(ANN),num_samples=0,vocabulary_path=str(ANN),image_root=str(OLD/"data/gqa/images"),
                           include_proxy_features=False,include_raw_images=True)
    vg=load_vg_ontology(OLD/"data/vg/v1.4/VG-SGG-dicts.json")
    return ds,vg,build_exact_mapping(ds,vg)


def prepare():
    from sgg_core.data.shared_vg_ontology import project_batch_to_vg
    manifest=ROOT/"manifests/R13_gqa_disjoint.json"
    source=fingerprint()
    if manifest.exists():
        if json.loads(manifest.read_text())["contract"]!=source:
            raise RuntimeError("R13 already registered with different sources")
        print("[PREPARED] Existing R13 registration verified",flush=True);return
    images=[r for r in json.loads((OLD/"data/vg/image_data.json").read_text()) if r["image_id"] not in {1592,1722,4616,4617}]
    with h5py.File(OLD/"data/vg/VG-SGG-with-attri.h5","r") as h:
        splits=h["split"][:]
    if len(splits)!=len(images):raise RuntimeError("VG metadata alignment error")
    test_ids={str(row["image_id"]) for row,s in zip(images,splits) if s==2}
    raw=json.loads(ANN.read_text()); candidates=sorted(test_ids & set(raw))
    ds,vg,mapping=loader(); lookup={str(row[0]):i for i,row in enumerate(ds.items)}
    eligible,excluded,counts=[],[],Counter();start=time.monotonic()
    for n,iid in enumerate(candidates):
        graph=raw[iid]
        counts["candidate_images"]+=1
        counts["raw_annotation_objects"]+=len(graph.get("objects",{}))
        counts["raw_annotation_relation_rows"]+=sum(len(o.get("relations",[])) for o in graph.get("objects",{}).values())
        if iid not in lookup:
            excluded.append(dict(image_id=iid,reason="loader_unavailable_image_or_graph"));continue
        batch=ds[lookup[iid]]
        projected,report=project_batch_to_vg(batch,mapping,vg)
        for key in ("source_objects","source_relations","retained_objects","retained_relations"):
            counts[key]+=int(report.get(key,0))
        if projected is None:
            excluded.append(dict(image_id=iid,reason="no_shared_relation_or_raw_image",details=report));continue
        eligible.append(iid)
        if n%100==0:
            print(json.dumps(dict(stage="prepare",images=n+1,total=len(candidates),eligible=len(eligible))),flush=True)
    if len(eligible)<182:raise RuntimeError("Expanded external evaluation unexpectedly smaller than old disjoint subset")
    with np.load(CACHE/"predictions.npz",allow_pickle=False) as p:
        old_ids=set(p["object_image_ids"].astype(str))
    smoke=sorted(old_ids & set(eligible))[:3]
    if len(smoke)!=3:raise RuntimeError("Need three old-cache-compatible smoke images")
    record=dict(contract=source,image_ids=eligible,smoke_image_ids=smoke,counts=dict(counts),excluded=excluded,
                annotation_candidates=len(candidates),seconds=time.monotonic()-start,
                mapping={k:v for k,v in mapping.items() if not k.endswith("_map")})
    atomic_json(manifest,record)
    atomic_json(OUT/"preparation.json",dict(status="complete",eligible_images=len(eligible),counts=dict(counts),excluded=len(excluded)))
    print("[PREPARED] R13 eligible=%d excluded=%d"%(len(eligible),len(excluded)),flush=True)


def verify():
    path=ROOT/"manifests/R13_gqa_disjoint.json"
    d=json.loads(path.read_text())
    if d["contract"]!=fingerprint():raise RuntimeError("R13 source contract changed")
    return d,sha256(path)


def export(smoke=False):
    from native_runtime import load_model
    from sgg_core.data.shared_vg_ontology import project_batch_to_vg
    from pysgg_live_worker import _run_request
    from sgg_core.models.adapters.pysgg_live import PySGGLiveAdapter
    registration,digest=verify()
    stage="smoke" if smoke else "export"
    out=OUT/stage
    if not smoke and not json.loads((OUT/"smoke/summary.json").read_text()).get("historical_cache_match"):
        raise RuntimeError("R13 old-cache smoke failed")
    model,cfg,transform,provenance=load_model("tde_motifs","sgcls",out)
    metadata=json.loads((CACHE/"metadata.json").read_text())
    if provenance["checkpoint_sha256"]!=metadata["checkpoint_sha256"]:
        raise RuntimeError("External evaluation checkpoint changed")
    ds,vg,mapping=loader();lookup={str(row[0]):i for i,row in enumerate(ds.items)}
    ids=registration["smoke_image_ids"] if smoke else registration["image_ids"]
    old=None
    if smoke:
        with np.load(CACHE/"predictions.npz",allow_pickle=False) as f:old={k:f[k] for k in f.files}
    start=time.monotonic(); rows=[]
    for index,iid in enumerate(ids):
        dest=out/"images"/(iid+".npz"); record=out/"images"/(iid+".json")
        if dest.exists() and record.exists():
            r=json.loads(record.read_text())
            if r["registration_sha256"]!=digest:raise RuntimeError("Resume registration mismatch")
            rows.append(r);continue
        projected,report=project_batch_to_vg(ds[lookup[iid]],mapping,vg)
        if projected is None:raise RuntimeError("Registered image no longer eligible: "+iid)
        source={k:projected[k].numpy() for k in ("image","boxes","entity_labels","rel_pairs","rel_labels")}
        source["require_gt_pairs"]=np.asarray(True)
        input_path=output_path(ROOT/"tmp/R13_gqa"/(stage+"_input.npz"))
        with input_path.open("wb") as stream:np.savez(stream,**source)
        result=_run_request(model,transform,input_path,"cuda",cfg.DATALOADER.SIZE_DIVISIBILITY)
        # The historical parent adapter converted worker probabilities to log space
        # before applying its identity-initialized calibration heads.
        scores={key:PySGGLiveAdapter._log_probabilities(torch.from_numpy(result[key])).numpy()
                for key in ("pred_entity_scores","pred_rel_scores")}
        payload=dict(object_scores=scores["pred_entity_scores"],object_targets=source["entity_labels"],
                     relation_scores=scores["pred_rel_scores"],relation_targets=source["rel_labels"],
                     relation_subject=source["rel_pairs"][:,0],relation_object=source["rel_pairs"][:,1])
        payload["object_image_ids"]=np.repeat(iid,len(payload["object_targets"]))
        payload["relation_image_ids"]=np.repeat(iid,len(payload["relation_targets"]))
        match=None
        if smoke:
            oi=old["object_image_ids"].astype(str)==iid;ri=old["relation_image_ids"].astype(str)==iid
            for key,sel in [("object_targets",oi),("relation_targets",ri)]:
                if not np.array_equal(payload[key],old[key][sel]):raise RuntimeError("Old cache targets differ: "+key)
            errors={key:float(np.abs(payload[key]-old[key][sel]).max()) for key,sel in [("object_scores",oi),("relation_scores",ri)]}
            if any(value>1e-3 for value in errors.values()):raise RuntimeError("Old cache logits/scores differ: "+str(errors))
            match=errors
        temporary=output_path(dest.with_suffix(".tmp"))
        with temporary.open("wb") as stream:np.savez_compressed(stream,**payload)
        temporary.replace(dest)
        r=dict(image_id=iid,registration_sha256=digest,objects=len(payload["object_targets"]),relations=len(payload["relation_targets"]),old_cache_max_errors=match)
        atomic_json(record,r);rows.append(r)
        elapsed=time.monotonic()-start
        progress=dict(stage=stage,images=index+1,total=len(ids),seconds=elapsed,eta_seconds=elapsed/(index+1)*(len(ids)-index-1))
        atomic_json(out/"progress.json",progress)
        if index%25==0:print(json.dumps(progress),flush=True)
    atomic_json(out/"summary.json",dict(status="complete",images=len(rows),registration_sha256=digest,
        objects=sum(r["objects"] for r in rows),relations=sum(r["relations"] for r in rows),
        historical_cache_match=bool(smoke and all(r["old_cache_max_errors"] is not None for r in rows)),seconds=time.monotonic()-start))


def evaluate():
    registration,digest=verify()
    done=json.loads((OUT/"export/summary.json").read_text())
    if done["status"]!="complete" or done["registration_sha256"]!=digest:raise RuntimeError("Export incomplete")
    arrays={};offset=0
    for iid in registration["image_ids"]:
        with np.load(OUT/"export/images"/(iid+".npz"),allow_pickle=False) as z:
            for key in z.files:
                value=z[key].copy()
                if key in ("relation_subject","relation_object"):value+=offset
                arrays.setdefault(key,[]).append(value)
            offset+=len(z["object_targets"])
    payload={key:np.concatenate(value) for key,value in arrays.items()}
    base=correctness(payload,payload["object_scores"],payload["relation_scores"])
    ids=set(registration["image_ids"])
    conditions={"base":summarize(payload,base,base,ids)}
    metadata=json.loads((CACHE/"metadata.json").read_text())
    for path in sorted((ROOT/"imported/experiment5").rglob("mitigated_state_dict.pth")):
        checkpoint=torch.load(str(path),map_location="cpu")
        if checkpoint["base_checkpoint_sha256"]!=metadata["checkpoint_sha256"]:raise RuntimeError("Historical checkpoint mismatch")
        state=checkpoint["grounding_state_dict"]
        scores=[]
        for key,prefix in [("object_scores","entity_calibrator"),("relation_scores","relation_calibrator")]:
            scores.append(torch.nn.functional.linear(torch.from_numpy(payload[key]).float(),state[prefix+".weight"].float(),state[prefix+".bias"].float()).numpy())
        name=path.parents[2].name+"/"+path.parent.name
        conditions[name]=summarize(payload,correctness(payload,*scores),base,ids)
    atomic_json(OUT/"summary.json",dict(status="complete",registration_sha256=digest,images=len(ids),spec=SPEC,
        conditions=conditions,coverage=registration["counts"],annotation_candidates=registration["annotation_candidates"],
        exclusions=registration["excluded"],warning="Shared-label GT-box transfer only, not full native GQA SGDet or evidence of a new mitigation"))
    print("[COMPLETE] R13 training-disjoint GQA",flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","smoke","export","evaluate"]);args=p.parse_args()
    ensure_storage();torch.set_num_threads(4)
    if args.stage=="prepare":prepare()
    elif args.stage=="evaluate":evaluate()
    else:export(smoke=args.stage=="smoke")


if __name__=="__main__":main()
