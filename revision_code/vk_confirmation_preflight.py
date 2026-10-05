"""Read-only native V-K readiness audit; never launches inference or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    result=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): result.update(chunk)
    return result.hexdigest()


def run(root,old,output):
    read=lambda p:json.loads(p.read_text())
    vk=root/"results/VK_mac_calibration_shift"
    summary=read(vk/"summary.json"); protocol=read(vk/"protocol.json")
    assert sha(vk/"protocol.json")==summary["protocol_sha256"]
    assert sha(vk/"selected.pth")==summary["full_checkpoint_sha256"]
    assert summary["screening_eligible"] and not summary["heldout_confirmation_evaluated"]
    gate=protocol["provenance"]["reserved_gate_ids"]; ids=set(gate)
    assert len(gate)==len(ids)==1000
    historical=[]
    for task in ["sgcls","sgdet"]:
        for path in [root/"cache"/name/task/"protocol.json" for name in ["VB","VC"]]+[root/"results/VE"/task/"splits.json"]:
            split=read(path); used=set(split["development"]+split["gate"])
            assert not ids&used,"Reserved IDs intersect previous adapter assessment: "+str(path)
            historical.append(dict(path=str(path),sha256=sha(path),overlap=len(ids&used)))
    vi=read(root/"results/VI_selective_fusion/splits.json")
    assert set(vi["gate"])==ids
    assert not ids&set(vi["train"]+vi["development"])
    consumed=[]
    for parent in [root/"results/VF",root/"results/VI_selective_fusion",root/"results/VK_native_confirmation"]:
        if parent.exists():
            for folder in parent.glob("*/gate"):
                consumed += [str(p) for p in folder.rglob("*.json")]
    assert not consumed,"Native confirmation artifacts already exist; inspect before running again"
    matches=list(old.glob("checkpoints/sgg/trained/pysgg/transformer_object_refine_fixed_20260724/sgcls/*/*/model_final.pth"))
    assert len(matches)==1,"Ambiguous SGCls baseline"
    weights=dict(sgcls=matches[0],sgdet=root/"results/R9_reproduction/sgdet/training/model_final.pth",
        dinov2=old/"checkpoints/foundation/dinov2/dinov2_vitb14_pretrain.pth")
    assets={k:dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else 0,
        sha256=sha(p) if p.is_file() else None) for k,p in weights.items()}
    assert assets["sgcls"]["sha256"]=="ed150f8ad4615cd059bc79a1819ffd8b13a0f54d97f1940156be79a83444a59a"
    reference={}
    for task,folder in dict(sgcls=root/"results/R9_reproduction/sgcls/eval_corrected/val/predictions",
            sgdet=root/"results/R14_pair_cap/sgdet/predictions/reference_all_pairs").items():
        reference[task]=dict(path=str(folder),present=sum((folder/(i+".npz")).is_file() for i in ids),
            required=1000,full_logits_available=False,
            limitation="Converted reference stores selected-class scores, not full proposal logits; native export still required")
    result=dict(status="awaiting_native_export_and_runtime_integration",timestamp=datetime.now(timezone.utc).isoformat(),
        root=str(root),old_inputs_read_only=str(old),primary_checkpoint_sha256=summary["full_checkpoint_sha256"],
        native_vk_gate=protocol["spec"]["gate"],confirmation_ids=gate,historical_split_checks=historical,
        gate_unconsumed_in_checked_paths=True,scope="Held out from adaptation; native validation audits have seen these images",
        assets=assets,reference_caches=reference,
        ready_to_launch=False,inference_or_training_launched=False,
        missing_steps=["Dedicated VK native runner, no-op parity and actual postprocessing checks",
            "Export full native object logits, ordered proposal boxes and DINO crop features on reserved1000",
            "Frozen VK inference plus native SGCls gate; SGDet only after SGCls passes",
            "Keep existing R16 GPU training untouched; no GPU sharing"],
        no_new_weights_required=all(x["exists"] for x in assets.values()),
        existing_gpu_queue={str(i):read(root/("status/R16_gpu%d.json"%i)) for i in [0,1]})
    output=output.resolve(); assert root.resolve() in output.parents
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(".tmp"); temp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); temp.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k not in ["confirmation_ids","historical_split_checks","existing_gpu_queue"]},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True); p.add_argument("--old",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args(); run(a.root,a.old,a.output)
