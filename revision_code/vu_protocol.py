"""Immutable V-U sources, input proposals and frozen visual assets."""
from pathlib import Path
from common import ROOT, OLD, sha256
from evidence_completion import read, sources, lock

NAME="VU_reobservation"
OUT=ROOT/"results"/NAME
CACHE=ROOT/"cache"/NAME
WEIGHTS=ROOT/"checkpoints"/NAME
MODEL=OLD/"checkpoints/foundation/hf_models/siglip2_b"
SOURCE_NAMES=["vu_protocol.py","vu_math.py","vu_features.py","vu_experiment.py",
              "vu_queue.py","test_vu_reobservation.py","VU_REOBSERVATION_PLAN.txt"]


def register(smoke=False):
    from vq_experiment import registration as parent_registration
    from vp_experiment import INPUT,VO
    from vu_math import ARMS,PRIMARY
    parent,digest,lane=parent_registration(smoke)
    assets=[MODEL/"config.json",MODEL/"preprocessor_config.json"]+sorted(MODEL.glob("*.safetensors"))
    if len(assets)<3 or not all(p.is_file() for p in assets): raise RuntimeError("Missing local SigLIP2 assets")
    rows=[]
    for iid in parent["image_ids"]:
        raw=INPUT/"raw"/(iid+".pt"); expected=read(VO/"replay/images"/(iid+".json"))["inputs"]["raw"]
        if sha256(raw)!=expected: raise RuntimeError("Raw proposal cache drift")
        rows.append(dict(image_id=iid,raw=str(raw),raw_sha256=expected))
    value=dict(version="vu_reobservation_v1",smoke=smoke,image_ids=parent["image_ids"],folds=parent["folds"],
        vq_protocol_sha256=digest,vn_protocol_sha256=parent["vn_protocol_sha256"],
        native_checkpoint_sha256=parent["native_checkpoint_sha256"],inputs=rows,
        visual_assets={str(p):sha256(p) for p in assets},model_dir=str(MODEL),
        sources=sources(SOURCE_NAMES),arms=ARMS,primary=PRIMARY,
        plan=Path(__file__).with_name("VU_REOBSERVATION_PLAN.txt").read_text(),
        validation_accessed=False,test_accessed=False,formal_gate_accepted=False,independent_confirmation=False,
        reused_train_images=True,native_trained_on_these_images=True)
    return value,lock(OUT/lane/"protocol.json",value),lane


def load(smoke=False,check_assets=False):
    lane="smoke" if smoke else "train3000"; path=OUT/lane/"protocol.json"
    value=read(path)
    if sources(SOURCE_NAMES)!=value["sources"]: raise RuntimeError("V-U source drift")
    if check_assets:
        for p,digest in value["visual_assets"].items():
            if sha256(p)!=digest: raise RuntimeError("Frozen visual asset drift: "+p)
    return value,sha256(path),lane
