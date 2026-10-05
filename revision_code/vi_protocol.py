"""Finite V-I selective fusion: training-only cross-fitting, then native gates."""
import copy
import hashlib
import json
from pathlib import Path

from common import ROOT, atomic_json, sha256
from vc_protocol import SPEC as VC_SPEC

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VI_selective_fusion"
CACHE = ROOT / "cache/VI_selective_fusion"
CKPT = ROOT / "checkpoints/VI_selective_fusion"
MANIFEST = ROOT / "manifests/VI_protocol.json"
SPEC = dict(
    version="vi_oof_selective_visual_fusion_v1", seed=17, folds=5,
    train_images=5000, development_images=500, gate_images=1000,
    encoder="frozen local DINOv2-B; reuse VH normalized GT-crop features",
    auxiliary="five image-disjoint cross-fitted linear 768-to150 probes, then one full-train probe",
    probe_epochs=40, probe_lr=.003, probe_weight_decay=.01, probe_batch=256,
    probe_selection="fixed40 epochs, no out-of-fold/development label-based epoch selection",
    router="standardized confidence/margin/entropy and cross-label probabilities; one logistic linear head",
    router_labels="OOF fusion correct and native wrong versus native correct and OOF fusion wrong; zero-utility rows excluded",
    router_training="all informative training-only OOF objects, unweighted BCE; no class balancing or development labels",
    router_epochs=200, router_lr=.01, router_l2=.01,
    alpha=.5, repair_probability_min=.6,
    routing="only disagreeing foreground predictions whose candidate changes label and router probability>=0.6",
    score_rule="mix foreground conditional probabilities with alpha0.5; preserve original background mass, not old foreground scores",
    requirements=dict(identity_gain_min=.005, correction_precision_min=.6, macro_accuracy_not_lower=True),
    development="same VH500 reused for screening only, not independent efficacy; no hyperparameter search",
    gate_scope="previously unconsumed VF adaptation gate; excludes earlier adaptation gates, but native reference audits saw validation",
    native="fixed Transformer predicate/context; updated object logits go through native NMS, class boxes and triplet ranking",
    sgdet="same SGCls-trained expert/router, no detected-proposal fitting or gate-driven selection",
    limitation="only the auxiliary expert is cross-fitted; the frozen native SGG was trained on native training images",
    stop="development failure stops gate; SGCls failure stops SGDet; no automatic extra seeds, test or parameter search",
    wall_hours=12,
)
SPEC["gate"] = copy.deepcopy(VC_SPEC["gate"])
SPEC["gate"]["candidate"] = "selective_fusion_seed17"
SOURCES = ["vi_protocol.py", "vi_selective.py", "vi_native.py", "vi_queue.py",
           "common.py", "vc_protocol.py", "vf_native.py", "ve_experiment.py",
           "native_runtime.py", "vb_native.py", "independent_identity_expert.py"]


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError("Immutable V-I record changed: " + str(path))
    if not path.exists():
        atomic_json(path, value)


def fold_map(ids):
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training image IDs")
    order = sorted(ids, key=lambda x: hashlib.sha256(("VI_crossfit:" + x).encode()).digest())
    return {iid: n % SPEC["folds"] for n, iid in enumerate(order)}


def register():
    from vf_protocol import verify as verify_vf
    vh = ROOT / "results/VH_independent_identity"
    split = read(ROOT / "results/VF/sgcls/splits.json")
    if read(vh / "summary.json")["status"] != "complete":
        raise RuntimeError("V-H evidence missing")
    for task in ["sgcls", "sgdet"]:
        folder = ROOT / "results/VF" / task / "gate"
        if folder.exists() and any(folder.rglob("*.json")):
            raise RuntimeError("VF adaptation gate already consumed")
    inputs = [vh / "protocol.json", vh / "annotations.json", ROOT / "results/VF/sgcls/splits.json"]
    value = dict(spec=SPEC, vf_registration=verify_vf(),
                 sources={n: sha256(HERE / n) for n in SOURCES},
                 inputs={str(p): sha256(p) for p in inputs},
                 encoder_sources=read(vh / "protocol.json")["encoder_sources"],
                 folds=fold_map(split["train"]))
    immutable(MANIFEST, value)
    immutable(OUT / "splits.json", dict(train=split["train"], development=split["development"],
                                      gate=split["gate"], protocol_sha256=sha256(MANIFEST)))


def verify():
    value = read(MANIFEST)
    if value["spec"] != SPEC or any(sha256(HERE / n) != h for n, h in value["sources"].items()):
        raise RuntimeError("Registered V-I code/spec changed")
    if any(sha256(Path(p)) != h for p, h in value["inputs"].items()):
        raise RuntimeError("V-I input provenance changed")
    return sha256(MANIFEST)
