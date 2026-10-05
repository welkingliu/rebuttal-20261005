"""Sealed inputs for the conditional-visual/background repair pilot."""
from pathlib import Path
from common import ROOT, sha256
from evidence_completion import read, sources, lock
from vu_protocol import load as load_parent, OUT as PARENT_OUT, CACHE as PARENT_CACHE
from vu_protocol import SOURCE_NAMES as VU_SOURCES
from vq_experiment import SOURCE_NAMES as VQ_SOURCES
from vv_math import SPEC, PRIMARY, TRAINED, ARMS

OUT = ROOT/"results/VV_conditional_repair"
WEIGHTS = ROOT/"checkpoints/VV_conditional_repair"
SOURCE_NAMES = sorted(set(VU_SOURCES + VQ_SOURCES + [
    "vv_protocol.py", "vv_math.py", "vv_experiment.py", "vv_queue.py",
    "test_vv_conditional.py", "VV_CONDITIONAL_REPAIR_PLAN.txt",
    "common.py", "evidence_completion.py", "vo_math.py", "vc_protocol.py",
    "vn_train_proposals.py", "vk_gate_native.py", "vb_native.py"]))


def register(smoke=False):
    parent, parent_sha, lane = load_parent(smoke, check_assets=True)
    pack = read(PARENT_OUT/lane/"prepare/summary.json")
    path = PARENT_CACHE/lane/"data.pt"
    if pack["protocol_sha256"] != parent_sha or sha256(path) != pack["bundle_sha256"]:
        raise RuntimeError("V-U feature bundle changed")
    value = dict(version="vv_conditional_visual_background_v1", smoke=smoke,
        image_ids=parent["image_ids"], folds=parent["folds"],
        vn_protocol_sha256=parent["vn_protocol_sha256"],
        native_checkpoint_sha256=parent["native_checkpoint_sha256"],
        vu_protocol_sha256=parent_sha, bundle_sha256=pack["bundle_sha256"],
        bundle_path=str(path), sources=sources(SOURCE_NAMES), spec=SPEC,
        primary=PRIMARY, fitted_arms=TRAINED, arms=ARMS,
        plan=Path(__file__).with_name("VV_CONDITIONAL_REPAIR_PLAN.txt").read_text(),
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False,
        independent_confirmation=False, reused_train_images=True,
        native_sgg_trained_on_these_images=True, designed_after_vu_results=True)
    return value, lock(OUT/lane/"protocol.json", value), lane


def load(smoke=False):
    lane = "smoke" if smoke else "train3000"
    path = OUT/lane/"protocol.json"
    value = read(path)
    if sources(SOURCE_NAMES) != value["sources"]:
        raise RuntimeError("V-V source drift")
    _, digest, _ = load_parent(smoke)
    if digest != value["vu_protocol_sha256"]:
        raise RuntimeError("Parent registration drift")
    return value, sha256(path), lane
