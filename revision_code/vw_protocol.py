"""V-W immutable train-only protocol, independent of old fitted visual experts."""
from pathlib import Path
from common import ROOT, sha256
from evidence_completion import read, sources, lock
from vv_protocol import load as load_parent, SOURCE_NAMES as PARENT_SOURCES
from vw_math import SPEC, PRIMARY, ARMS

OUT = ROOT / "results/VW_endpoint_relation"
WEIGHTS = ROOT / "checkpoints/VW_endpoint_relation"
SOURCE_NAMES = sorted(set(PARENT_SOURCES + ["vw_math.py", "vw_protocol.py", "vw_experiment.py",
    "vw_queue.py", "test_vw_endpoint.py", "VW_ENDPOINT_PLAN.txt", "r19_proposal_sensitivity.py",
    "audit_identity_metric_history.py"]))


def register(smoke=False):
    parent, parent_sha, lane = load_parent(smoke)
    value = {k: parent[k] for k in ["image_ids", "folds", "vn_protocol_sha256", "native_checkpoint_sha256",
                                   "vu_protocol_sha256", "bundle_sha256", "bundle_path"]}
    value.update(version="vw_endpoint_relation_v1", smoke=smoke, vv_protocol_sha256=parent_sha,
        spec=SPEC, primary=PRIMARY, arms=ARMS, sources=sources(SOURCE_NAMES),
        plan=Path(__file__).with_name("VW_ENDPOINT_PLAN.txt").read_text(),
        validation_accessed=False, test_accessed=False, formal_gate_accepted=False,
        independent_confirmation=False, reused_train_images=True, native_sgg_trained_on_these_images=True,
        auxiliary_expert_checkpoint_used=False, all_new_head_selection_nested=True,
        confirmation_policy="Exposure audit and separate registration required; no automatic gate/test dispatch")
    return value, lock(OUT / lane / "protocol.json", value), lane


def load(smoke=False):
    lane = "smoke" if smoke else "train3000"
    path = OUT / lane / "protocol.json"
    value = read(path)
    if value["sources"] != sources(SOURCE_NAMES):
        raise RuntimeError("V-W source drift")
    _, parent_sha, _ = load_parent(smoke)
    if parent_sha != value["vv_protocol_sha256"]:
        raise RuntimeError("V-W parent drift")
    return value, sha256(path), lane
