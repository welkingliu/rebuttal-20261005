"""V-X versioned train-only region adaptation; no changes to historical gates."""
import json
import os
import time
from pathlib import Path

from common import ROOT, atomic_json, output_path, sha256
from evidence_completion import read, sources, lock

NAME = "VX_region_adaptation_v2"
OUT = ROOT / "results" / NAME
CACHE = ROOT / "cache" / NAME
WEIGHTS = ROOT / "checkpoints" / NAME
HERE = Path(__file__).resolve().parent
PRIMARY = "adapt_diagnostic"
ARMS = ["native", "frozen_head", "adapt_plain", PRIMARY]
SPEC = dict(seed=17, tail_blocks=2, head_lr=.0005, tail_lr=.00001,
            epochs=5, accumulation_images=8, crop_chunk=8, clip=1.,
            weight_decay=.01, endpoint_weight=3., background_mass=.25,
            kl_weight=1., relation_weight=1., prefix_dtype="float32",
            input_view="predicted_input_proposal_tight_crop", classes=151,
            held_fold=0, refit=False, max_run_hours=24)
FILES = ["vx_common.py", "vx_math.py", "vx_native.py", "vx_vision.py",
         "vx_queue.py", "test_vx_region.py", "VX_REGION_PLAN.txt"]


def lane_for(smoke):
    return "smoke" if smoke else "train3000"


def register(smoke):
    from vu_protocol import load as parent_load
    from vp_math import split_fold
    from vw_protocol import SOURCE_NAMES
    parent, parent_sha, _ = parent_load(smoke, check_assets=True)
    split = split_fold(parent["image_ids"], parent["folds"], 0)
    deps = sorted(set(FILES + SOURCE_NAMES))
    value = dict(version=NAME, smoke=smoke, spec=SPEC, arms=ARMS, primary=PRIMARY,
                 split=split, image_ids=parent["image_ids"], inputs=parent["inputs"],
                 model_dir=parent["model_dir"], visual_assets=parent["visual_assets"],
                 vn_protocol_sha256=parent["vn_protocol_sha256"],
                 native_checkpoint_sha256=parent["native_checkpoint_sha256"],
                 parent_protocol_path=str(ROOT/"results/VU_reobservation"/lane_for(smoke)/"protocol.json"),
                 parent_protocol_sha256=parent_sha, sources=sources(deps),
                 plan=(HERE/"VX_REGION_PLAN.txt").read_text(),
                 validation_accessed=False, test_accessed=False,
                 reused_train_images=True, native_sgg_trained_on_these_images=True,
                 independent_confirmation=False, formal_gate_accepted=False)
    path = OUT/lane_for(smoke)/"protocol.json"
    return value, lock(path, value), lane_for(smoke)


def load(smoke=False):
    lane = lane_for(smoke)
    path = OUT/lane/"protocol.json"
    value = read(path)
    if sources(value["sources"]) != value["sources"]:
        raise RuntimeError("V-X source drift; preserve this run and register a new version")
    if sha256(value["parent_protocol_path"]) != value["parent_protocol_sha256"]:
        raise RuntimeError("Parent protocol drift")
    return value, sha256(path), lane


def save(path, value):
    import torch
    path = output_path(path)
    tmp = path.with_name(path.name+".tmp.%d" % os.getpid())
    torch.save(value, str(tmp), pickle_protocol=2)
    os.replace(tmp, path)


def progress(stage, label, n, total, start, **extra):
    elapsed = time.monotonic()-start
    value = dict(status="running", pid=os.getpid(), stage=label, images=n,
                 total=total, seconds=elapsed,
                 eta_seconds=elapsed/n*(total-n) if n else None, **extra)
    atomic_json(stage/"progress.json", value)
    print(json.dumps(value), flush=True)


def finish(stage, reg, n, **values):
    value = dict(status="complete", protocol_sha256=reg, images=n, **values)
    atomic_json(stage/"summary.json", value)
    atomic_json(stage/"progress.json", dict(status="complete", images=n, total=n))
    return value


def checked_done(stage, reg):
    if not (stage/"summary.json").exists():
        return False
    value = read(stage/"summary.json")
    if value["status"] != "complete" or value["protocol_sha256"] != reg:
        raise RuntimeError("Completion provenance drift: "+str(stage))
    return True
