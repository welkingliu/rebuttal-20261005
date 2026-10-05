"""Finite follow-up registration, separate from the running V-C protocol."""
import json
from pathlib import Path

from common import ROOT, atomic_json, sha256

SPEC = {
    "version": "review_extension_20260929_v1",
    "R6": {"model": "SGTR", "task": "SGDet", "candidate_images": 2000,
           "selection": "same frozen R4 candidate list; highest degree versus area-matched degree-zero",
           "strengths": [0.25, 0.5, 1.0], "Ks": [1, 5, 10, 20, 50, 100],
           "interpretation": "end-to-end spatial evidence sensitivity, not identity-specific causation",
           "minimum_images": 100, "bootstrap": 2000,
           "integration": "clean predictions agree with released cache on three images before intervention"},
    "R7": {"datasets": ["gqa", "vrd"], "split": "all validation/test annotations",
           "mapping": "normalized exact names only, no outcome-driven synonym expansion",
           "estimand": "ontology eligibility, not model accuracy or missing-label negatives"},
    "VD": {"family": "tde_motifs", "tasks": ["sgcls", "sgdet"], "seed": 17,
           "heads": ["conservative", "supervised"], "alphas": [0.05, 0.1, 0.2, 0.4, 0.6],
           "direction": "validation-selected shrinkage of the V-C visual residual",
           "selection": "rank ten candidates by cached development foreground accuracy; at most two finalists",
           "cached_min_gain": 0.002, "full_development_images": 500, "fresh_gate_images": 1000,
           "development_rule": "post-NMS identity gain >=0.005; R50 and mR50 point deltas >=-0.005",
           "gate_rule": "unchanged V-C gate, both tasks; paired image one-sided 95% bounds",
           "split_policy": "fresh full-system development and gate exclude every VB/VC development/gate image",
           "stop": "no finalist or failed gate stops; no test, more seeds or adaptive alpha search",
           "claim": "exploratory validation result only; passing does not automatically establish a formal test gain"},
}


def hashes():
    folder = Path(__file__).resolve().parent
    names = ["extension_protocol.py", "extension_queue.py", "modern_live_control.py",
             "external_coverage.py", "vd_experiment.py", "test_extensions.py",
             "common.py", "native_runtime.py", "vc_experiment.py", "vc_native.py",
             "vc_protocol.py", "vb_native.py", "vb_protocol.py", "paired_visual_control.py"]
    return {n: sha256(folder / n) for n in names}


def verify():
    record = json.loads((ROOT / "manifests/review_extension.json").read_text())
    if record["spec"] != SPEC or record["sources"] != hashes():
        raise RuntimeError("Extension sources/protocol changed after registration")
    return record


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("Refusing incompatible resume: " + str(path))
    atomic_json(path, value)
