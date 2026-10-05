"""R12 frozen protocol: detected-proposal identity paths, not an SGDet repair."""
import json
from pathlib import Path

from common import ROOT, atomic_json, sha256

OUT = ROOT/"results/R12_sgdet_identity"
SPEC = dict(families=["tde_motifs", "transformer"], task="SGDet-conditioned predictor diagnostic",
    samples=dict(smoke=3, validation=1000, dev=100, test=2000), seeds=[17,23,31], strengths=[.25,.5,1.],
    matching="Class-blind maximum-cardinality one-to-one IoU>=0.5, then maximum inclusive-pixel IoU",
    interventions=["clean", "oracle", "oracle_frequency", "oracle_both", "confusable", "matched_random"],
    routes=["embedding", "frequency", "both"],
    corruption="Only matched proposals whose consumed identity is initially correct; same selected nodes in each route",
    oracle="Correct only geometry-matched proposals; unmatched proposals remain native",
    reference="Require full native SGDet reproduction using the existing registered tolerance, never tune on test",
    held_fixed=["ROI features", "union features", "proposal geometry", "native candidate relation pairs"],
    proposal_stage="Native detector proposals at relation-predictor input, before relation-head object refinement/NMS",
    identity_scope="Classes consumed by the predictor's object-identity channel, not assumed equal to final emitted labels",
    denominator="All GT relations reported; predicate metrics only on pairs with matched endpoints and an existing native candidate",
    frequency_route="Preserve posterior probability multiset while swapping native/replacement class coordinates",
    TDE_average_context="unchanged", background_policy="Foreground predicate argmax, ID 0 excluded",
    dev_gate=dict(images=100, minimum_evaluable_images=20, minimum_relations=50),
    interpretation="Internal channel intervention on native detected proposals; not a learned repair or full-pipeline recall gain")


def fingerprint():
    names = ["sgdet_identity_protocol.py", "sgdet_identity.py", "identity_evidence.py", "identity_intervention.py",
             "native_runtime.py", "repro_experiment.py", "repro_protocol.py", "common.py", "sgdet_identity_queue.py"]
    files = [ROOT/"code"/name for name in names]
    files += [ROOT/"manifests/R9_reproduction.json", ROOT/"results/R1/tde_motifs/sgdet/summary.json"]
    return dict(spec=SPEC, sources={str(p): sha256(p) for p in files})


def register():
    path = ROOT/"manifests/R12_sgdet_identity.json"
    value = fingerprint()
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("R12 protocol already registered differently")
    atomic_json(path, value)


def verify():
    path = ROOT/"manifests/R12_sgdet_identity.json"
    if json.loads(path.read_text()) != fingerprint():
        raise RuntimeError("R12 registered source/protocol changed")
    return sha256(path)


def reference_gate(family):
    if family == "tde_motifs":
        path = ROOT/"results/R1/tde_motifs/sgdet/summary.json"
        row = json.loads(path.read_text())
        checks = row.get("reproduction_checks", {})
        passed = row.get("full_split") and checks and all(v["passes_existing_tolerance"] for v in checks.values())
    else:
        path = ROOT/"results/R9_reproduction/sgdet/eval_retrained/test/summary.json"
        if not path.exists():
            return dict(passed=False, reason="R9 full SGDet evaluation unavailable", path=str(path))
        row = json.loads(path.read_text())
        passed = row.get("full_test") and row.get("reproduction_passed") is True
    return dict(passed=bool(passed), path=str(path), sha256=sha256(path),
                reason="native_reference_passed" if passed else "reference_gap_not_execution_failure")
