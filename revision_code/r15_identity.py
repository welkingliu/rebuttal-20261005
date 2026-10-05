"""R15: R12 identity diagnostic under the reference-validated all-pair protocol."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

from common import ROOT, atomic_json, ensure_storage, sha256
import sgdet_identity as engine
from sgdet_identity_protocol import SPEC as ORIGINAL_SPEC
from reproduction_pair_audit import ChunkUnion

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/R15_transformer_identity"
BASE = ROOT / "results/R14_reference_sgdet_test"
MANIFEST = ROOT / "manifests/R15_identity.json"
SPEC = copy.deepcopy(ORIGINAL_SPEC)
SPEC.update(version="r15_transformer_all_pairs_v1", families=["transformer"],
            candidate_rule="All non-self native proposal pairs, identical to R14",
            union_chunk=256, cache_probability_tolerance=1e-5,
            reference="R14 full 26446-image SGDet, unchanged absolute 2-pp tolerance",
            selection="Same deterministic R12 image selection for transparent protocol comparison")


def reference_gate(family="transformer"):
    if family != "transformer":
        raise ValueError("R15 is a Transformer-only follow-up")
    row = json.loads((BASE / "summary.json").read_text())
    checks = row.get("reference_checks", {})
    passed = (row.get("status") == "complete" and row.get("images") == 26446
              and row.get("reproduction_passed") is True and bool(checks)
              and all(x["passes"] and abs(x["delta"]) <= .02 for x in checks.values()))
    return dict(passed=passed, path=str(BASE / "summary.json"),
                sha256=sha256(BASE / "summary.json"),
                reason="corrected_all_pair_reference_passed" if passed else "reference_failed")


def fingerprint():
    names = ["r15_identity.py", "r15_queue.py", "sgdet_identity.py", "sgdet_identity_protocol.py",
             "identity_intervention.py", "identity_evidence.py", "native_runtime.py",
             "repro_experiment.py", "repro_protocol.py", "reproduction_pair_audit.py", "common.py"]
    return dict(spec=SPEC, sources={n: sha256(HERE / n) for n in names},
                reference=reference_gate(), baseline_protocol_sha256=sha256(BASE / "protocol.json"))


def register():
    value = fingerprint()
    if not value["reference"]["passed"]:
        raise RuntimeError("R15 reference gate failed")
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != value:
        raise RuntimeError("R15 already registered differently")
    atomic_json(MANIFEST, value)


def verify():
    if json.loads(MANIFEST.read_text()) != fingerprint():
        raise RuntimeError("R15 source or baseline changed")
    return sha256(MANIFEST)


original_load = engine.load
original_infer = engine.infer


def load(family, out):
    model, cfg, transform, provenance = original_load(family, out)
    baseline = json.loads((BASE / "model_provenance.json").read_text())
    if provenance["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise RuntimeError("R15 does not use the R14 checkpoint")
    model.roi_heads.relation.samp_processor.max_proposal_pairs = 1000000
    model._r15_chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
    model._r15_output = out
    return model, cfg, transform, provenance


def checked_infer(model, cfg, transform, ds, index):
    pred, target = original_infer(model, cfg, transform, ds, index)
    iid = engine.image_id(ds, index)
    candidates = [BASE / "predictions" / (iid + ".npz"),
                  ROOT / "results/R14_pair_cap/sgdet/predictions/reference_all_pairs" / (iid + ".npz")]
    paths = [p for p in candidates if p.exists()]
    if len(paths) != 1:
        raise RuntimeError("Missing or ambiguous R14 baseline cache: " + iid)
    from export_pysgg_vg_task import convert_prediction
    current = convert_prediction(pred.to("cpu"))
    errors = {}
    with np.load(paths[0], allow_pickle=False) as z:
        for key, value in current.items():
            reference = z[key]
            if value.shape != reference.shape:
                raise RuntimeError("Baseline shape mismatch: " + key)
            if np.issubdtype(value.dtype, np.integer):
                valid = np.array_equal(value, reference)
            else:
                valid = np.allclose(value, reference, atol=SPEC["cache_probability_tolerance"], rtol=1e-5)
            if not valid:
                raise RuntimeError("R15 baseline differs from R14: " + key)
            errors[key] = float(np.max(np.abs(value - reference))) if value.size else 0.
    atomic_json(model._r15_output / "baseline_checks" / (iid + ".json"),
                dict(image_id=iid, cache=str(paths[0]), max_errors=errors, passed=True))
    return pred, target


def main():
    ensure_storage()
    if sys.argv[1:] == ["--register"]:
        register()
        print("[REGISTERED] R15; old R12 outputs remain unchanged", flush=True)
        return
    engine.OUT, engine.SPEC = OUT, SPEC
    engine.verify, engine.reference_gate = verify, reference_gate
    engine.load, engine.infer = load, checked_infer
    engine.main()
    stage = sys.argv[sys.argv.index("--stage") + 1]
    path = OUT / "transformer" / stage / "summary.json"
    row = json.loads(path.read_text())
    row["experiment"] = "R15"
    row["baseline_cache_checked"] = True
    if stage == "test":
        gate = OUT / "transformer/dev/summary.json"
        row["development_gate_passed"] = json.loads(gate.read_text())["development_gate_passed"]
        row["development_gate_sha256"] = sha256(gate)
    elif stage != "dev":
        row["development_gate_passed"] = None
    atomic_json(path, row)


if __name__ == "__main__":
    main()
