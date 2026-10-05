"""Bounded V-E pilot: one family, an active shared-context update, fixed gates."""
import copy
import json
from pathlib import Path

from common import ROOT, atomic_json, sha256
from vc_protocol import SPEC as PREVIOUS_SPEC

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VE"
MANIFEST = ROOT / "manifests/VE_protocol.json"
SPEC = dict(version="ve_transformer_shared_context_v1", family="transformer",
    tasks=["sgcls", "sgdet"], modes=["supervised", "relation_aware"], seeds=[17, 23, 31], pilot_seed=17,
    train_images=5000, development_images=500, gate_images=1000,
    min_epochs=3, max_epochs=5, patience=1, learning_rate=0.0001, weight_decay=0.0001,
    image_batch_size=1, accumulate_images=4, gradient_clip=1.0, background_weight=0.25,
    adapter=dict(width=512, hidden=64, max_relative_coordinate_change=.1,
                 location="context_obj output before both out_obj and edge context",
                 formula="x + 0.1 * detached_rms(x) * tanh(up(gelu(down(layer_norm(x)))))",
                 initialization="zero up projection; seeded down projection"),
    objective=dict(supervised="object CE only", relation_aware="object CE + object KL + predicate KL + pair-score KL",
                   weights=dict(object_kl=1., predicate_kl=1., pair_score_kl=1.),
                   teacher="frozen task-matched native predictor on the same proposals, ROI and union features",
                   score="pre-NMS pair log-score = max foreground logp(subject) + max foreground logp(object) + max foreground logp(predicate); softmax over pairs",
                   reduction="per-image CE/KL means, then equal image accumulation; not official pretraining"),
    train_policy="Native model frozen/eval; only adapter trained; no image augmentation; no test selection",
    checkpoint_selection="lowest development foreground NLL, evaluated every epoch; gate not used to select epoch",
    split_policy="reuse VB training/development; fresh adaptation gate excludes VB and VC development/gate IDs",
    gate_scope="Not used for adapter fitting or checkpoint selection; native baseline audits previously evaluated the VG validation split",
    candidate_rule=dict(sgcls="retain audited native 2048 cap", sgdet="reference-defined all non-self pairs; chunk union extraction 256"),
    interpretation="shared object-context adaptation, NOT an object-classifier-only update or pure identity intervention",
    stop_rule="One fixed relation-aware candidate, seed17; both tasks must pass unchanged gates, else no extra seeds/test/search",
    formal_policy="Only if pilot passes: seeds23/31 both modes, then report all seeds/modes without selecting favourable ones")
SPEC["gate"] = copy.deepcopy(PREVIOUS_SPEC["gate"])
SPEC["gate"].update(candidate="relation_aware_seed17")


def baseline(task):
    relative = "R14_reference_sgdet_test" if task == "sgdet" else "R9_reproduction/sgcls/eval_corrected/test"
    return ROOT / "results" / relative


def registration():
    names = ["ve_protocol.py", "ve_native.py", "ve_experiment.py", "ve_queue.py", "common.py",
             "repro_experiment.py", "repro_protocol.py", "native_runtime.py", "reproduction_pair_audit.py",
             "sgdet_identity.py", "sgdet_identity_protocol.py", "identity_intervention.py", "identity_evidence.py",
             "vb_native.py", "vb_protocol.py", "vc_protocol.py"]
    refs = {}
    for task in SPEC["tasks"]:
        p = baseline(task) / "summary.json"
        row = json.loads(p.read_text())
        if row.get("status") != "complete" or row.get("images") != 26446 or row.get("reproduction_passed") is not True:
            raise RuntimeError("V-E reference gate failed for " + task)
        refs[task] = dict(summary_sha256=sha256(p),
                          model_sha256=sha256(baseline(task) / "model_provenance.json"))
    return dict(spec=SPEC, sources={n: sha256(HERE / n) for n in names}, reference=refs)


def register():
    value = registration()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != value:
        raise RuntimeError("V-E already registered differently")
    atomic_json(MANIFEST, value)


def verify():
    if json.loads(MANIFEST.read_text()) != registration():
        raise RuntimeError("V-E registered code or reference changed")
    return sha256(MANIFEST)
