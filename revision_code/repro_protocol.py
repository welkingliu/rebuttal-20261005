"""Registered repair of supervised object refinement; old runs remain immutable."""
import json
from pathlib import Path

from common import ROOT, OLD, sha256, atomic_json

RUN = ROOT / "results/R9_reproduction"
SPEC = dict(
    version="r9_supervised_object_refinement_v1", family="transformer", seed=666,
    sgcls="Evaluate existing object-refine-trained checkpoint with its two saved refinement flags enabled",
    sgdet="Retrain Transformer relation/context heads from frozen detector initialization, not old unsupervised object head",
    flags=dict(OBJECT_CLASSIFICATION_REFINE=True, REL_OBJ_MULTI_TASK_LOSS=True),
    training=dict(optimizer_steps=16000, image_microbatch=1, global_image_microsteps=16, gpus=2,
                  effective_images_per_update=16, base_lr=.001, optimizer_lr_factor=16,
                  momentum=.9, weight_decay=.0001, gradient_clip=5., milestones=[10000,16000],
                  warmup_steps=500, precision="float32", checkpoint_period=1000,
                  freeze=["backbone", "rpn", "roi_heads.box"], num_workers=0,
                  checkpoint_selection="fixed final iteration; no test-based checkpoint selection",
                  memory_adaptation="per-image gradient accumulation; not bitwise equivalent to batch16 proposal-weighted losses"),
    reference="Scene-Graph-Benchmark.pytorch README Transformer schedule and METRICS table; corrected PySGG implementation, not an official released Transformer checkpoint",
    detector="Keep existing shared detector for a scoped correction; preserve provenance and report initialization difference from reference",
    post_training=["full native validation", "full 26446-image test", "official R/mR/zR and provenance report"],
    evaluation_failure="record separately; never discard training checkpoint or terminate independent GPU lane",
    sgcls_fallback="Report any remaining reference gap; no automatic test-driven hyperparameter or checkpoint search",
    motifs="Retain failed original reproduction record; reference-validated TDE-Motifs remains primary classical baseline, no redundant Motifs retraining",
)


def sources():
    folder = Path(__file__).resolve().parent
    names = ["repro_protocol.py", "repro_experiment.py", "repro_queue.py", "test_reproduction.py",
             "common.py", "native_runtime.py"]
    result = {n: sha256(folder / n) for n in names}
    source = OLD / "external/official_repos/PySGG/pysgg"
    for sub in ["modeling/roi_heads/relation_head", "solver"]:
        for p in (source / sub).rglob("*.py"):
            result[str(p.relative_to(OLD))] = sha256(p)
    return result


def verify():
    record = json.loads((ROOT / "manifests/R9_reproduction.json").read_text())
    if record["spec"] != SPEC or record["sources"] != sources():
        raise RuntimeError("Reproduction protocol/source changed after registration")
    return record


def immutable(path, record):
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError("Incompatible reproduction resume: " + str(path))
    atomic_json(path, record)
