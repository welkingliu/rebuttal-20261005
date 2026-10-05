"""Record recipe differences without tuning against already observed test scores."""
import json

import yaml

from common import ROOT, OLD, atomic_json, ensure_storage, sha256
from native_runtime import configure, assets


def flatten(value, prefix=""):
    out = {}
    for k, v in value.items():
        name = prefix + str(k)
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        else:
            out[name] = v
    return out


def main():
    ensure_storage()
    out = ROOT / "results/R8_transformer_recipe"
    _, checkpoint, _, _, _ = assets("transformer", "sgdet")
    saved = checkpoint.parent / "config.yml"
    train = yaml.safe_load(saved.read_text())
    runtime = yaml.safe_load(configure("transformer", "sgdet", out).dump())
    a, b = flatten(train), flatten(runtime)
    differences = {k: dict(training=a.get(k), current=b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    source = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch"
    metrics = source / "METRICS.md"
    readme = source / "README.md"
    metrics_lines = [{"line": i, "text": line} for i, line in enumerate(metrics.read_text().splitlines(), 1)
                     if "Transformer |" in line or "X-101-FPN backbone" in line]
    recipe_lines = [{"line": i, "text": line} for i, line in enumerate(readme.read_text().splitlines(), 1)
                    if "Transformer Model" in line]
    # These are documented reference-recipe values, not a new search space.
    reference = {"SOLVER.BASE_LR": .001, "SOLVER.MAX_ITER": 16000, "SOLVER.IMS_PER_BATCH": 16,
                 "SOLVER.STEPS": [10000, 16000], "SOLVER.SCHEDULE.TYPE": "WarmupMultiStepLR"}
    comparison = {k: dict(saved_training=a.get(k), documented_reference=v) for k, v in reference.items()}
    ref_samples = reference["SOLVER.MAX_ITER"] * reference["SOLVER.IMS_PER_BATCH"]
    actual_samples = a["SOLVER.MAX_ITER"] * a["SOLVER.IMS_PER_BATCH"]
    atomic_json(out / "summary.json", dict(status="complete", model="SGG Transformer", task="SGDet",
        checkpoint_sha256=sha256(checkpoint), training_config_sha256=sha256(saved),
        reference_files={str(p): sha256(p) for p in [metrics, readme]}, reference_metric_lines=metrics_lines,
        reference_recipe_lines=recipe_lines, reference_recipe_comparison=comparison,
        scheduled_image_exposures=dict(reference=ref_samples, current=actual_samples,
                                      fraction=actual_samples / ref_samples, actual_completed_iterations_not_verified=True),
        training_vs_current_runtime_differences=differences,
        interpretation="Self-trained PySGG implementation; not the reference repository's released Transformer checkpoint. Recipe differences are documented, but not demonstrated to cause the SGDet score gap.",
        actions=["Separate controlled reimplementation from successful reference reproduction.",
                 "Do not alter test thresholds or select inference settings to recover a target number.",
                 "Use independently reference-validated TDE/SGTR evidence for primary intervention claims."]))
    print(json.dumps(dict(status="complete", scheduled_image_exposures=[actual_samples, ref_samples],
                          runtime_differences=list(differences))), flush=True)


if __name__ == "__main__":
    main()
