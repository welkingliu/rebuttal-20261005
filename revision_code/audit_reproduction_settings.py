"""Read-only, tensor- and source-level audit of R9 against its stated reference."""
import ast
import json
from pathlib import Path
import runpy

import torch
import yaml

from common import OLD, ROOT, atomic_json, ensure_storage, sha256
from native_runtime import assets
from repro_protocol import RUN, SPEC, verify


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def normalized_state(path):
    state = torch.load(str(path), map_location="cpu")["model"]
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def compare_tensors(a, b):
    identical, different, missing = [], [], []
    for key, value in a.items():
        if key not in b:
            missing.append(key)
        elif value.shape == b[key].shape and torch.equal(value.float(), b[key].float()):
            identical.append(key)
        else:
            different.append(key)
    return dict(total=len(a), identical=len(identical), different=different, missing=missing)


def selected_class(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name),
                    include_attributes=False)


def main():
    ensure_storage()
    verify()
    out = ROOT / "results/R14_reproduction_audit"
    reference = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch"
    local = OLD / "external/official_repos/PySGG/pysgg"
    provenance = json.loads((RUN / "sgdet/training/model_provenance.json").read_text())
    actual = yaml.safe_load(provenance["config"])
    ref_cfg = runpy.run_path(str(reference / "maskrcnn_benchmark/config/defaults.py"))["_C"].clone()
    ref_cfg.merge_from_file(str(reference / "configs/e2e_relation_X_101_32_8_FPN_1x.yaml"))
    ref_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
    ref_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    ref_cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = "TransformerPredictor"
    ref_cfg.SOLVER.BASE_LR = .001
    ref_cfg.SOLVER.IMS_PER_BATCH = 16
    ref_cfg.SOLVER.MAX_ITER = 16000
    ref_cfg.SOLVER.STEPS = (10000, 16000)
    ref_cfg.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"
    a, b = flatten(actual), flatten(yaml.safe_load(ref_cfg.dump()))
    relevant = ("MODEL.", "INPUT.", "SOLVER.", "TEST.", "DATASETS.")
    different = {k: dict(r9=a[k], reference=b[k]) for k in sorted(a.keys() & b.keys())
                 if a[k] != b[k] and k.startswith(relevant)}
    extra = {k: a[k] for k in sorted(a.keys() - b.keys()) if k.startswith(relevant)}
    detector = OLD / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"
    official = assets("tde_motifs", "sgdet")[1]
    weights = normalized_state(detector)
    detector_comparison = compare_tensors(weights, normalized_state(official))
    final = RUN / "sgdet/training/model_final.pth"
    trained = normalized_state(final)
    detector_frozen = compare_tensors(weights, trained)
    batchnorm_updates = {k: int(v) for k, v in trained.items()
                         if k.startswith("roi_heads.relation.") and k.endswith("num_batches_tracked")}
    model_classes = ["ScaledDotProductAttention", "MultiHeadAttention", "PositionwiseFeedForward", "EncoderLayer", "TransformerEncoder", "TransformerContext"]
    original_transformer = reference / "maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py"
    native_transformer = local / "modeling/roi_heads/relation_head/model_transformer.py"
    structure = {}
    for name in model_classes:
        try:
            structure[name] = selected_class(original_transformer, name) == selected_class(native_transformer, name)
        except StopIteration:
            structure[name] = "class_not_found"
    structure["TransformerPredictor"] = selected_class(
        reference / "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py", "TransformerPredictor") == selected_class(
        local / "modeling/roi_heads/relation_head/roi_relation_predictors.py", "TransformerPredictor")
    # Confirm reference instructions and runtime optimizers both include batch scaling.
    original_trainer = reference / "tools/relation_train_net.py"
    has_ref_lr_scale = "rl_factor=float(num_batch)" in original_trainer.read_text()
    sampler = local / "modeling/roi_heads/relation_head/sampling.py"
    reference_sampler = reference / "maskrcnn_benchmark/modeling/roi_heads/relation_head/sampling.py"
    result = dict(status="complete", config_shared_key_differences=different, extra_pysgg_settings=extra,
                  transformer_class_ast_equal=structure,
                  detector_vs_official_causal_checkpoint=detector_comparison,
                  detector_frozen_through_r9_training=detector_frozen,
                  relation_batchnorm_update_counts=batchnorm_updates,
                  reference_expected_batchnorm_updates_per_rank=16000,
                  initialization_note="Source filenames differ, but tensor equality is the evidential test",
                  learning_rate=dict(base=.001, multiplier=16, peak=.016, reference_batch_multiplier_confirmed=has_ref_lr_scale),
                  known_protocol_differences=[
                      dict(name="pre_predictor_pair_cap", r9=2048, reference="all non-self pairs", stage="inference", action="R14 paired validation audit"),
                      dict(name="negative_relation_sampling", r9="highest object-score products up to 2*n_negative, then random sample",
                           reference="random sample from all eligible negatives", stage="training"),
                      dict(name="actual_image_microbatch", r9="1 image/GPU, accumulate 8, 2 GPUs", reference="8 images/GPU, 2 GPUs for batch16",
                           consequences=["per-image versus per-proposal loss weighting", "BatchNorm statistics differ", "padding/grouped image sampling may differ"], stage="training"),
                      dict(name="zero_shot_seen_set", r9="reconstructed from all VG split0 including validation annotations",
                           reference="released zeroshot_triplet.pytorch", stage="evaluation", action="compare exact triplet sets before claiming reference zR")],
                  training_protocol=json.loads((RUN / "sgdet/training/protocol.json").read_text()),
                  gradient_check=json.loads((RUN / "sgdet/training/gradient_check.json").read_text()),
                  training_summary=json.loads((RUN / "sgdet/training/summary.json").read_text()),
                  source_hashes={str(p): sha256(p) for p in [Path(__file__).resolve(), sampler, reference_sampler,
                      original_transformer, native_transformer, original_trainer]},
                  next_step="Measure validation-only inference differences before authorizing any additional training recipe")
    atomic_json(out / "settings.json", result)
    print(json.dumps({k: result[k] for k in ["status", "config_shared_key_differences", "transformer_class_ast_equal",
                      "detector_vs_official_causal_checkpoint", "detector_frozen_through_r9_training", "learning_rate"]}), flush=True)


if __name__ == "__main__":
    main()
