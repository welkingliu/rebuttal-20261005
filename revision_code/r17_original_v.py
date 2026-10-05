"""Real-input audit of the released Experiment V objective, not historical replay."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import ROOT, OLD, atomic_json, ensure_storage, sha256

OUT = ROOT / "results/R17_original_v"
WEIGHTS = ["mild_weight", "dependency_weight", "uncertainty_weight", "object_weight",
           "object_consistency_weight", "object_calibration_weight", "object_margin_weight"]


def gradients(loss, parameters):
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters), device=parameters[0].device)
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, values)])


def main():
    ensure_storage()
    torch.set_num_threads(4)
    from sgg_core.models.adapters.pysgg_live import PySGGLiveAdapter
    from sgg_core.mitigation.grounding_regularizer import GroundingDependencyRegularizer
    from sgg_core.mitigation.run_mitigation import _deduplicate_training_relations, _task_object_weights
    from sgg_core.audits.perturbation_sweep import VisualPerturbation
    from sgg_core.data.data_utils import build_vg_test_loader

    sources = [OLD / "sgg_core/models/adapters/pysgg_live.py",
               OLD / "sgg_core/mitigation/grounding_regularizer.py",
               OLD / "sgg_core/mitigation/run_mitigation.py",
               ROOT / "imported/anonymous_release_20260929/scripts/run_paper_experiment5_tde_motifs.py"]
    protocol = dict(images=64, seeds=[17,23,31], modes=["supervised_control","grounding"],
                    source_sha256={str(p):sha256(p) for p in sources}, code_sha256=sha256(Path(__file__)),
                    interpretation="Real-input gradient audit at saved states; not a replay of historical optimizer steps",
                    object_weight=2.25, calibration_weight=.05, mask_logits=False,
                    task_object_weight=3., object_margin_weight=0., weight_decay=.0001)
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("R17 protocol changed")
    atomic_json(path, protocol)
    manifest = json.loads((OLD / "checkpoints/sgg/manifests/pysgg_tde_motifs_vg_live.json").read_text())
    model = PySGGLiveAdapter({k:v["path"] for k,v in manifest["checkpoints"].items()},
                            torch.device("cuda"), manifest["config"], "sgcls")
    for p in model.relation_calibrator.parameters():
        p.requires_grad_(False)
    model.eval()
    params = list(model.entity_calibrator.parameters())
    loader = build_vg_test_loader(str(OLD / "data/vg/v1.4"), 64, split=0,
                                 include_proxy_features=False, include_raw_images=True)
    predicate_ids = [7,11,14,19,21,22,24,26,28,31,32,35,38,40,41,43,44,45,46,48,49]
    perturb = VisualPerturbation(noise_std=1.)
    rows = []
    history_checks = []
    saved = {}
    for mode in protocol["modes"]:
        for seed in protocol["seeds"]:
            base = ROOT / "imported/experiment5" / mode / "TDE_Neural_Motifs" / ("seed_"+str(seed))
            result = json.loads((base / "mitigation_results.json").read_text())
            state = torch.load(base / "mitigated_state_dict.pth", map_location="cpu", weights_only=False)
            actual = state["training_args"]
            for k,v in dict(object_weight=2.25, object_calibration_weight=.05,
                            object_margin_weight=0., object_focal_gamma=0.,
                            freeze_relation_parameters=True, task_object_weight=3.).items():
                if actual[k] != v:
                    raise RuntimeError("Historical coefficient mismatch: "+k)
            saved[mode,seed] = state["grounding_state_dict"]
            atomic_json(OUT / mode / ("seed_"+str(seed)+"_training_args.json"),actual)
            for h in result["history"]:
                expected = h["supervised"] + 2.25*h["object_supervised"]
                if mode == "grounding":
                    expected += .5*(h["mild_consistency"]+h["dependency_penalty"]+h["ablated_uncertainty"])
                    expected += .25*h["object_consistency"] + .05*h["object_calibration_proxy"]
                history_checks.append(dict(mode=mode, seed=seed, epoch=h["epoch"], recorded=h["loss"],
                    reconstructed=expected, absolute_error=abs(expected-h["loss"]),
                    object_only=h["optimized_parameter_groups"]==["object"],
                    checkpoint_sha256=sha256(base / "mitigated_state_dict.pth")))
    try:
        for index, raw in enumerate(loader):
            for seed in protocol["seeds"]:
                raw_batch, _ = _deduplicate_training_relations(raw, seed+index)
                batch = {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in raw_batch.items()}
                weights, _ = _task_object_weights(batch, predicate_ids, 3.)
                mild = perturb.inject_visual_noise(batch, strength=.05, seed=seed+index)
                ablated = perturb.attenuate_union_features(batch, strength=1.)
                outputs = [model._live_raw(b, require_gt_pairs=True) for b in (batch,mild,ablated)]
                for mode in protocol["modes"]:
                    model.load_grounding_state_dict(saved[mode,seed])
                    clean, noise, absent = [model._calibrate(x) for x in outputs]
                    if mode == "supervised_control":
                        noise = absent = clean
                    args = (clean["pred_rel_scores"],noise["pred_rel_scores"],absent["pred_rel_scores"],batch["rel_labels"])
                    kwargs = dict(object_logits=clean["pred_entity_scores"],object_targets=batch["entity_labels"],
                                  object_sample_weights=weights,mask_object_logits=clean.get("mask_entity_scores"))
                    coefficients = dict(mild_weight=.5,dependency_weight=.5,uncertainty_weight=.5,
                                        object_weight=2.25,object_consistency_weight=.25,
                                        object_calibration_weight=.05,object_margin_weight=0.)
                    if mode == "supervised_control":
                        coefficients = {k:(v if k=="object_weight" else 0.) for k,v in coefficients.items()}
                    total = GroundingDependencyRegularizer(**coefficients)(*args,**kwargs)["loss"]
                    total_grad = gradients(total,params)
                    terms = {}
                    zero = {k:0. for k in WEIGHTS}
                    base_loss = GroundingDependencyRegularizer(**zero)(*args,**kwargs)["loss"]
                    terms["relation_supervision"] = dict(value=float(base_loss.detach()),
                                                          gradient_norm=float(gradients(base_loss,params).norm()))
                    effective = base_loss.detach()*0.
                    for term, coefficient in coefficients.items():
                        active = dict(zero, **{term:1.})
                        loss = GroundingDependencyRegularizer(**active)(*args,**kwargs)["loss"]-base_loss
                        g = gradients(loss,params)
                        terms[term] = dict(coefficient=coefficient,value=float(loss.detach()),
                                           gradient_norm=float(g.norm()),weighted_gradient_norm=float((coefficient*g).norm()))
                        if term in ("object_weight","object_calibration_weight"):
                            effective = effective + coefficient*loss
                    effective_grad = gradients(effective,params)
                    error = float((total_grad-effective_grad).abs().max())
                    if error > 1e-5 or not torch.isfinite(total_grad).all():
                        raise RuntimeError("Effective-objective gradient equivalence failed")
                    rows.append(dict(image_index=index, image_id=str(raw.get("image_id",index)),seed=seed,mode=mode,
                        objects=int(batch["entity_labels"].numel()),relations=int(batch["rel_labels"].numel()),
                        terms=terms, effective_gradient_max_error=error,
                        worker_outputs_have_grad=any(x.requires_grad for o in outputs for x in o.values())))
            atomic_json(OUT / "records.json",rows)
            progress=dict(images=index+1,total=64,records=len(rows))
            atomic_json(OUT / "progress.json",progress)
            print(json.dumps(progress),flush=True)
    finally:
        model.close()
    atomic_json(OUT / "summary.json",dict(status="complete",protocol=protocol,records=len(rows),
        history_checks=history_checks, max_history_reconstruction_error=max(x["absolute_error"] for x in history_checks),
        max_gradient_equivalence_error=max(x["effective_gradient_max_error"] for x in rows),
        findings=["Only the shared 151x151 object affine readout receives updates (22952 parameters).",
                  "The effective grounding gradient is 2.25 * task-weighted object CE + 0.05 * object Brier.",
                  "Relation supervision, relation consistency, dependency and confidence penalties have zero object gradient.",
                  "Mask consistency is absent; the logged margin is multiplied by zero.",
                  "No end-to-end grounding or relation-protective training gradient is implemented in this original run."] ))


if __name__ == "__main__":
    main()
