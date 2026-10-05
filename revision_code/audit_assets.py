"""R0/R4: immutable asset, gradient-path, and metric-contract audits."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import ROOT, OLD, ensure_storage, atomic_json, sha256, canonical_path


def gradient_audit():
    from sgg_core.models.adapters.pysgg_live import PySGGLiveAdapter
    from sgg_core.mitigation.grounding_regularizer import GroundingDependencyRegularizer
    # Bypass native inference only. Exercise the actual deployed calibration
    # and regularizer methods with frozen synthetic worker outputs.
    adapter = PySGGLiveAdapter.__new__(PySGGLiveAdapter)
    nn.Module.__init__(adapter)
    adapter.relation_calibrator = nn.Linear(51,51)
    adapter.entity_calibrator = nn.Linear(151,151)
    with torch.no_grad():
        for module in (adapter.relation_calibrator,adapter.entity_calibrator):
            module.weight.copy_(torch.eye(module.weight.shape[0]))
            module.bias.zero_()
    for p in adapter.relation_calibrator.parameters():
        p.requires_grad_(False)
    torch.manual_seed(17)
    raw = dict(pred_rel_scores=torch.randn(19,51), pred_entity_scores=torch.randn(12,151))
    results = []
    weights = ["mild_weight","dependency_weight","uncertainty_weight","object_weight",
               "object_consistency_weight","object_calibration_weight","object_margin_weight"]
    for term in ["relation_supervision"] + weights:
        active = {key:0. for key in weights}
        if term != "relation_supervision":
            active[term]=1.
        regularizer = GroundingDependencyRegularizer(**active)
        out = adapter._calibrate(raw)
        loss = regularizer(out["pred_rel_scores"], out["pred_rel_scores"]+.01,
                           out["pred_rel_scores"]*.1, torch.arange(19)%50+1,
                           object_logits=out["pred_entity_scores"],object_targets=torch.arange(12)+1)["loss"]
        gradients = torch.autograd.grad(loss, list(adapter.entity_calibrator.parameters()),
                                      allow_unused=True) if loss.requires_grad else [None,None]
        norm = sum(float(g.square().sum()) for g in gradients if g is not None) ** .5
        results.append(dict(term=term, object_parameter_gradient_norm=norm,
                            loss_requires_grad=loss.requires_grad, nonzero=norm>0))
    # __del__ on some historical adapters expects worker-related attributes.
    adapter._worker = None
    return dict(test_kind="structural_synthetic_frozen_worker_outputs", updated_object_parameters=22952,
                terms=results, absent_mask_logits=True,
                caveat="No actual training run is reproduced by this structural test")


def main():
    ensure_storage()
    torch.set_num_threads(4)
    assets=[]
    for name in ["pysgg_transformer_vg_live.json","pysgg_tde_motifs_vg_live.json"]:
        path = OLD/"checkpoints/sgg/manifests"/name
        d=json.loads(path.read_text())
        for task,item in d["checkpoints"].items():
            p=canonical_path(item["path"])
            assets.append(dict(manifest=name,task=task,path=str(p),exists=p.is_file(),
                               bytes=p.stat().st_size if p.exists() else None,declared_sha256=item["sha256"]))
    caches=[]
    for name in ["causal_motifs_sum_tde_vg_entity_aligned","pysgg_transformer_object_refine_fixed_20260724"]:
        root=OLD/"artifacts/prediction_cache"/name
        for task in ["predcls","sgcls","sgdet"]:
            files=sorted(p for p in (root/"predictions"/task).glob("*.npz") if not p.name.startswith("._"))
            with np.load(files[0]) as z:
                x=z["pred_entity_scores"]
                caches.append(dict(cache=name,task=task,images=len(files),sample=files[0].name,
                                   mean_nonzero_classes=float(np.count_nonzero(x,axis=1).mean()),
                                   full_object_distribution_available=bool(np.median(np.count_nonzero(x,axis=1))>2)))
    files=[OLD/"sgg_core/models/adapters/pysgg_live.py",OLD/"sgg_core/mitigation/grounding_regularizer.py",
           OLD/"sgg_core/audits/perturbation_sweep.py",OLD/"sgg_core/audits/graph_audit.py"]
    atomic_json(ROOT/"results/R0/audit.json",dict(status="complete",assets=assets,caches=caches,
        source_sha256={str(p):sha256(p) for p in files},gradients=gradient_audit(),
        findings=["Old Transformer live manifest selects the uncorrected SGCls checkpoint; R2 uses the explicitly hashed corrected checkpoint.",
                  "Sparse top-class caches are insufficient to apply a dense-logit-trained affine calibrator consistently. R3 re-exports dense logits.",
                  "No historical result file was modified."]))
    from sgg_core.data.data_utils import build_vg_test_loader
    ds=build_vg_test_loader(str(OLD/"data/vg/v1.4"),6000,split=0,
           include_proxy_features=False,include_raw_images=False).dataset
    train_ids=[str(ds.index_to_image_meta[int(i)]["image_id"]) for i in ds.image_indices[:5000]]
    val_ids=[str(ds.index_to_image_meta[int(i)]["image_id"]) for i in ds.image_indices[5000:6000]]
    assert len(val_ids)==1000 and not set(train_ids)&set(val_ids)
    atomic_json(ROOT/"manifests/R3_validation_ids.json",dict(training_image_ids=train_ids,
        validation_image_ids=val_ids,protocol="original Experiment V first 5000/next 1000 eligible split-0 images"))
    summaries=[]
    for path in sorted((ROOT/"imported").rglob("experiment_*.json")):
        d=json.loads(path.read_text())
        summaries.append(dict(path=str(path),sha256=sha256(path),experiment=d.get("experiment"),
                              contains_per_image_raw_records=any(k in d for k in ("per_image","records","image_records"))))
    atomic_json(ROOT/"results/R4/contract_audit.json",dict(status="audit_complete",sources=summaries,
        background_policy=dict(II_Hit="foreground only", II_calibration="includes background",III="includes background"),
        paired_reanalysis_status="requires per-image records or rerun; aggregate CI endpoints cannot reconstruct paired CI",
        conclusion="The low III clean accuracy is not comparable to II foreground Hit@1 without aligning background policy, checkpoint and selected pairs.",
        new_record_policy="R2 stores clean logits, foreground predictions, GT pairs and per-image conditions."))
    print("[COMPLETE] R0 and R4 contract audit",flush=True)


if __name__=="__main__":
    main()
