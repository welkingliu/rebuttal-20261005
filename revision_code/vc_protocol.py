"""Locked, validation-gated V-C exploration. No test-driven model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, atomic_json, sha256


VC = ROOT / "results/VC"
KS = [1, 5, 10, 20, 50, 100]
SPEC = dict(
    version="vc_actual_proposal_logit_residual_v1", family="tde_motifs",
    tasks=["sgcls", "sgdet"], seeds=[17, 23, 31], pilot_seed=17,
    train_images=5000, development_images=500, gate_images=1000,
    max_epochs=20, patience=3, batch_size=512, learning_rate=0.001,
    weight_decay=0.0001, background_weight=0.25,
    kl_weights={"supervised": 0.0, "conservative": 1.0},
    feature="frozen 4096-D relation ROI features, layer-normalized without affine parameters",
    adapter="zero-initialized residual linear 4096-to-151 on detector proposal logits before relation predictor",
    update="only residual head; native model remains in evaluation mode and frozen",
    counterfactual_branch="unchanged; do not inject observed context into TDE average branch",
    training="cached native inference features; no relation loss or claim of relation-loss gradients",
    proposal_labels="SGDet fixed input proposal IoU>=0.5 positive, <0.3 background, otherwise ignored",
    checkpoint_selection="minimum development foreground NLL; max20/patience3; never gate/test selection",
    temperature="positive scalar fitted on development only, one per task",
    gate=dict(object_gain_min=0.005, object_bootstrap_lower_min=0.0,
              relation_noninferiority_margin=0.005, bootstrap_samples=2000,
              lower_quantile=0.05, metrics=["R@50", "mR@50"],
              minimum_positive_objects=1000,
              requires_both_tasks=True, candidate="conservative_seed17",
              policy="all conditions must hold; reject stops seeds23/31 and test, with no threshold change"),
    formal_policy="after pilot pass, retain all three seeds and both training modes; no reselection",
    endpoint="gate uses emitted post-NMS identity on the same input proposals; also report pre-NMS identity and full SGG",
    split_policy="reuse VB train/development; fresh gate excludes all VB development/gate images",
    routing="native OBJECT_CLASSIFICATION_REFINE=False retained; assert updated logits reach postprocessor every image",
    pair_policy="native pre-predictor pair selection retained; downstream context, class-specific NMS and triplet ranking rerun",
    caveat="exploratory model extension, not a new SOTA claim or proof of general grounding repair",
)


class ResidualHead(nn.Module):
    def __init__(self, width=4096, classes=151):
        super().__init__()
        self.linear = nn.Linear(width, classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features, logits):
        return logits + self.linear(F.layer_norm(features.float(), (features.shape[-1],)))


def objective(head, features, logits, targets, kl_weight):
    updated = head(features, logits)
    valid = targets >= 0
    if not valid.any():
        raise ValueError("No eligible training proposals")
    weights = logits.new_ones(logits.shape[-1])
    weights[0] = SPEC["background_weight"]
    ce = F.cross_entropy(updated[valid], targets[valid], weight=weights)
    kl = F.kl_div(F.log_softmax(updated, -1), F.softmax(logits.detach(), -1), reduction="batchmean")
    return ce + kl_weight * kl, ce, kl


def iou_pixel(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo + 1, 0).prod(-1)
    area = lambda x: np.maximum(x[:, 2:] - x[:, :2] + 1, 0).prod(-1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-12)


def proposal_targets(boxes, gt_boxes, gt_labels, task):
    if task == "sgcls":
        if boxes.shape != gt_boxes.shape or not np.allclose(boxes, gt_boxes, atol=1e-3):
            raise RuntimeError("SGCls proposals are not GT-aligned")
        return gt_labels.astype(np.int64).copy()
    overlap = iou_pixel(boxes, gt_boxes)
    matched = overlap.argmax(1)
    quality = overlap.max(1)
    target = np.full(len(boxes), -1, dtype=np.int64)
    target[quality < .3] = 0
    positive = quality >= .5
    target[positive] = gt_labels[matched[positive]]
    return target


def choose_ids(ids, n, salt):
    return sorted(ids, key=lambda x: hashlib.sha256((salt + str(x)).encode()).digest())[:n]


def protocol_hash():
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def source_hashes():
    folder=Path(__file__).resolve().parent
    files=list(folder.glob("vc_*.py"))+[folder/"common.py",folder/"native_runtime.py",
                                       folder/"vb_native.py",folder/"vb_protocol.py"]
    return {p.name: sha256(p) for p in sorted(files)}


def verify_registration():
    record = json.loads((ROOT / "manifests/VC_protocol.json").read_text())
    if record["spec"] != SPEC or record["source_hashes"] != source_hashes():
        raise RuntimeError("V-C registered protocol/source changed; do not continue silently")
    return record


def summarize_rows(rows):
    r = np.stack([x["recalls"] for x in rows])
    cls = np.asarray([x["class_recalls"] for x in rows], dtype=float)
    present = np.isfinite(cls)
    cm = np.divide(np.nansum(cls, axis=0), present.sum(0),
                   out=np.zeros(cls.shape[1:]), where=present.sum(0)>0)
    counts = np.asarray([[x["positive_correct"], x["positive_objects"], x["post_nms_correct"]] for x in rows])
    bins = np.sum([x["calibration_bins"] for x in rows], axis=0)
    n = int(counts[:, 1].sum())
    valid = bins[:, 0] > 0
    oc=np.sum([x.get("object_class_correct",[0]*150) for x in rows],axis=0)
    on=np.sum([x.get("object_class_count",[0]*150) for x in rows],axis=0)
    supported=on>0
    return dict(images=len(rows), object_top1=float(counts[:, 0].sum()/max(n, 1)),
                object_macro_accuracy=float((oc[supported]/on[supported]).mean()) if supported.any() else None,
                object_supported_classes=int(supported.sum()),
                post_nms_object_top1=float(counts[:, 2].sum()/max(n, 1)), positive_objects=n,
                ece=float(np.abs(bins[valid, 1]-bins[valid, 2]).sum()/max(n, 1)),
                foreground_nll=sum(x["nll_sum"] for x in rows)/max(n, 1),
                foreground_brier=sum(x["brier_sum"] for x in rows)/max(n, 1),
                R={str(k):float(r[:, i].mean()) for i,k in enumerate(KS)},
                mR={str(k):float(cm[i].mean()) for i,k in enumerate(KS)})


def paired_gate(base, changed):
    if [x["image_id"] for x in base] != [x["image_id"] for x in changed]:
        raise RuntimeError("Unpaired validation image IDs")
    n = len(base)
    bc = np.asarray([[x["post_nms_correct"], x["positive_objects"]] for x in base])
    cc = np.asarray([[x["post_nms_correct"], x["positive_objects"]] for x in changed])
    if not np.array_equal(bc[:, 1], cc[:, 1]):
        raise RuntimeError("Identity denominator changed")
    index = KS.index(50)
    br = np.asarray([x["recalls"][index] for x in base])
    cr = np.asarray([x["recalls"][index] for x in changed])
    bm = np.asarray([x["class_recalls"][index] for x in base], dtype=float)
    cm = np.asarray([x["class_recalls"][index] for x in changed], dtype=float)
    if not np.array_equal(np.isfinite(bm), np.isfinite(cm)):
        raise RuntimeError("Predicate support changed")
    def macro(a):
        count=np.isfinite(a).sum(0)
        return float(np.divide(np.nansum(a,0),count,out=np.zeros(50),where=count>0).mean())
    point=dict(object=float((cc[:,0].sum()-bc[:,0].sum())/max(bc[:,1].sum(),1)),
               R50=float((cr-br).mean()), mR50=macro(cm)-macro(bm))
    draws=[]
    rng=np.random.default_rng(17029)
    for _ in range(SPEC["gate"]["bootstrap_samples"]):
        ix=rng.integers(0,n,n)
        draws.append([(cc[ix,0].sum()-bc[ix,0].sum())/max(bc[ix,1].sum(),1),
                      (cr[ix]-br[ix]).mean(),macro(cm[ix])-macro(bm[ix])])
    lower=dict(zip(["object","R50","mR50"],np.quantile(draws,SPEC["gate"]["lower_quantile"],axis=0).tolist()))
    margin=SPEC["gate"]["relation_noninferiority_margin"]
    checks=dict(object_gain=point["object"]>=SPEC["gate"]["object_gain_min"],
                object_lower_positive=lower["object"]>0,
                R_noninferior=lower["R50"]>=-margin,
                mR_noninferior=lower["mR50"]>=-margin,
                sufficient_support=int(bc[:,1].sum())>=SPEC["gate"]["minimum_positive_objects"],
                full_gate=n==SPEC["gate_images"])
    return dict(accepted=all(checks.values()),checks=checks,delta=point,
                paired_image_bootstrap_one_sided_95_lower=lower,images=n,
                positive_objects=int(bc[:,1].sum()),exploratory_gate_not_confirmatory_claim=True)
