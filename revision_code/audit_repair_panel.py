"""Audit completed V-Q/R/S without changing them or selecting a new winner."""
from pathlib import Path
import time
import numpy as np
import torch

from common import ROOT, ensure_storage, sha256
from evidence_completion import read, paired_ratio
from vp_experiment import progress, finish

NAMES = ["VQ_bounded_residual", "VR_kernel_residual", "VS_visual_prior"]


def main():
    ensure_storage(); torch.set_num_threads(2)
    out = ROOT/"results/E5_panel_audit_20261004"; started = time.monotonic()
    report = dict(status="complete", images=3000, branches={}, audit_source_sha256=sha256(__file__),
        validation_accessed=False, test_accessed=False,
        formal_gate_accepted=False, evidence_scope="Reused VG training images; descriptive audit only")
    canonical = {}
    for name in NAMES:
        folder = ROOT/"results"/name/"train3000"
        protocol = read(folder/"protocol.json"); reg = sha256(folder/"protocol.json")
        for file,digest in protocol["sources"].items():
            if sha256(Path(__file__).with_name(file)) != digest: raise RuntimeError("Source drift: "+file)
        summary = read(folder/"summary.json"); rows = []; noops = []
        if summary["protocol_sha256"] != reg: raise RuntimeError("Summary protocol mismatch")
        for fold in range(5):
            stage = folder/("fold%d" % fold); split = read(stage/"split.json")
            cp = ROOT/"checkpoints"/name/"train3000"/("fold%d.pth" % fold)
            digest = sha256(cp); complete = read(stage/"summary.json")
            if complete["checkpoint_sha256"] != digest or complete["protocol_sha256"] != reg:
                raise RuntimeError("Checkpoint/fold provenance mismatch")
            a,b,c,d = [set(split[k]) for k in ["training", "held", "fit", "inner_validation"]]
            if a & b or c & d or c|d != a or a|b != set(protocol["image_ids"]):
                raise RuntimeError("Invalid image split")
            choices = complete.get("selected_epochs", complete.get("selected_strengths"))
            for iid in split["held"]:
                row = read(stage/"images"/(iid+".json"))
                if row["protocol_sha256"] != reg or row["checkpoint_sha256"] != digest or row["fold"] != fold:
                    raise RuntimeError("Per-image provenance mismatch")
                native = row["metrics"]["native"]
                if iid in canonical and native != canonical[iid]: raise RuntimeError("Cross-branch native mismatch")
                canonical[iid] = native
                for arm,value in choices.items():
                    if value == 0:
                        for k in ["positive_correct", "post_nms_correct", "positive_objects", "recalls", "class_recalls"]:
                            if row["metrics"][arm][k] != native[k]: raise RuntimeError("No-op parity mismatch")
                rows.append(row)
            noops.append(dict(fold=fold, no_op_arms=[k for k,v in choices.items() if v == 0]))
            progress(out, "audit_"+name, fold+1, 5, started)
        if len(rows) != 3000 or len({r["image_id"] for r in rows}) != 3000:
            raise RuntimeError("Incomplete coverage")
        values = {}; base = [r["metrics"]["native"] for r in rows]
        den = np.array([r["positive_objects"] for r in base])
        for arm in summary["arms"]:
            trial = [r["metrics"][arm] for r in rows]
            if [r["positive_objects"] for r in trial] != den.tolist(): raise RuntimeError("Denominator drift")
            before = paired_ratio([r["positive_correct"] for r in trial], [r["positive_correct"] for r in base], den)
            after = paired_ratio([r["post_nms_correct"] for r in trial], [r["post_nms_correct"] for r in base], den)
            values[arm] = dict(pre_nms_identity=before, post_nms_identity=after,
                pre_nms_net_correct=sum(r["positive_correct"]-b["positive_correct"] for r,b in zip(trial,base)),
                post_nms_net_correct=sum(r["post_nms_correct"]-b["post_nms_correct"] for r,b in zip(trial,base)),
                R50=summary["arms"][arm]["R"]["50"], mR50=summary["arms"][arm]["mR"]["50"])
        report["branches"][name] = dict(protocol_sha256=reg, summary_sha256=sha256(folder/"summary.json"),
            records_verified=len(rows), no_op_folds=noops, arms=values,
            caveat="Net stage differences are not a causal mediation estimate; counts are matched-positive input proposals, not unique GT objects")
    bundle = torch.load(str(ROOT/"cache/VQ_bounded_residual/train3000/data.pt"),map_location="cpu")
    mapping = {iid:i for i,iid in enumerate(bundle["image_ids"])}
    activations = []
    for fold in range(5):
        path = ROOT/"checkpoints/VR_kernel_residual/train3000"/("fold%d.pth" % fold)
        cp = torch.load(str(path),map_location="cpu"); state=cp["states"]["kernel_residual"]
        samples = {}
        for split in ["fit_ids", "held_ids"]:
            ix=[]
            for iid in cp[split]:
                j=mapping[iid]; a,b=bundle["offsets"][j:j+2]; ix.extend(range(a,b))
            x=bundle["features"][np.asarray(ix)[::37]]
            phi=((x@state["landmarks"].t()-1)/.07).exp()
            residual=phi@state["coefficients"]
            samples[split]=dict(proposals=len(x), kernel_median=float(phi.median()),
                kernel_mean=float(phi.mean()), residual_abs_mean=float(residual.abs().mean()),
                residual_abs_max=float(residual.abs().max()))
        activations.append(dict(fold=fold, checkpoint_sha256=sha256(path), samples=samples))
    report["kernel_activation_audit"] = activations
    report["interpretation"] = "V-R tested a strongly attenuated kernel. V-T must be a new protocol, not an overwritten successful V-R."
    report["elapsed_seconds"] = time.monotonic()-started
    finish(out,report)


if __name__ == "__main__": main()
