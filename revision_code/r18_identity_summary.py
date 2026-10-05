"""Image-cluster confidence intervals for the completed open-vocabulary audit."""
import json
from pathlib import Path
import numpy as np
from common import ROOT, atomic_json, sha256


def main():
    out=ROOT/"results/R18_ovsgtr/formal"
    summary=json.loads((out/"summary.json").read_text())
    assert summary["status"]=="complete"
    ids=json.loads((out/"protocol.json").read_text())["image_ids"]
    rows=[]
    for iid in ids:
        audit=json.loads((out/"images"/(str(iid)+".json")).read_text())["clean"]["identity_audit"]
        pairs=audit["pairs"]
        good=[p for p in pairs if p["identity_correct"]]
        bad=[p for p in pairs if not p["identity_correct"]]
        rows.append([len(good),sum(p["predicate_correct"] for p in good),len(bad),sum(p["predicate_correct"] for p in bad)])
    x=np.asarray(rows,dtype=float)
    def metric(a):
        n,h,m,k=a.sum(0)
        return [h/n if n else np.nan,k/m if m else np.nan,
                h/n-k/m if n and m else np.nan,m/(n+m) if n+m else np.nan]
    rng=np.random.default_rng(17)
    boot=np.asarray([metric(x[rng.integers(len(x),size=len(x))]) for _ in range(2000)])
    values=metric(x)
    metrics={k:dict(estimate=float(values[j]),ci95=np.nanquantile(boot[:,j],[.025,.975]).tolist(),
                    valid_bootstrap_replicates=int(np.isfinite(boot[:,j]).sum())) for j,k in
             enumerate(["predicate_hit_identity_correct","predicate_hit_identity_wrong","observational_gap","endpoint_identity_error"])}
    atomic_json(out/"identity_confidence_intervals.json",dict(status="complete",images=len(x),metrics=metrics,
        pairs=int(x[:,[0,2]].sum()),ci_unit="image cluster, jointly resample both identity groups, 2000 seed-17 replicates",
        interpretation="observational association conditional on class-agnostic localization and predicted pair coverage, not a causal identity effect",
        source_summary_sha256=sha256(out/"summary.json"),source_protocol_sha256=sha256(out/"protocol.json")))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=="__main__":main()
