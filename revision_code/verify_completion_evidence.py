"""Recount final R20 summaries from saved per-image prediction arrays."""
import json

import numpy as np

from common import ROOT, atomic_json, sha256
from evidence_completion import read, paired_ratio


def main():
    base=ROOT/"results/R20_plain_motifs_native_bridge";audits=[]
    for stage,expected in [("identity",2000),("visual",200)]:
        out=base/stage;summary=read(out/"summary.json");protocol=read(out/"protocol.json")
        rows=[read(p) for p in sorted((out/"images").glob("*.json"))]
        by_id={r["image_id"]:r for r in rows}
        order=[iid for iid in protocol["image_ids"] if iid in by_id]
        rows=[by_id[iid] for iid in order]
        assert len(rows)==expected==summary["images"] and len(by_id)==expected
        if stage=="identity":assert set(by_id)==set(protocol["image_ids"])
        checked=0
        for row in rows:
            assert row["protocol_sha256"]==sha256(out/"protocol.json") and row["invariance_passed"]
            assert row["native_cache_max_error"]<=1e-4
            with np.load(out/"images"/(row["image_id"]+".npz")) as z:
                truth=z["relation_gt"];initial=z["clean_prediction"]==truth
                assert len(truth)==row["relations"]==len(z["pairs"])
                for name,stats in row["stats"].items():
                    prediction=z[("prediction_"+name) if stage=="identity" else name]
                    assert int((prediction==truth).sum())==stats["correct"]
                    assert int(initial.sum())==stats["clean_correct"]
                    assert len(truth)==stats["relations"]
                    checked+=1
        for name in rows[0]["stats"]:
            a=[r["stats"][name] for r in rows]
            recalculated=paired_ratio([r["correct"] for r in a],[r["clean_correct"] for r in a],[r["relations"] for r in a])
            assert recalculated==summary["results"][name]
        if stage=="visual":
            for strength in (.25,.5,1.):
                key="key_%.2f"%strength;control="unrelated_%.2f"%strength
                recalculated=paired_ratio([r["stats"][key]["correct"] for r in rows],
                    [r["stats"][control]["correct"] for r in rows],[r["relations"] for r in rows])
                assert recalculated==summary["results"]["key_minus_unrelated_%.2f"%strength]
        audits.append(dict(stage=stage,images=len(rows),raw_prediction_conditions_checked=checked,
            relations=sum(r["relations"] for r in rows),max_native_cache_error=max(r["native_cache_max_error"] for r in rows),
            summaries_and_bootstrap_exact=True,summary_sha256=sha256(out/"summary.json")))
    validation=set(read(base/"validation/protocol.json")["image_ids"])
    test=set(read(base/"identity/protocol.json")["image_ids"])
    assert not validation&test
    report=dict(status="complete",checks=audits,validation_test_disjoint=True,scientific_verdict_not_changed=True)
    atomic_json(ROOT/"results/R22_response_evidence/raw_record_audit.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
