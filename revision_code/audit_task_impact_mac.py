"""Independent recount of the sealed T1 pilot, without changing its decisions."""
import argparse
import json
from pathlib import Path

import numpy as np

from vj_mac import digest, write


def audit(out):
    read=lambda name:json.loads((out/name).read_text())
    summary,protocol,splits=read("summary.json"),read("protocol.json"),read("splits.json")
    assert digest(out/"protocol.json")==summary["protocol_sha256"]
    assert digest(out/"thresholds.json")==summary["threshold_sha256"]
    assert digest(Path(__file__).with_name("task_impact_mac.py"))==protocol["source_sha256"]
    assert digest(Path(__file__).with_name("T1_MAC_PLAN.md"))==protocol["plan_sha256"]
    assert digest(out/"inputs/annotations.json")==protocol["input_hashes"]["annotations"]
    assert digest(out/"inputs/VG-SGG-dicts.json")==protocol["input_hashes"]["vocabulary"]
    cal,screen,gate=map(set,[splits["calibration"],splits["screening"],splits["confirmation"]])
    assert len(cal)==len(screen)==500 and len(gate)==1000
    assert not (cal&screen or cal&gate or screen&gate)
    rows=read("decisions.json"); tasks=read("tasks.json")
    assert len(rows)==len({r["id"] for r in rows})==summary["screening_queries"]
    assert {r["image_id"] for r in rows}==screen
    assert [{k:r[k] for k in ["query","answers","stratum","image_id","id"]} for r in rows]==tasks["screening"]
    annotations={r["image_id"]:r for r in read("inputs/annotations.json")["development"]}
    totals={name:dict(positive=0,positive_correct=0,positive_target_correct=0,negative=0,
        false_answer=0,category_absent=0,category_false_answer=0,relation_absent=0,
        relation_false_answer=0,emitted=0,correct_emitted=0) for name in summary["results"]}
    thresholds=read("thresholds.json")
    for t in thresholds.values():
        if "empirical_false_answer_rate" in t: assert t["empirical_false_answer_rate"]<=.05
    for name in ["vk_oof_geometry","mismatched_geometry"]:
        assert thresholds[name]["threshold"]==thresholds["soft_geometry"]["threshold"]
    for row in rows:
        item=annotations[row["image_id"]]; labels=item["labels"]; q=row["query"]
        box=np.asarray(item["boxes"],dtype=float)
        centers=(box[:,:2]+box[:,2:])/(2*np.asarray(item["size"]))
        answers=[]
        for i,t in enumerate(labels):
            for j,a in enumerate(labels):
                if i==j or t!=q["target"] or a!=q["anchor"]: continue
                dx,dy=centers[j]-centers[i]
                if dict(left=dx>.05,right=dx<-.05,above=dy>.05,below=dy<-.05)[q["relation"]]:
                    answers.append([i,j])
        assert answers==row["answers"]
        positive=bool(answers); assert positive==(row["stratum"]=="positive")
        if row["stratum"]=="category_absent": assert q["target"] not in labels
        if row["stratum"]=="relation_absent": assert q["target"] in labels and q["anchor"] in labels
        for name,d in row["decisions"].items():
            emitted=d["pair"] is not None
            assert emitted==(d["score"]>thresholds[name]["threshold"])
            correct=d["pair"] in answers if positive else not emitted
            assert correct==d["correct"]
            assert (positive and d["pair"] in answers)==d["positive_pair_success"]
            c=totals[name]; c["positive"]+=positive; c["negative"]+=not positive
            c["positive_correct"]+=positive and correct
            c["positive_target_correct"]+=bool(positive and emitted and d["pair"][0] in {x[0] for x in answers})
            c["false_answer"]+=not positive and emitted
            c["emitted"]+=emitted; c["correct_emitted"]+=positive and correct
            for stratum,prefix in [("category_absent","category"),("relation_absent","relation")]:
                c[stratum]+=row["stratum"]==stratum
                c[prefix+"_false_answer"]+=row["stratum"]==stratum and emitted
    for name,counts in totals.items():
        for k,v in counts.items(): assert v==summary["results"][name][k],(name,k,v)
    transitions={}
    for a,b in [("soft_geometry","hard_geometry"),("vk_oof_geometry","soft_geometry")]:
        values={}
        for s in ["positive","category_absent","relation_absent"]:
            selected=[r for r in rows if r["stratum"]==s]
            values[s]=dict(n=len(selected),
                repair=sum(not r["decisions"][b]["correct"] and r["decisions"][a]["correct"] for r in selected),
                damage=sum(r["decisions"][b]["correct"] and not r["decisions"][a]["correct"] for r in selected))
        transitions[a+"_vs_"+b]=values
    result=dict(status="passed",images=len(screen),queries=len(rows),checked_decisions=len(rows)*len(totals),
        exact_answers_reconstructed_from_boxes=True,calibration_screen_confirmation_disjoint=True,
        source_and_threshold_hashes_unchanged=True,transitions=transitions,
        scientific_acceptance="NOT PASSED: primary negative-false-answer bound failed; VK task gain CI includes zero",
        summary_sha256=digest(out/"summary.json"),auditor_sha256=digest(Path(__file__)))
    write(out/"postrun_audit.json",result); print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    audit(parser.parse_args().output)
