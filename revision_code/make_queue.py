"""Create the finite rebuttal execution matrix on the experiment disk."""
import os
from common import ROOT, ensure_storage, atomic_json


def main():
    ensure_storage()
    native=os.environ["SGG_NATIVE_PYTHON"]
    modern=os.environ["SGG_MODERN_PYTHON"]
    jobs=[]
    def dep(job,path):
        return dict(job=job,completion=str(ROOT/path))
    def add(name,lane,script,flags,result,deps=(),python=native):
        jobs.append(dict(id=name,lane=lane,command=[python,"-u",str(ROOT/"code"/script)]+flags,
                         completion=str(ROOT/result),dependencies=list(deps)))
    for family,gpu,pid in [("transformer","gpu0",1148721),("tde_motifs","gpu1",1148722)]:
        validation=dict(completion=str(ROOT/("results/R2/%s/validation/summary.json"%family)),
                        pid=pid,expected_command="identity_intervention.py --family "+family)
        dev="R2_%s_dev"%family
        test="R2_%s_test"%family
        add(dev,gpu,"identity_intervention.py",["--family",family,"--stage","dev"],
            "results/R2/%s/dev/summary.json"%family,[validation])
        add(test,gpu,"identity_intervention.py",["--family",family,"--stage","test"],
            "results/R2/%s/test/summary.json"%family,[dep(dev,"results/R2/%s/dev/summary.json"%family),
              dep("R1_%s_sgcls"%family,"results/R1/%s/sgcls/summary.json"%family)])
    add("R3_dense_validation_sgcls","gpu0","export_dense.py",["--task","sgcls","--split","validation"],
        "cache/R3_dense/validation/sgcls/summary.json")
    add("R3_dense_test_sgcls","gpu0","export_dense.py",["--task","sgcls","--split","test"],
        "cache/R3_dense/test/sgcls/summary.json",[dep("R3_dense_validation_sgcls","cache/R3_dense/validation/sgcls/summary.json")])
    add("R3_dense_test_sgdet","gpu1","export_dense.py",["--task","sgdet","--split","test"],
        "cache/R3_dense/test/sgdet/summary.json")
    for family in ("transformer","tde_motifs"):
        for task in ("sgcls","predcls","sgdet"):
            add("R1_%s_%s"%(family,task),"cpu","evaluator_audit.py",["--family",family,"--task",task],
                "results/R1/%s/%s/summary.json"%(family,task))
    for task in ("sgcls","sgdet"):
        add("R3_decomposition_"+task,"cpu","mitigation_decomposition.py",["--task",task],
            "results/R3/%s/summary.json"%task,[dep("R3_dense_validation_sgcls","cache/R3_dense/validation/sgcls/summary.json"),
             dep("R3_dense_test_"+task,"cache/R3_dense/test/%s/summary.json"%task)],python=modern)
    for family,lane,task in [("transformer","gpu0","sgcls"),("tde_motifs","gpu1","sgdet")]:
        add("R4_paired_"+family,lane,"paired_visual_control.py",["--family",family],
            "results/R4_paired/%s/summary.json"%family,
            [dep("R3_dense_test_"+task,"cache/R3_dense/test/%s/summary.json"%task),
             dep("R2_%s_test"%family,"results/R2/%s/test/summary.json"%family)])
    jobs.append(dict(id="R4_historical_paired_CI",lane="cpu",blocked_reason=
        "Historical per-image raw intervention records have not been located; aggregate means/CI cannot reconstruct a paired analysis. Contract audit is separately available."))
    jobs.append(dict(id="modern_live_extension",lane="cpu",blocked_reason=
        "Optional EGTR/SGTR semantic-intervention adapter is not implemented; original endpoint outputs must not be presented as identity-channel intervention."))
    atomic_json(ROOT/"manifests/queue.json",dict(schema="rebuttal_queue_v1",old_artifacts_read_only=True,jobs=jobs))
    print("[READY] Queued executable stages:",sum("command" in j for j in jobs))


if __name__=="__main__":
    main()
