"""Independent plain-Motifs tasks; one failure never suppresses another task."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from common import ROOT, atomic_json, ensure_storage


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--tasks",nargs="+",choices=["predcls","sgcls","sgdet"],required=True)
    a=p.parse_args();ensure_storage()
    status=ROOT/"status"/("R16_gpu%d.json"%a.gpu)
    state=dict(status="waiting_dependency",pid=os.getpid(),gpu=[a.gpu],tasks=a.tasks,completed=[],failed=[],command=[__file__])
    atomic_json(status,state)
    statistics=ROOT/"results/R16_plain_motifs/statistics_audit/summary.json"
    deadline=time.monotonic()+1800
    while not statistics.exists() or not json.loads(statistics.read_text()).get("corrected_sha256"):
        if time.monotonic()>deadline:
            state.update(status="blocked",reason="Upstream statistics rebuild not complete")
            atomic_json(status,state);return
        time.sleep(5)
    state.update(status="waiting_gpu")
    atomic_json(status,state)
    lock=(ROOT/"status"/("gpu%d.resource.lock"%a.gpu)).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu))
    for task in a.tasks:
        smoke=ROOT/"results/R16_plain_motifs"/task/"smoke/summary.json"
        if not smoke.exists():
            log=ROOT/"logs"/("R16_"+task+"_smoke_fixed.log")
            with log.open("a",buffering=1) as f:
                child=subprocess.Popen([os.environ["SGG_NATIVE_PYTHON"],str(ROOT/"code/r16_motifs.py"),"--task",task,"--smoke"],
                                       env=env,stdout=f,stderr=subprocess.STDOUT)
                state.update(status="running",task=task+" smoke",child_pid=child.pid,log=str(log),
                             progress_file=str(smoke.parent/"progress.json"),completion=str(smoke))
                atomic_json(status,state);rc=child.wait()
            if rc or not smoke.exists():
                state["failed"].append(dict(task=task,stage="smoke",exit_code=rc,log=str(log)))
                atomic_json(status,state);continue
        summary=ROOT/"results/R16_plain_motifs"/task/"formal/summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("status")=="complete":
            state["completed"].append(task);continue
        log=ROOT/"logs"/("R16_"+task+"_formal.log")
        command=[os.environ["SGG_NATIVE_PYTHON"],str(ROOT/"code/r16_motifs.py"),"--task",task]
        with log.open("a",buffering=1) as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
            state.update(status="running",task=task,child_pid=child.pid,log=str(log),
                         progress_file=str(summary.parent/"progress.json"),completion=str(summary))
            atomic_json(status,state)
            rc=child.wait()
        if rc or not summary.exists():
            state["failed"].append(dict(task=task,exit_code=rc,log=str(log)))
        else:state["completed"].append(task)
        atomic_json(status,state)
    state.update(status="complete" if not state["failed"] else "completed_with_failures",child_pid=None)
    atomic_json(status,state)


if __name__=="__main__":main()
