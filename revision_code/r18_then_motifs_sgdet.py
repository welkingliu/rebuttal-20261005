"""Finish the open-vocabulary audit, then use GPU 1 for plain Motifs SGDet."""
import fcntl
import json
import os
import subprocess
from common import ROOT, atomic_json, ensure_storage


def main():
    ensure_storage()
    smoke=ROOT/"results/R18_ovsgtr/smoke/summary.json"
    d=json.loads(smoke.read_text())
    if d.get("status")!="complete" or d.get("images")!=2 or d.get("controlled_images",0)<1:
        raise RuntimeError("OvSGTR smoke must validate both inference and intervention")
    state=ROOT/"status/R18_then_R16.json"
    record=dict(pid=os.getpid(),gpu=[1],status="waiting_gpu",stage="R18 open-vocabulary audit",
                command=[__file__],completion=str(ROOT/"results/R18_ovsgtr/formal/summary.json"))
    atomic_json(state,record)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="1")
    lock=(ROOT/"status/gpu1.resource.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    jobs=[("R18_ovsgtr_formal",ROOT/"external/R18_ovsgtr/venv/bin/python",[str(ROOT/"code/r18_ovsgtr.py")],
            ROOT/"results/R18_ovsgtr/formal/summary.json",ROOT/"results/R18_ovsgtr/formal/progress.json"),
          ("R16_sgdet_smoke",os.environ["SGG_NATIVE_PYTHON"],[str(ROOT/"code/r16_motifs.py"),"--task","sgdet","--smoke"],
            ROOT/"results/R16_plain_motifs/sgdet/smoke/summary.json",ROOT/"results/R16_plain_motifs/sgdet/smoke/progress.json")]
    failures=[]
    for name,python,args,summary,progress in jobs:
        if summary.exists() and json.loads(summary.read_text()).get("status")=="complete":continue
        log=ROOT/"logs"/(name+".log")
        with log.open("a",buffering=1) as f:
            child=subprocess.Popen([str(python)]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
            record.update(status="running",stage=name,child_pid=child.pid,log=str(log),
                          completion=str(summary),progress_file=str(progress))
            atomic_json(state,record)
            code=child.wait()
        if code or not summary.exists():failures.append(dict(stage=name,returncode=code))
    lock.close()
    record.update(status="complete" if not failures else "completed_with_failures",failures=failures,child_pid=None)
    atomic_json(state,record)
    if any(x["stage"]=="R16_sgdet_smoke" for x in failures):
        record.update(status="blocked",reason="SGDet smoke failed; formal run not started")
        atomic_json(state,record);return
    subprocess.check_call([os.environ["SGG_NATIVE_PYTHON"],str(ROOT/"code/r16_queue.py"),"--gpu","1","--tasks","sgdet"],env=env)


if __name__=="__main__":main()
