"""Append V-B behind the existing diagnostic queue without interrupting it."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage, sha256
from vb_protocol import SPEC, VB, source_hashes, verify_registration


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def register():
    ensure_storage()
    path=ROOT/"manifests/VB_protocol.json"
    if path.exists():
        verify_registration()
        print("[REGISTERED] Existing unchanged V-B protocol",flush=True)
        return
    queue=json.loads((ROOT/"manifests/queue.json").read_text())
    predecessors=[j["id"] for j in queue["jobs"] if "command" in j]
    atomic_json(path,dict(registered_at=now(),spec=SPEC,source_hashes=source_hashes(),
                         predecessors=predecessors,previous_queue_sha256=sha256(ROOT/"manifests/queue.json"),
                         note="Registered before V-B pilot/gate/test; historical paper results already known"))
    print("[REGISTERED] V-B after",len(predecessors),"diagnostic stages",flush=True)


def status(value):
    atomic_json(ROOT/"status/VB_queue.json",dict(updated_at=now(),**value))


def wait_for_current():
    registration=verify_registration()
    while True:
        states={}
        for name in registration["predecessors"]:
            p=ROOT/"status"/(name+".json")
            states[name]=json.loads(p.read_text()).get("status") if p.exists() else "pending"
        pending={k:v for k,v in states.items() if v not in ("complete","failed","blocked")}
        status(dict(status="waiting_predecessors",pending=pending,states=states))
        if not pending:
            atomic_json(VB/"predecessor_outcomes.json",dict(observed_at=now(),states=states))
            return
        time.sleep(30)


def prerequisite():
    for task in SPEC["tasks"]:
        p=ROOT/"results/R1/tde_motifs"/task/"summary.json"
        if not p.is_file():
            raise RuntimeError("V-B needs full TDE-Motifs "+task+" reproduction audit")
        report=json.loads(p.read_text())
        checks=report.get("reproduction_checks",{})
        if not report.get("full_split") or not checks or not all(c["passes_existing_tolerance"] for c in checks.values()):
            raise RuntimeError("V-B base checkpoint failed existing "+task+" reference gate; no retraining scheduled")


def run_stage(task,stage,mode="conservative",seed=17,split="gate"):
    verify_registration();ensure_storage()
    name="VB_%s_%s_%s_s%d_%s"%(task,stage,mode,seed,split)
    if stage=="prepare":
        completion=ROOT/"cache/VB"/task/"summary.json"
    elif stage=="train":
        completion=VB/task/"training"/(mode+"_seed%d"%seed)/"summary.json"
    else:
        completion=VB/task/split/(mode+"_seed%d"%seed)/"summary.json"
    state=ROOT/"status"/(name+".json")
    if completion.is_file() and json.loads(completion.read_text()).get("status")=="complete":
        atomic_json(state,dict(status="complete",completion=str(completion),observed_at=now()))
        return
    gpu="0" if task=="sgcls" else "1"
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpu
    lock=None
    if stage!="train":
        lock=(ROOT/"status"/("gpu"+gpu+".resource.lock")).open("w")
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            busy=subprocess.check_output(["nvidia-smi","-i",gpu,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
            if not busy:
                break
            atomic_json(state,dict(status="waiting_gpu",compute_pids=busy,updated_at=now()))
            time.sleep(20)
    command=[os.environ["SGG_NATIVE_PYTHON"],"-u",str(ROOT/"code/vb_experiment.py"),stage,
             "--task",task,"--mode",mode,"--seed",str(seed),"--split",split]
    log=ROOT/"logs"/(name+"_"+time.strftime("%Y%m%d_%H%M%S")+".log")
    print("[START]",name,now(),flush=True)
    try:
        with log.open("w") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT,env=env)
            record=dict(status="running",pid=process.pid,command=command,log=str(log),started_at=now())
            atomic_json(state,record)
            code=process.wait()
        ok=code==0 and completion.is_file() and json.loads(completion.read_text()).get("status")=="complete"
        atomic_json(state,dict(record,status="complete" if ok else "failed",returncode=code,
                               completion=str(completion),finished_at=now()))
        if not ok:
            raise RuntimeError("V-B stage failed: "+name+"; inspect "+str(log))
        print("[COMPLETE]",name,now(),flush=True)
    finally:
        if lock is not None:
            lock.close()


def pilot_task(task):
    run_stage(task,"prepare")
    for mode in SPEC["kl_weights"]:
        run_stage(task,"train",mode)
    for mode in ["native","temperature","supervised","conservative"]:
        run_stage(task,"evaluate",mode)


def formal_task(task):
    for seed in [23,31]:
        for mode in SPEC["kl_weights"]:
            run_stage(task,"train",mode,seed)
            run_stage(task,"evaluate",mode,seed,"gate")
    for mode in ["native","temperature"]:
        run_stage(task,"evaluate",mode,17,"test")
    for seed in SPEC["seeds"]:
        for mode in SPEC["kl_weights"]:
            run_stage(task,"evaluate",mode,seed,"test")


def both(function):
    # These are independent experiment processes, not parallel tool invocations.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(function,task) for task in SPEC["tasks"]]
        errors=[]
        for f in futures:
            try:
                f.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    args=p.parse_args();ensure_storage()
    if args.register:
        register();return
    lock=(ROOT/"status/VB_queue.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        wait_for_current();prerequisite()
        status(dict(status="running_pilot"));both(pilot_task)
        command=[os.environ["SGG_NATIVE_PYTHON"],"-u",str(ROOT/"code/vb_experiment.py"),"decide"]
        subprocess.check_call(command,cwd=ROOT)
        decision=json.loads((VB/"pilot_decision.json").read_text())
        if not decision["accepted"]:
            status(dict(status="stopped_by_validation",decision=str(VB/"pilot_decision.json"),
                        reason="No formal test or additional seeds; registered identity/relation gate not met"))
            atomic_json(VB/"summary.json",dict(status="complete",accepted=False,phase="pilot_only",decision=decision,
                        interpretation="Exploratory candidate rejected; no full-test mitigation success claimed"))
            print("[STOP] V-B validation gate not met; results retained",flush=True)
            return
        status(dict(status="running_formal"));both(formal_task)
        command=[os.environ["SGG_NATIVE_PYTHON"],"-u",str(ROOT/"code/vb_experiment.py"),"report"]
        subprocess.check_call(command,cwd=ROOT)
        status(dict(status="complete",summary=str(VB/"summary.json")))
    except Exception as exc:
        status(dict(status="failed",reason=str(exc)))
        raise


if __name__=="__main__":
    main()
