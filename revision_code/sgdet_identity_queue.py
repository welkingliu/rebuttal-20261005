"""Append R12 after R9 without modifying or interrupting its active schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import subprocess
import threading
import time

from common import ROOT, atomic_json, ensure_storage
from sgdet_identity_protocol import OUT, SPEC, register, verify, reference_gate


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid,0)
        return True
    except OSError:
        return False


def run(family, stage, gpu):
    verify()
    out = OUT/family/stage
    name = "R12_%s_%s"%(family,stage)
    state = ROOT/"status"/(name+".json")
    record = dict(name=name, family=family, stage=stage, gpu=[gpu], completion=str(out/"summary.json"),
                  progress_file=str(out/"progress.json"), status="waiting_gpu")
    existing = read(out/"summary.json")
    if existing.get("status")=="complete":
        valid = existing.get("development_gate_passed") if stage=="dev" else existing.get("invariance_passed")
        if stage=="smoke":
            valid = valid and existing.get("coverage",{}).get("evaluable_relations",0)>0
        atomic_json(state,dict(record,status="complete" if valid else "failed",resumed=True))
        return bool(valid)
    atomic_json(state,record)
    lock = (ROOT/"status"/("gpu%d.resource.lock"%gpu)).open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():
            time.sleep(20)
        command = [os.environ["SGG_NATIVE_PYTHON"],"-u",str(ROOT/"code/sgdet_identity.py"),"--family",family,"--stage",stage]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = ROOT/"logs"/(name+"_"+time.strftime("%Y%m%d_%H%M%S")+".log")
        with log.open("w") as stream:
            process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            record.update(status="running",pid=process.pid,log=str(log),command=command)
            atomic_json(state,record)
            rc = process.wait()
        summary = read(out/"summary.json")
        ok = rc==0 and summary.get("status")=="complete"
        atomic_json(state,dict(record,status="complete" if ok else "failed",returncode=rc))
        print("[%s] %s log=%s"%("COMPLETE" if ok else "FAILED",name,log),flush=True)
        return ok
    except Exception as exc:
        atomic_json(state,dict(record,status="failed",error=str(exc)))
        print("[FAILED] %s: %s"%(name,exc),flush=True)
        return False
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--register",action="store_true")
    args = parser.parse_args(); ensure_storage()
    if args.register:
        register(); print("[REGISTERED] R12 waits for R9; independent guarded GPU lanes",flush=True); return
    verify()
    lock = (ROOT/"status/R12_queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = ROOT/"status/R12_queue.json"
    atomic_json(state,dict(status="waiting_dependency",pid=os.getpid(),dependency="R9_queue",
                          plan="GPU0 TDE-Motifs; GPU1 corrected Transformer; smoke -> validation -> development gate -> test",
                          note="No GPU lock is held while waiting"))
    while True:
        r9 = read(ROOT/"status/R9_queue.json")
        if r9.get("status") in ("complete","needs_attention"):
            break
        if r9.get("status")=="running" and r9.get("pid") and not alive(r9["pid"]):
            atomic_json(state,dict(status="needs_attention",reason="R9 queue process disappeared; no GPU launched"))
            return
        time.sleep(30)
    outcomes, current, mutex = {}, {}, threading.Lock()
    stages = ["smoke","validation","dev","test"]
    def publish():
        atomic_json(state,dict(status="running",pid=os.getpid(),current=current.copy(),outcomes=outcomes.copy(),
                              total=8,complete=sum(value=="complete" for value in outcomes.values())))
    def lane(family,gpu):
        gate = reference_gate(family)
        if not gate["passed"]:
            with mutex:
                outcomes[family]="reference_gate_failed"
                atomic_json(ROOT/"status"/("R12_"+family+"_eligibility.json"),dict(status="blocked",reference=gate,
                            completion=str(OUT/family/"test/summary.json"),reason=gate["reason"]))
                publish()
            return
        for stage in stages:
            name = family+"/"+stage
            with mutex:
                current[str(gpu)]=name; publish()
            ok = run(family,stage,gpu)
            with mutex:
                outcomes[name]="complete" if ok else "failed"
                current.pop(str(gpu),None); publish()
            if not ok:
                break
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane,family,gpu) for gpu,family in enumerate(SPEC["families"])]
        for future in futures:
            future.result()
    complete = all(outcomes.get(family+"/test")=="complete" for family in SPEC["families"])
    atomic_json(state,dict(status="complete" if complete else "needs_attention",outcomes=outcomes,
                          note="Reference/support/invariance failures do not cancel the other model lane"))


if __name__=="__main__":
    main()
