"""Detached two-runtime/two-GPU re-observation queue with strict prerequisites."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from common import ROOT,ensure_storage,atomic_json,output_path,sha256
from evidence_completion import read
from vu_protocol import OUT,load


def main():
    ensure_storage();here=Path(__file__).resolve().parent
    guard=output_path(ROOT/"status/VU_queue.lock").open("w");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    _,reg,lane=load(True)
    smoke=read(OUT/lane/"fold0/summary.json")
    if smoke["status"]!="complete" or smoke["protocol_sha256"]!=reg or smoke["images"]!=3:
        raise RuntimeError("Current real-data smoke missing")
    full,full_reg,_=load(False,check_assets=True)
    mutex=threading.Lock();outcomes={};errors=[];deadline=time.monotonic()+12*3600

    def task(name,script,args,directory,modern=False):
        return dict(name=name,script=script,args=args,modern=modern,
            completion=str(directory/"summary.json"),progress_file=str(directory/"progress.json"),
            log=str(ROOT/"logs"/(name+".log")))

    extraction=[task("VU_extract%d"%s,"vu_features.py",["--shard",str(s)],OUT/"train3000"/("extract%d"%s),True) for s in [0,1]]
    prepare=task("VU_prepare","vu_experiment.py",["--stage","prepare"],OUT/"train3000/prepare")
    folds=[task("VU_fold%d"%f,"vu_experiment.py",["--stage","fold","--fold",str(f)],OUT/"train3000"/("fold%d"%f)) for f in range(5)]
    summary=task("VU_summarize","vu_experiment.py",["--stage","summarize"],OUT/"train3000")

    def update(row,status,**extra):
        with mutex:
            outcomes[row["name"]]=status
            atomic_json(ROOT/"status"/(row["name"]+".json"),dict(row,status=status,**extra))
            atomic_json(ROOT/"status/VU_queue.json",dict(status="running",pid=os.getpid(),protocol_sha256=full_reg,
                outcomes=dict(outcomes),errors=list(errors)))

    def execute(row,gpu=None):
        row=dict(row,gpu=gpu)
        try:
            while gpu is not None and subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():
                update(row,"waiting_gpu",detail="Existing GPU work is preserved")
                if time.monotonic()>=deadline:raise TimeoutError("GPU wait budget expired")
                time.sleep(20)
            if time.monotonic()>=deadline:raise TimeoutError("Dispatch budget expired")
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                PYTHONUNBUFFERED="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
            interpreter=env["SGG_MODERN_PYTHON" if row["modern"] else "SGG_NATIVE_PYTHON"]
            command=[interpreter,str(here/row["script"])]+row["args"]
            print("[start] %s GPU=%s"%(row["name"],gpu),flush=True)
            with output_path(row["log"]).open("a") as log:
                p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(row,"running",pid=p.pid,started=time.time())
                try:code=p.wait(timeout=6*3600)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    code=124
            if code or not Path(row["completion"]).exists():raise RuntimeError("Stage exit %s"%code)
            result=read(row["completion"])
            if result["status"]!="complete" or result["protocol_sha256"]!=full_reg:raise RuntimeError("Completion/provenance mismatch")
            update(row,"complete",finished=time.time());print("[complete] "+row["name"],flush=True);return True
        except Exception as error:
            with mutex:errors.append(dict(task=row["name"],error=str(error)))
            update(row,"failed",detail=str(error));print("[failed] %s: %s"%(row["name"],error),flush=True);return False

    def parallel(rows):
        pending=deque(rows)
        def worker(gpu):
            while True:
                with mutex:
                    if not pending:return
                    row=pending.popleft();failed=bool(errors)
                if failed:update(row,"blocked",detail="Earlier prerequisite failed");continue
                execute(row,gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(worker,gpu) for gpu in [0,1]]
            for job in jobs:job.result()
    all_rows=extraction+[prepare]+folds+[summary]
    for row in all_rows:update(row,"queued")
    parallel(extraction)
    if not errors and execute(prepare):
        parallel(folds)
        if not errors:execute(summary)
    for row in all_rows:
        if outcomes[row["name"]]=="queued":update(row,"blocked",detail="Prerequisite failed")
    status="complete" if all(x=="complete" for x in outcomes.values()) else "failed"
    atomic_json(ROOT/"status/VU_queue.json",dict(status=status,pid=os.getpid(),protocol_sha256=full_reg,
        outcomes=outcomes,errors=errors,report=summary["completion"],
        note="Bounded train-only exploration ended; no automatic formal validation/test, extra seed or shutdown"))
    print("[VU-finished] "+status,flush=True)
    if status!="complete":raise SystemExit(1)


if __name__=="__main__":main()
