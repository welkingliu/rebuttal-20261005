"""Detached two-GPU V-V queue; never expands past the registered training screen."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from common import ROOT, ensure_storage, atomic_json, output_path
from evidence_completion import read
from vv_protocol import OUT, load


def main():
    ensure_storage()
    here=Path(__file__).resolve().parent
    guard=output_path(ROOT/"status/VV_queue.lock").open("w")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    _,smoke_reg,lane=load(True)
    smoke=read(OUT/lane/"fold0/summary.json")
    if (smoke["status"]!="complete" or smoke["protocol_sha256"]!=smoke_reg
            or smoke["images"]!=3 or not smoke["smoke_gradients_exercised"]):
        raise RuntimeError("Current real-data gradient/native smoke missing")
    _,reg,_=load(False)
    mutex=threading.Lock();outcomes={};errors=[]
    deadline=time.monotonic()+12*3600

    def task(name,args,directory):
        return dict(name=name,args=args,completion=str(directory/"summary.json"),
            progress_file=str(directory/"progress.json"),log=str(ROOT/"logs"/(name+".log")))

    folds=[task("VV_fold%d"%f,["--stage","fold","--fold",str(f)],OUT/"train3000"/("fold%d"%f)) for f in range(5)]
    summary=task("VV_summarize",["--stage","summarize"],OUT/"train3000")

    def update(row,status,**extra):
        with mutex:
            outcomes[row["name"]]=status
            atomic_json(ROOT/"status"/(row["name"]+".json"),dict(row,status=status,**extra))
            atomic_json(ROOT/"status/VV_queue.json",dict(status="running",pid=os.getpid(),
                protocol_sha256=reg,outcomes=dict(outcomes),errors=list(errors)))

    def execute(row,gpu=None):
        row=dict(row,gpu=gpu)
        try:
            while gpu is not None and subprocess.check_output(["nvidia-smi","-i",str(gpu),
                    "--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():
                update(row,"waiting_gpu",detail="Preserving existing GPU processes")
                if time.monotonic()>=deadline:
                    raise TimeoutError("GPU wait budget expired")
                time.sleep(20)
            if time.monotonic()>=deadline:
                raise TimeoutError("Dispatch deadline exceeded")
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",PYTHONUNBUFFERED="1",
                OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
            command=[env["SGG_NATIVE_PYTHON"],str(here/"vv_experiment.py")]+row["args"]
            print("[start] %s GPU=%s"%(row["name"],gpu),flush=True)
            with output_path(row["log"]).open("a") as log:
                child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(row,"running",pid=child.pid,started=time.time())
                try:
                    code=child.wait(timeout=6*3600)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
            if code or not Path(row["completion"]).exists():
                raise RuntimeError("Stage exit %s"%code)
            value=read(row["completion"])
            if value["status"]!="complete" or value["protocol_sha256"]!=reg:
                raise RuntimeError("Completion contract mismatch")
            update(row,"complete",finished=time.time())
            print("[complete] "+row["name"],flush=True)
        except Exception as error:
            with mutex:errors.append(dict(task=row["name"],error=str(error)))
            update(row,"failed",detail=str(error))
            print("[failed] %s: %s"%(row["name"],error),flush=True)

    for row in folds+[summary]:
        update(row,"queued")
    pending=deque(folds)

    def worker(gpu):
        while True:
            with mutex:
                if not pending:return
                row=pending.popleft();failed=bool(errors)
            if failed: update(row,"blocked",detail="Earlier stage failed")
            else: execute(row,gpu)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu) for gpu in [0,1]]
        for future in futures:future.result()
    if not errors:execute(summary)
    else:update(summary,"blocked",detail="Incomplete outer folds")
    status="complete" if all(s=="complete" for s in outcomes.values()) else "failed"
    atomic_json(ROOT/"status/VV_queue.json",dict(status=status,pid=os.getpid(),protocol_sha256=reg,
        outcomes=outcomes,errors=errors,report=summary["completion"],
        note="Bounded training exploration ended; no automatic confirmation, test, tuning or shutdown"))
    print("[VV-finished] "+status,flush=True)
    if status!="complete":raise SystemExit(1)


if __name__=="__main__":
    main()
