"""Detached bounded V-T follow-up; never accesses validation or test."""
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


def main():
    ensure_storage(); here = Path(__file__).resolve().parent
    guard = output_path(ROOT/"status/VT_queue.lock").open("w")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from vt_experiment import registration, NAME
    _,reg,lane = registration(True)
    smoke = read(ROOT/"results"/NAME/lane/"fold0/summary.json")
    if smoke["status"] != "complete" or smoke["protocol_sha256"] != reg or smoke["images"] != 3:
        raise RuntimeError("Current real-data smoke missing")
    registration(False)
    mutex=threading.Lock(); outcomes={}; errors=[]; deadline=time.monotonic()+4*3600

    def task(name,script,args,directory,gpu=None):
        return dict(name=name,script=script,args=args,gpu=gpu,
            completion=str(directory/"summary.json"),progress_file=str(directory/"progress.json"),
            log=str(ROOT/"logs"/(name+".log")))

    audit=task("VT_panel_audit","audit_repair_panel.py",[],ROOT/"results/E5_panel_audit_20261004")
    folds=[task("VT_fold%d"%f,"vt_experiment.py",["--stage","fold","--fold",str(f)],
        ROOT/"results"/NAME/"train3000"/("fold%d"%f)) for f in range(5)]
    aggregate=task("VT_summarize","vt_experiment.py",["--stage","summarize"],ROOT/"results"/NAME/"train3000")

    def update(row,status,**extra):
        with mutex:
            outcomes[row["name"]]=status
            atomic_json(ROOT/"status"/(row["name"]+".json"),dict(row,status=status,**extra))
            atomic_json(ROOT/"status/VT_queue.json",dict(status="running",pid=os.getpid(),outcomes=dict(outcomes),errors=list(errors)))

    def execute(row,gpu=None):
        row=dict(row,gpu=gpu)
        try:
            while gpu is not None and subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():
                update(row,"waiting_gpu",detail="Existing compute job is preserved")
                if time.monotonic()>=deadline: raise TimeoutError("GPU wait budget")
                time.sleep(20)
            if time.monotonic()>=deadline: raise TimeoutError("Four-hour dispatch budget")
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                PYTHONUNBUFFERED="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
            command=[os.environ["SGG_NATIVE_PYTHON"],str(here/row["script"])]+row["args"]
            print("[start] %s GPU=%s"%(row["name"],gpu),flush=True)
            with output_path(row["log"]).open("a") as log:
                process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(row,"running",pid=process.pid,started=time.time())
                try: code=process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    code=124
            if code or not Path(row["completion"]).exists() or read(row["completion"])["status"]!="complete":
                raise RuntimeError("Stage failed: exit %s"%code)
            update(row,"complete",finished=time.time()); print("[complete] "+row["name"],flush=True)
            return True
        except Exception as error:
            with mutex: errors.append(dict(task=row["name"],error=str(error)))
            update(row,"failed",detail=str(error)); print("[failed] %s: %s"%(row["name"],error),flush=True)
            return False

    for row in [audit]+folds+[aggregate]: update(row,"queued")
    # Provenance audit is a prerequisite, not merely an informational side task.
    if execute(audit):
        pending=deque(folds)
        def worker(gpu):
            while True:
                with mutex:
                    if not pending: return
                    row=pending.popleft(); prior_error=bool(errors)
                if prior_error: update(row,"blocked",detail="Earlier stage failed; no further dispatch"); continue
                execute(row,gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(worker,gpu) for gpu in [0,1]]
            for job in jobs: job.result()
        if all(outcomes[r["name"]]=="complete" for r in folds): execute(aggregate)
        else: update(aggregate,"blocked",detail="Incomplete folds")
    else:
        for row in folds+[aggregate]: update(row,"blocked",detail="Input audit failed")
    status="complete" if all(v=="complete" for v in outcomes.values()) else "failed"
    atomic_json(ROOT/"status/VT_queue.json",dict(status=status,pid=os.getpid(),outcomes=outcomes,errors=errors,
        report=aggregate["completion"],note="Stopped after train-only exploratory screen; no validation/test, seed expansion or shutdown"))
    print("[VT-finished] "+status,flush=True)
    if status!="complete": raise SystemExit(1)


if __name__ == "__main__": main()
