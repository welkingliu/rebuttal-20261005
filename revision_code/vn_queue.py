"""Finite V-N queue. Never changes another process or expands after evaluation."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage, output_path


def run():
    ensure_storage()
    guard=output_path(ROOT/"status/VN_queue.lock").open("w")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    here=Path(__file__).resolve().parent
    stages=[("export",0,"SGG_NATIVE_PYTHON"),("features",1,"SGG_MODERN_PYTHON"),
            ("outcomes",0,"SGG_NATIVE_PYTHON"),("fit",None,"SGG_NATIVE_PYTHON"),
            ("evaluate",0,"SGG_NATIVE_PYTHON")]
    base=ROOT/"results/VN_train_proposal_router/train3000"
    queue=ROOT/"status/VN_queue.json"
    plan=[]
    for stage,gpu,python in stages:
        plan.append(dict(name="VN_"+stage,gpu=gpu,stage=stage,
            command=["bash","-lc","source '%s/storage_runtime.sh' && exec env CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$%s\" '%s/vn_train_proposals.py' --stage %s" %
                     (here,gpu if gpu is not None else "",python,here,stage)],
            completion=str(base/stage/"summary.json"),progress_file=str(base/stage/"progress.json"),
            log=str(ROOT/"logs"/("VN_train3000_"+stage+".log"))))
    for row in plan:
        if not Path(row["completion"]).exists():atomic_json(ROOT/"status"/(row["name"]+".json"),dict(row,status="queued"))
    outcomes={}
    plan_text="3000 VG train images: native proposals -> DINO crops -> post-NMS labels -> fixed routers -> one exploratory evaluation"
    for position,row in enumerate(plan,1):
        status_path=ROOT/"status"/(row["name"]+".json")
        if Path(row["completion"]).exists():
            outcomes[row["name"]]="complete"
            atomic_json(status_path,dict(row,status="complete"));continue
        atomic_json(queue,dict(status="running",pid=os.getpid(),current=row["name"],position=position,total=len(plan),plan=plan_text,stages=plan,outcomes=outcomes))
        if row["gpu"] is not None:
            deadline=time.monotonic()+4*3600
            while True:
                result=subprocess.check_output(["nvidia-smi","-i",str(row["gpu"]),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                if not result:break
                atomic_json(status_path,dict(row,status="waiting_gpu",detail="Existing compute processes: "+result))
                if time.monotonic()>deadline:raise TimeoutError("GPU wait budget reached")
                time.sleep(30)
        print("[start] "+row["name"],flush=True)
        with output_path(row["log"]).open("a") as log:
            process=subprocess.Popen(row["command"],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(status_path,dict(row,status="running",pid=process.pid))
            try:
                result=process.wait(timeout=4*3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                result=124
        if result or not Path(row["completion"]).exists():
            atomic_json(status_path,dict(row,status="failed",exit_code=result))
            for later in plan[position:]:atomic_json(ROOT/"status"/(later["name"]+".json"),dict(later,status="blocked",detail="Prerequisite failed: "+row["name"]))
            atomic_json(queue,dict(status="failed",pid=os.getpid(),current=row["name"],position=position,total=len(plan),plan=plan_text,stages=plan,outcomes=outcomes,exit_code=result))
            raise RuntimeError("Stopped at "+row["name"])
        atomic_json(status_path,dict(row,status="complete"));outcomes[row["name"]]="complete"
        print("[complete] "+row["name"],flush=True)
    atomic_json(queue,dict(status="complete",pid=os.getpid(),outcomes=outcomes,plan=plan_text,stages=plan,
        note="No automatic extra seeds, final-test evaluation, threshold search, or formal gate override"))


if __name__=="__main__":run()
