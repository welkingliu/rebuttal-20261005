"""Bounded training-only replay and CPU cross-validation; no gate evaluation."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage, output_path


def main():
    ensure_storage()
    guard = output_path(ROOT/"status/VO_queue.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    here = Path(__file__).resolve().parent
    base = ROOT/"results/VO_grouped_train_validation/train3000"
    plan = []
    for stage, gpu in [("replay", 0), ("crossval", None)]:
        command = ["bash", "-lc", "source '%s/storage_runtime.sh' && exec env CUDA_VISIBLE_DEVICES=%s OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$SGG_NATIVE_PYTHON\" '%s/vo_grouped_validation.py' --stage %s" % (here, gpu if gpu is not None else "", here, stage)]
        plan.append(dict(name="VO_"+stage, stage=stage, gpu=gpu, command=command,
            completion=str(base/stage/"summary.json"), progress_file=str(base/stage/"progress.json"),
            log=str(ROOT/"logs"/("VO_"+stage+".log"))))
    for row in plan:
        if not Path(row["completion"]).exists(): atomic_json(ROOT/"status"/(row["name"]+".json"), dict(row, status="queued"))
    outcomes = {}
    for i, row in enumerate(plan):
        status = ROOT/"status"/(row["name"]+".json")
        if Path(row["completion"]).exists():
            atomic_json(status, dict(row, status="complete")); outcomes[row["name"]] = "complete"; continue
        atomic_json(ROOT/"status/VO_queue.json", dict(status="running", pid=os.getpid(), current=row["name"],
            position=i+1, total=len(plan), stages=plan, outcomes=outcomes, plan="VG train-only grouped utility diagnosis; never validation/test"))
        try:
            if row["gpu"] is not None:
                deadline = time.monotonic()+2*3600
                while True:
                    running = subprocess.check_output(["nvidia-smi", "-i", str(row["gpu"]), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if not running: break
                    atomic_json(status, dict(row, status="waiting_gpu", detail=running))
                    if time.monotonic() > deadline: raise TimeoutError("GPU wait expired")
                    time.sleep(30)
            print("[start] "+row["name"], flush=True)
            with output_path(row["log"]).open("a") as log:
                p = subprocess.Popen(row["command"], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(status, dict(row, status="running", pid=p.pid))
                try: code = p.wait(timeout=2*3600)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    code = 124
            if code or not Path(row["completion"]).exists(): raise RuntimeError("Stage failed, exit="+str(code))
        except Exception as error:
            atomic_json(status, dict(row, status="failed", detail=str(error)))
            for later in plan[i+1:]: atomic_json(ROOT/"status"/(later["name"]+".json"), dict(later, status="blocked", detail="Prerequisite failed"))
            atomic_json(ROOT/"status/VO_queue.json", dict(status="failed", current=row["name"], detail=str(error)))
            raise
        outcomes[row["name"]] = "complete"; atomic_json(status, dict(row, status="complete"))
        print("[complete] "+row["name"], flush=True)
    atomic_json(ROOT/"status/VO_queue.json", dict(status="complete", outcomes=outcomes,
        note="No automatic threshold search, validation/test evaluation or deployment"))


if __name__ == "__main__": main()
