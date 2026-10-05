"""Bounded two-GPU proposal-expert screening; never invokes validation or test."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from common import ROOT, atomic_json, ensure_storage, output_path
from evidence_completion import read


def main():
    ensure_storage()
    guard = output_path(ROOT/"status/VP_queue.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    from vp_experiment import registration
    _, smoke_reg, _ = registration(True)
    smoke = read(ROOT/"results/VP_proposal_expert/smoke/fold0/summary.json")
    fitted = read(ROOT/"results/VP_proposal_expert/smoke/fold0/training_summary.json")
    if (smoke.get("status") != "complete" or smoke.get("images") != 3
            or smoke.get("protocol_sha256") != smoke_reg
            or fitted.get("foreground_weight_max_change", 0) <= 0):
        raise RuntimeError("A matching real-gradient, native-replay smoke test is required")
    here = Path(__file__).resolve().parent
    base = ROOT/"results/VP_proposal_expert/train3000"
    plan = []
    for name, stage, fold, gpu in [("prepare", "prepare", None, None)]+[
            ("fold%d" % f, "fold", f, f % 2) for f in range(5)]+[("summary", "summarize", None, None)]:
        arguments = "--stage "+stage+(" --fold %d" % fold if fold is not None else "")
        command = ["bash", "-lc", "source '%s/storage_runtime.sh' && exec env CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$SGG_NATIVE_PYTHON\" '%s/vp_experiment.py' %s" % (here, gpu if gpu is not None else "", here, arguments)]
        directory = base if stage == "summarize" else base/name
        plan.append(dict(name="VP_"+name, stage=stage, fold=fold, gpu=gpu, command=command,
            completion=str(directory/"summary.json"), progress_file=str(directory/"progress.json"),
            log=str(ROOT/"logs"/("VP_"+name+".log"))))
    mutex = threading.Lock(); stop = threading.Event(); outcomes = {}; active = {}; errors = []

    def update(row, status, **extra):
        with mutex:
            outcomes[row["name"]] = status
            if status in ["running", "waiting_gpu"]: active[row["name"]] = row["gpu"]
            else: active.pop(row["name"], None)
            atomic_json(ROOT/"status"/(row["name"]+".json"), dict(row, status=status, **extra))
            atomic_json(ROOT/"status/VP_queue.json", dict(status="running", pid=os.getpid(),
                current={(str(gpu) if gpu is not None else "CPU"): name for name, gpu in active.items()},
                stages=plan, outcomes=dict(outcomes),
                plan="Detected-proposal classifier; five train-only image folds; no validation/test access"))

    def execute(row):
        if stop.is_set():
            update(row, "blocked", detail="A prerequisite or sibling fold failed; no further expansion")
            return False
        try:
            if row["gpu"] is not None and not Path(row["completion"]).exists():
                deadline = time.monotonic()+3*3600
                while True:
                    if stop.is_set():
                        update(row, "blocked", detail="Sibling fold failed while waiting for GPU")
                        return False
                    occupied = subprocess.check_output(["nvidia-smi", "-i", str(row["gpu"]),
                        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if not occupied: break
                    update(row, "waiting_gpu", detail=occupied)
                    if time.monotonic() > deadline: raise TimeoutError("GPU wait expired")
                    stop.wait(30)
            print("[start] "+row["name"], flush=True)
            with output_path(row["log"]).open("a") as log:
                p = subprocess.Popen(row["command"], stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                update(row, "running", pid=p.pid, started=time.time())
                try: code = p.wait(timeout=2*3600)
                except subprocess.TimeoutExpired:
                    # Terminate only this queue's own process group, never other GPU work.
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    code = 124
            if code or not Path(row["completion"]).exists():
                raise RuntimeError("Stage failed, exit="+str(code))
            if read(row["completion"]).get("status") != "complete": raise RuntimeError("Stage did not complete")
            update(row, "complete", finished=time.time()); print("[complete] "+row["name"], flush=True)
            return True
        except Exception as error:
            stop.set()
            with mutex: errors.append(dict(stage=row["name"], error=str(error)))
            update(row, "failed", detail=str(error)); print("[failed] %s: %s" % (row["name"], error), flush=True)
            return False

    for row in plan: update(row, "queued")
    if execute(plan[0]):
        def lane(gpu):
            for row in plan[1:-1]:
                if row["gpu"] == gpu: execute(row)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, gpu) for gpu in [0, 1]]
            for future in futures: future.result()
        execute(plan[-1])
    else:
        for row in plan[1:]: update(row, "blocked", detail="Preparation failed")
    atomic_json(ROOT/"status/VP_queue.json", dict(status="failed" if errors else "complete", pid=os.getpid(),
        outcomes=outcomes, errors=errors, note="Stop after training-only screening; no automatic gate, seeds or deployment"))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
