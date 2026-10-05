"""Two-GPU bounded V-W queue; summary never schedules confirmation automatically."""
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
from vw_protocol import OUT, load


def main():
    ensure_storage()
    here = Path(__file__).resolve().parent
    guard = output_path(ROOT / "status/E5_VW_queue.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _, smoke_reg, lane = load(True)
    smoke = read(OUT / lane / "fold0/summary.json")
    if (smoke["status"] != "complete" or smoke["protocol_sha256"] != smoke_reg
            or smoke["images"] != 3 or not smoke["route_gradient_checks_passed"]):
        raise RuntimeError("Current real-data gradient/route smoke missing")
    _, reg, _ = load(False)
    mutex = threading.Lock()
    outcomes, errors = {}, []
    deadline = time.monotonic() + 12*3600

    def task(label, args, directory):
        return dict(name="E5_"+label, args=args, completion=str(directory / "summary.json"),
                    progress_file=str(directory / "progress.json"), log=str(ROOT / "logs" / (label+".log")))

    folds = [task("VW_fold%d" % f, ["--stage", "fold", "--fold", str(f)], OUT / "train3000" / ("fold%d" % f)) for f in range(5)]
    summary = task("VW_summarize", ["--stage", "summarize"], OUT / "train3000")

    def update(row, status, **extra):
        with mutex:
            outcomes[row["name"]] = status
            atomic_json(ROOT / "status" / (row["name"]+".json"), dict(row, status=status, **extra))
            atomic_json(ROOT / "status/E5_VW_queue.json", dict(status="running", pid=os.getpid(),
                protocol_sha256=reg, outcomes=dict(outcomes), errors=list(errors)))

    def execute(row, gpu=None):
        row = dict(row, gpu=gpu)
        try:
            while gpu is not None and subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                    "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
                update(row, "waiting_gpu", detail="Preserving existing GPU processes")
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU waiting deadline")
                time.sleep(20)
            if time.monotonic() >= deadline:
                raise TimeoutError("Dispatch deadline exceeded")
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "", PYTHONUNBUFFERED="1",
                       OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            command = [env["SGG_NATIVE_PYTHON"], str(here / "vw_experiment.py")] + row["args"]
            print("[start] {} GPU={}".format(row["name"], gpu), flush=True)
            with output_path(row["log"]).open("a") as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(row, "running", pid=child.pid, started=time.time())
                try:
                    code = child.wait(timeout=min(4*3600, max(1, deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
            if code or not Path(row["completion"]).exists():
                raise RuntimeError("Stage exited {} or missing summary".format(code))
            done = read(row["completion"])
            if done["status"] != "complete" or done["protocol_sha256"] != reg:
                raise RuntimeError("Wrong completion contract")
            update(row, "complete", finished=time.time())
            print("[complete] "+row["name"], flush=True)
        except Exception as error:
            with mutex:
                errors.append(dict(task=row["name"], error=str(error)))
            update(row, "failed", detail=str(error))
            print("[failed] {}: {}".format(row["name"], error), flush=True)

    for row in folds + [summary]:
        update(row, "queued")
    pending = deque(folds)

    def worker(gpu):
        gpu_guard = output_path(ROOT / "status" / ("gpu%d.resource.lock" % gpu)).open("a")
        acquired = False
        try:
            while not acquired:
                try:
                    fcntl.flock(gpu_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Resource lock deadline")
                    time.sleep(20)
            while True:
                with mutex:
                    if not pending:
                        return
                    row = pending.popleft()
                    failed = bool(errors)
                if failed:
                    update(row, "blocked", detail="Earlier stage failed; no further dispatch")
                else:
                    execute(row, gpu)
        except Exception as error:
            with mutex:
                errors.append(dict(task="gpu%d_worker" % gpu, error=str(error)))
        finally:
            gpu_guard.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in [0, 1]]
        for future in futures:
            future.result()
    while pending:
        update(pending.popleft(), "blocked", detail="Workers failed or timed out")
    if not errors:
        execute(summary)
    else:
        update(summary, "blocked", detail="Incomplete folds")
    status = "complete" if all(v == "complete" for v in outcomes.values()) else "failed"
    atomic_json(ROOT / "status/E5_VW_queue.json", dict(status=status, pid=os.getpid(), protocol_sha256=reg,
        outcomes=outcomes, errors=errors, report=summary["completion"],
        note="Bounded training-only pilot ended; no automatic confirmation, seeds, tuning or shutdown"))
    print("[VW-finished] "+status, flush=True)
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
