"""Detached, bounded two-GPU queue for V-X; never dispatches official test sets."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import signal
import subprocess
import threading
import time

from common import ROOT, ensure_storage, atomic_json, output_path
from evidence_completion import read
from vx_common import HERE, OUT, SPEC, ARMS, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_storage()
    p, reg, lane = load(args.smoke)
    if not args.smoke:
        _, smoke_reg, _ = load(True)
        smoke = read(OUT/"smoke/summary.json")
        if smoke["status"] != "complete" or smoke["protocol_sha256"] != smoke_reg:
            raise RuntimeError("Current end-to-end smoke required")
        for arm in ARMS[1:]:
            audit = read(OUT/"smoke"/arm/"train/gradient_audit.json")
            if audit["protocol_sha256"] != smoke_reg or not audit["all_checks_passed"]:
                raise RuntimeError("Current visual gradient audit missing")
    guard = output_path(ROOT/"status/E5_VX_queue.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_guards = []
    deadline = time.monotonic()+SPEC["max_run_hours"]*3600
    status_path = ROOT/"status/E5_VX_queue.json"
    mutex = threading.Lock()
    outcomes, errors = {}, []

    def task(name, script, flags, completion, stage):
        return dict(name="E5_VX_"+name, script=script, args=flags,
                    completion=str(completion), progress_file=str(stage/"progress.json"),
                    log=str(ROOT/"logs"/("VX_%s_%s.log" % (lane, name))))

    prepare = task("prepare", "vx_native.py", ["--stage", "prepare"], OUT/lane/"prepare/summary.json", OUT/lane/"prepare")
    extract = [task("extract%d" % i, "vx_vision.py", ["--stage", "extract", "--shard", str(i)],
                    OUT/lane/("extract%d" % i)/"summary.json", OUT/lane/("extract%d" % i)) for i in range(2)]
    train = [task("train_"+a, "vx_vision.py", ["--stage", "train", "--arm", a],
                  OUT/lane/a/"train/summary.json", OUT/lane/a/"train") for a in ["frozen_head", "adapt_diagnostic", "adapt_plain"]]
    held = [task("held_"+a, "vx_vision.py", ["--stage", "held", "--arm", a],
                 OUT/lane/a/"held/summary.json", OUT/lane/a/"held") for a in ARMS[1:]]
    summary = task("summary", "vx_native.py", ["--stage", "summarize"], OUT/lane/"summary.json", OUT/lane)
    all_tasks = [prepare]+extract+train+held+[summary]

    def update(row, status, **extra):
        with mutex:
            outcomes[row["name"]] = status
            atomic_json(ROOT/"status"/(row["name"]+".json"), dict(row, status=status,
                        protocol_sha256=reg, lane=lane, **extra))
            atomic_json(status_path, dict(status="running", pid=os.getpid(), protocol_sha256=reg,
                        lane=lane, outcomes=dict(outcomes), errors=list(errors)))

    def execute(row, gpu):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("V-X bounded dispatch deadline")
            if os.path.exists(row["completion"]):
                done = read(row["completion"])
                if done["status"] != "complete" or done["protocol_sha256"] != reg:
                    raise RuntimeError("Stale completion record")
                update(row, "complete", gpu=gpu, resumed=True)
                return
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            runtime = env["SGG_NATIVE_PYTHON"] if row["script"] == "vx_native.py" else env["SGG_MODERN_PYTHON"]
            command = [runtime, str(HERE/row["script"])]+row["args"]+(["--smoke"] if args.smoke else [])
            with mutex:
                print("[start] %s GPU=%s" % (row["name"], gpu), flush=True)
            with output_path(row["log"]).open("a") as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(row, "running", pid=child.pid, gpu=gpu, started=time.time())
                try:
                    code = child.wait(timeout=max(1, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
            if code or not os.path.exists(row["completion"]):
                raise RuntimeError("Stage exited %d or no completion file" % code)
            done = read(row["completion"])
            if done["status"] != "complete" or done["protocol_sha256"] != reg:
                raise RuntimeError("Wrong stage completion")
            update(row, "complete", gpu=gpu, finished=time.time())
            with mutex:
                print("[complete] "+row["name"], flush=True)
        except Exception as error:
            with mutex:
                errors.append(dict(task=row["name"], error=str(error)))
            update(row, "failed", gpu=gpu, detail=str(error))
            with mutex:
                print("[failed] %s: %s" % (row["name"], error), flush=True)

    def group(tasks):
        pending = deque(tasks)

        def worker(gpu):
            while True:
                with mutex:
                    if not pending:
                        return
                    row = pending.popleft()
                    bad = bool(errors)
                if bad:
                    update(row, "blocked", detail="Earlier stage failed; inspect logs")
                else:
                    execute(row, gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, gpu) for gpu in [0, 1]]
            for future in futures:
                future.result()

    for row in all_tasks:
        update(row, "queued")
    try:
        for gpu in [0, 1]:
            handle = output_path(ROOT/"status"/("gpu%d.resource.lock" % gpu)).open("a")
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("GPU resource reservation timeout")
                    time.sleep(20)
            gpu_guards.append(handle)
            while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Existing GPU task not finished")
                time.sleep(20)
        execute(prepare, 0)
        for jobs in [extract, train, held, [summary]]:
            group(jobs)
    except Exception as error:
        errors.append(dict(task="queue", error=str(error)))
    finally:
        for handle in gpu_guards:
            handle.close()
    for row in all_tasks:
        if outcomes.get(row["name"]) == "queued":
            update(row, "blocked", detail="Queue could not complete prerequisites")
    status = "failed" if errors else "complete"
    atomic_json(status_path, dict(status=status, pid=os.getpid(), protocol_sha256=reg,
                lane=lane, outcomes=outcomes, errors=errors, report=str(OUT/lane/"summary.json"),
                note="Bounded exploratory pilot only. No automatic confirmation, extra seeds or shutdown."))
    print("[VX-finished]", status, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
