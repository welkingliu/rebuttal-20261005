"""Independent follow-ups, GPU locks and V-C dependency; no current job interruption."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import json
import os
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage
from extension_protocol import SPEC, hashes, verify


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(name, python, script, arguments, completion, gpu=None):
    verify()
    state = ROOT / "status" / (name + ".json")
    if completion.exists() and json.loads(completion.read_text()).get("status") == "complete":
        atomic_json(state, dict(status="complete", resumed=True, completion=str(completion)))
        return
    lock = None; env = os.environ.copy()
    if gpu is not None:
        lock = (ROOT / "status" / ("gpu%d.resource.lock" % gpu)).open("a")
        atomic_json(state, dict(status="waiting_gpu", gpu=gpu, updated_at=now()))
        fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            time.sleep(20)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [python, "-u", str(ROOT / "code" / script)] + arguments
    log = ROOT / "logs" / (name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT, env=env)
            value = dict(status="running", pid=process.pid, command=command, log=str(log), gpu=gpu, started_at=now())
            atomic_json(state, value)
            rc = process.wait()
        ok = rc == 0 and completion.exists() and json.loads(completion.read_text()).get("status") == "complete"
        atomic_json(state, dict(value, status="complete" if ok else "failed", returncode=rc, finished_at=now()))
        if not ok:
            raise RuntimeError("Failed " + name + ": " + str(log))
    finally:
        if lock is not None:
            lock.close()


def vd():
    state = ROOT / "status/VD_dependency.json"
    while True:
        vc = json.loads((ROOT / "status/VC_queue.json").read_text())["status"]
        atomic_json(state, dict(status="waiting_VC" if vc not in ("complete", "stopped_by_validation", "failed") else vc, updated_at=now()))
        if vc == "failed":
            raise RuntimeError("V-C technical failure must be repaired before V-D reuse")
        if vc in ("complete", "stopped_by_validation"):
            break
        time.sleep(30)
    py = os.environ["SGG_NATIVE_PYTHON"]
    run("VD_shortlist", py, "vd_experiment.py", ["shortlist"], ROOT / "results/VD/shortlist.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, "VD_" + t, py, "vd_experiment.py", ["task", "--task", t],
                               ROOT / "results/VD" / t / "summary.json", gpu)
                   for t, gpu in [("sgcls", 0), ("sgdet", 1)]]
        for future in futures:
            future.result()
    run("VD_report", py, "vd_experiment.py", ["report"], ROOT / "results/VD/summary.json")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--register", action="store_true")
    args = p.parse_args(); ensure_storage()
    manifest = ROOT / "manifests/review_extension.json"
    if args.register:
        if manifest.exists():
            verify()
        else:
            atomic_json(manifest, dict(spec=SPEC, sources=hashes(), registered_at=now()))
        print("[REGISTERED] R6/R7/V-D finite follow-up", flush=True); return
    verify()
    lock = (ROOT / "status/review_extension.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(ROOT / "status/review_extension.json", dict(status="running", pid=os.getpid(), started_at=now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {
            "R6": pool.submit(run, "R6_sgtr_live", "/home/USER/miniconda3/envs/sgtr_runtime/bin/python",
                               "modern_live_control.py", [], ROOT / "results/R6_sgtr_live/summary.json", 0),
            "R7": pool.submit(run, "R7_coverage", os.environ["SGG_NATIVE_PYTHON"], "external_coverage.py", [], ROOT / "results/R7_coverage/summary.json"),
            "VD": pool.submit(vd),
        }
        results = {}
        for name, future in jobs.items():
            try:
                future.result(); results[name] = "complete"
            except Exception as exc:
                results[name] = str(exc)
    atomic_json(ROOT / "status/review_extension.json", dict(status="complete" if all(x == "complete" for x in results.values()) else "needs_attention",
                results=results, finished_at=now()))


if __name__ == "__main__":
    main()
