"""Server-only, at-most-12-hour V-F queue; never interrupts the V-E lane."""
import argparse
import fcntl
import os
import subprocess
import sys
import time

from common import ROOT, atomic_json, ensure_storage
from ve_experiment import read
from vf_protocol import OUT, SPEC, register, verify

STATE = ROOT / "status/VF_queue.json"
CURRENT, OUTCOMES = {}, {}
DEADLINE = None


def publish(status="running", **extra):
    atomic_json(STATE, dict(status=status, pid=os.getpid(), current=CURRENT.copy(), outcomes=OUTCOMES.copy(),
        deadline_epoch=DEADLINE, deadline_local=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(DEADLINE)),
        plan="V-F SGCls GPU1 -> fresh gate -> SGDet GPU0 after V-E; stop on rejection; no test/extra seeds", **extra))


def read_if(path):
    return read(path) if path.exists() else {}


def run(task, stage, mode="native"):
    verify()
    if time.time() >= DEADLINE:
        raise TimeoutError("12-hour queue budget exhausted")
    gpu = 1 if task == "sgcls" else 0
    out = OUT / task / ({"prepare": "prepare", "train": "training", "evaluate": "gate"}.get(stage, ""))
    if stage in ("train", "evaluate"):
        out = out / mode
    completion = OUT / task / "decision.json" if stage == "decision" else out / "summary.json"
    name = "VF_%s_%s_%s" % (task, stage, mode)
    cmd = [os.environ["SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code/vf_experiment.py"),
           "--task", task, "--stage", stage, "--mode", mode, "--deadline", str(DEADLINE)]
    state_path = ROOT / "status" / (name + ".json")
    record = dict(status="waiting_gpu", gpu=[gpu], command=cmd, completion=str(completion),
                  progress_file=str(out / "progress.json"))
    if read_if(completion).get("status") == "complete":
        if read(completion).get("registration_sha256") != verify():
            raise RuntimeError("Cannot resume incompatible V-F stage")
        OUTCOMES[name] = "complete"
        atomic_json(state_path, dict(record, status="complete", resumed=True)); publish()
        return
    atomic_json(state_path, record)
    lock = (ROOT / "status" / ("gpu%d.resource.lock" % gpu)).open("a")
    try:
        while True:
            if time.time() >= DEADLINE:
                raise TimeoutError("GPU reservation exceeded queue budget")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(15)
        while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            if time.time() >= DEADLINE:
                raise TimeoutError("GPU is occupied beyond queue budget")
            time.sleep(15)
        log = ROOT / "logs" / (name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
        with log.open("w") as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            record.update(status="running", pid=child.pid, log=str(log))
            CURRENT[str(gpu)] = name
            atomic_json(state_path, record); publish()
            code = child.wait()
        CURRENT.pop(str(gpu), None)
        ok = code == 0 and read_if(completion).get("status") == "complete"
        OUTCOMES[name] = "complete" if ok else "failed"
        atomic_json(state_path, dict(record, status="complete" if ok else "failed", returncode=code)); publish()
        print("[%s] %s log=%s" % ("COMPLETE" if ok else "FAILED", name, log), flush=True)
        if not ok:
            raise RuntimeError("Stage failed; no downstream training: " + name)
    finally:
        lock.close()


def task_pilot(task):
    run(task, "prepare")
    for mode in SPEC["modes"]:
        run(task, "train", mode)
    development = read(OUT / task / "training/score_protected/summary.json")
    if not development["development_eligible"]:
        atomic_json(OUT / task / "decision.json", dict(status="complete", accepted=False,
            reason="development_object_gain_below_0.5pp; fresh gate not evaluated",
            development=development["development"], registration_sha256=verify(), exploratory=True, test_evaluated=False))
        return False
    for mode in ["native"] + SPEC["modes"]:
        run(task, "evaluate", mode)
    run(task, "decision")
    return read(OUT / task / "decision.json")["accepted"]


def main():
    global DEADLINE
    p = argparse.ArgumentParser(); p.add_argument("--register", action="store_true")
    p.add_argument("--hours", type=float, default=12.)
    args = p.parse_args(); ensure_storage()
    if args.register:
        register(); print("[REGISTERED] V-F finite exploratory pilot"); return
    if not 0 < args.hours <= 12:
        raise ValueError("Queue budget must be in (0,12] hours")
    verify()
    lock = (ROOT / "status/VF_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    DEADLINE = time.time() + args.hours * 3600
    publish()
    try:
        if read(ROOT / "results/VE_failure_audit_20261001/summary.json")["status"] != "complete":
            raise RuntimeError("Failure audit must complete first")
        passed = task_pilot("sgcls")
        if passed:
            CURRENT["0"] = "waiting for V-E controller to terminate before V-F SGDet"
            publish()
            while True:
                row = read(ROOT / "status/VE_queue.json")
                if row["status"] == "complete":
                    break
                if row["status"] != "running":
                    raise RuntimeError("V-E needs attention; do not take its reserved GPU lane")
                os.kill(row["pid"], 0)
                if time.time() >= DEADLINE:
                    raise TimeoutError("V-E still running at V-F budget boundary")
                time.sleep(30)
            CURRENT.pop("0", None)
            passed = task_pilot("sgdet")
        atomic_json(OUT / "summary.json", dict(status="complete", accepted=passed,
            registration_sha256=verify(), exploratory_pilot_only=True, test_evaluated=False,
            extra_seeds_run=False, next_step="review pilot; no automatic search or claim of final efficacy"))
        publish("complete", accepted=passed)
    except Exception as exc:
        publish("budget_exhausted" if isinstance(exc, TimeoutError) else "needs_attention", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
