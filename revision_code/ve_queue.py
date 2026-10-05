"""Run V-E after native smoke; reuse GPU0 only when the R15 lane is terminal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import subprocess
import threading
import time

from common import ROOT, atomic_json, ensure_storage
from ve_protocol import OUT, SPEC, register, verify

STATE = ROOT / "status/VE_queue.json"
MUTEX = threading.Lock()
CURRENT, OUTCOMES = {}, {}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def publish(status="running", **extra):
    with MUTEX:
        atomic_json(STATE, dict(status=status, pid=os.getpid(), current=CURRENT.copy(),
                    outcomes=OUTCOMES.copy(), plan="GPU1 SGCls; GPU0 SGDet after R15; fixed pilot gate before expansion", **extra))


def run(task, stage, mode="native", seed=17, split="gate", gpu=1):
    verify()
    name = "VE_%s_%s_%s_s%d%s" % (task, stage, mode, seed, "_" + split if stage == "evaluate" else "")
    if stage == "smoke":
        out = OUT / task / "smoke"
    elif stage == "train":
        out = OUT / task / "training" / (mode + "_seed%d" % seed)
    else:
        out = OUT / task / split / (mode + "_seed%d" % seed)
    command = [os.environ["SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code/ve_experiment.py"),
               "--stage", stage, "--task", task, "--mode", mode, "--seed", str(seed), "--split", split]
    state = ROOT / "status" / (name + ".json")
    record = dict(status="waiting_gpu", gpu=[gpu], command=command,
                  completion=str(out / "summary.json"), progress_file=str(out / "progress.json"))
    existing = read(out / "summary.json")
    if existing.get("status") == "complete":
        if existing.get("registration_sha256") != verify():
            raise RuntimeError("Completed V-E stage belongs to a different registration")
        with MUTEX:
            OUTCOMES[name] = "complete"
        atomic_json(state, dict(record, status="complete", resumed=True))
        publish()
        return True
    atomic_json(state, record)
    lock = (ROOT / "status" / ("gpu%d.resource.lock" % gpu)).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True).strip():
            time.sleep(20)
        log = ROOT / "logs" / (name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            record.update(status="running", pid=child.pid, log=str(log))
            atomic_json(state, record)
            with MUTEX:
                CURRENT[str(gpu)] = name
            publish()
            rc = child.wait()
        ok = rc == 0 and read(out / "summary.json").get("status") == "complete"
        with MUTEX:
            CURRENT.pop(str(gpu), None)
            OUTCOMES[name] = "complete" if ok else "failed"
        atomic_json(state, dict(record, status="complete" if ok else "failed", returncode=rc))
        publish()
        print("[%s] %s log=%s" % ("COMPLETE" if ok else "FAILED", name, log), flush=True)
        return ok
    finally:
        lock.close()


def wait_r15():
    with MUTEX:
        CURRENT["0"] = "waiting for R15 before V-E SGDet"
    publish()
    while True:
        row = read(ROOT / "status/R15_queue.json")
        if row.get("status") in ("complete", "needs_attention"):
            return
        if row.get("status") == "running" and row.get("pid"):
            try:
                os.kill(row["pid"], 0)
            except OSError:
                raise RuntimeError("R15 process disappeared; inspect before taking its reserved lane")
        time.sleep(30)


def parallel_lanes(callback):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, task, gpu) for task, gpu in [("sgcls", 1), ("sgdet", 0)]]
        outcomes = [future.result() for future in futures]
    return all(outcomes)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--register", action="store_true")
    args = p.parse_args(); ensure_storage()
    if args.register:
        register(); print("[REGISTERED] V-E fixed pilot, active-gradient smoke before training", flush=True); return
    verify()
    lock = (ROOT / "status/VE_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish()
    try:
        # Check both task interfaces immediately, while R15 uses the other GPU.
        for task in SPEC["tasks"]:
            if not run(task, "smoke", gpu=1):
                publish("needs_attention", reason="native_integration_smoke_failed"); return

        def pilot(task, gpu):
            if gpu == 0:
                wait_r15()
            for mode in SPEC["modes"]:
                if not run(task, "train", mode, gpu=gpu):
                    return False
            for mode in ["native"] + SPEC["modes"]:
                if not run(task, "evaluate", mode, gpu=gpu):
                    return False
            return True

        if not parallel_lanes(pilot):
            publish("needs_attention", reason="pilot_execution_failed"); return
        command = [os.environ["SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code/ve_experiment.py"), "--stage", "decision"]
        with (ROOT / "logs/VE_decision.log").open("a") as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
        decision = read(OUT / "pilot_decision.json")
        if not decision["accepted"]:
            atomic_json(OUT / "summary.json", dict(status="complete", accepted=False,
                        stopped_after_pilot=True, decision=str(OUT / "pilot_decision.json")))
            publish("complete", accepted=False, reason="validation_gate_not_met; no seeds23/31 or test")
            return

        def confirm(task, gpu):
            for seed in [23, 31]:
                for mode in SPEC["modes"]:
                    if not run(task, "train", mode, seed, gpu=gpu):
                        return False
            if not run(task, "evaluate", "native", split="test", gpu=gpu):
                return False
            for seed in SPEC["seeds"]:
                for mode in SPEC["modes"]:
                    if not run(task, "evaluate", mode, seed, "test", gpu):
                        return False
            return True

        if not parallel_lanes(confirm):
            publish("needs_attention", reason="confirmation_execution_failed"); return
        atomic_json(OUT / "summary.json", dict(status="complete", pilot_accepted=True,
                    formal_results="All seeds and controls retained; compare full-test metrics before claiming mitigation success"))
        publish("complete", pilot_accepted=True)
    except Exception as exc:
        publish("needs_attention", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
