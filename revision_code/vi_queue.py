"""Detached finite V-I pilot: cross-fit -> development -> SGCls -> SGDet."""
import argparse
import fcntl
import os
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage
from vi_protocol import OUT, SPEC, register, verify, read

STATE = ROOT / "status/VI_queue.json"
MODERN = "/home/USER/miniconda3/envs/py14/bin/python"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--register", action="store_true")
    args = p.parse_args(); ensure_storage()
    if args.register:
        register(); print("[REGISTERED] V-I training-only selective fusion", flush=True); return
    verify()
    lock = (ROOT / "status/VI_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.time() + SPEC["wall_hours"] * 3600
    outcomes, current = {}, {}

    def publish(status="running", **extra):
        atomic_json(STATE, dict(status=status, pid=os.getpid(), current=current, outcomes=outcomes,
            deadline_epoch=deadline, deadline_local=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)),
            plan="GPU1 five-fold auxiliary probes + selector; dev -> SGCls native gate -> SGDet native gate; stop on rejection",
            **extra))

    def run(stage, task="sgcls"):
        reg = verify()
        modern = stage in ["develop", "gate_features"]
        script = "vi_selective.py" if modern else "vi_native.py"
        folder = (OUT / "development" if stage == "develop" else OUT / task / {
            "gate_features": "expert", "evaluate": "gate/selective", "decision": "", "smoke": "smoke", "export": "export"}[stage])
        completion = OUT / task / "decision.json" if stage == "decision" else folder / "summary.json"
        name = "VI_" + task + "_" + stage
        if completion.exists():
            record = read(completion)
            if record.get("status") != "complete" or record.get("protocol_sha256") != reg:
                raise RuntimeError("Incompatible completed VI stage")
            outcomes[name] = "complete"; publish(); return
        command = [MODERN if modern else os.environ["SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code" / script),
                   stage, "--task", task, "--deadline", str(deadline)]
        log = ROOT / "logs" / (name + ".log")
        state_path = ROOT / "status" / (name + ".json")
        record = dict(status="waiting_gpu", gpu=[1], command=command, completion=str(completion),
                      progress_file=str(folder / "progress.json"), log=str(log))
        atomic_json(state_path, record)
        gpu_lock = (ROOT / "status/gpu1.resource.lock").open("a")
        try:
            while True:
                if time.time() >= deadline:
                    raise TimeoutError("VI GPU reservation budget exhausted")
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    time.sleep(10)
            while subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
                if time.time() >= deadline:
                    raise TimeoutError("VI GPU occupied beyond budget")
                time.sleep(10)
            with log.open("a") as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1"))
                record.update(status="running", pid=child.pid)
                atomic_json(state_path, record); current["1"] = name; publish()
                try:
                    code = child.wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise TimeoutError("VI stage exhausted registered wall budget")
            current.clear()
            ok = code == 0 and completion.exists() and read(completion).get("status") == "complete"
            outcomes[name] = "complete" if ok else "failed"
            atomic_json(state_path, dict(record, status=outcomes[name], returncode=code)); publish()
            print("[%s] %s log=%s" % (outcomes[name].upper(), name, log), flush=True)
            if not ok:
                raise RuntimeError("VI stage failed: " + name)
        finally:
            gpu_lock.close()

    publish()
    try:
        run("smoke")
        run("develop")
        passed = read(OUT / "development/summary.json")["development_eligible"]
        reason = "development_requirements_not_met"
        if passed:
            for task in ["sgcls", "sgdet"]:
                if task == "sgdet":
                    run("smoke", task)
                for stage in ["export", "gate_features", "evaluate", "decision"]:
                    run(stage, task)
                passed = read(OUT / task / "decision.json")["accepted"]
                reason = task + "_native_gate_" + ("passed" if passed else "not_met")
                if not passed:
                    break
        atomic_json(OUT / "summary.json", dict(status="complete", accepted=passed, reason=reason,
            protocol_sha256=verify(), test_evaluated=False, extra_seeds_run=False, exploratory_pilot_only=True,
            next_step="review fixed pilot; no automatic hyperparameter search or final efficacy claim"))
        publish("complete", accepted=passed, reason=reason)
    except Exception as exc:
        publish("needs_attention", reason=str(exc)); raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
