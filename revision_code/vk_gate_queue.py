"""Detached finite V-K confirmation, after an existing GPU queue releases its lane."""
import argparse
import fcntl
import os
import signal
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage
from vk_gate_protocol import OUT, SPEC, read, register, verify, paths

STATE = ROOT / "status/VK_queue.json"


def process_alive(pid):
    if not pid: return False
    try:
        os.kill(int(pid), 0)
        from pathlib import Path
        stat = Path("/proc/%d/stat" % int(pid))
        return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError): return False


def lane_available(gpu):
    path = ROOT / "status" / ("R16_gpu%d.json" % gpu)
    row = read(path) if path.exists() else {}
    if (row.get("status") in ["running", "waiting_gpu", "waiting_dependency"]
            and process_alive(row.get("pid"))):
        return False, "R16 queue active: " + row.get("task", "waiting")
    try:
        active = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, "GPU telemetry unavailable: " + str(exc)
    return not active, "other compute processes: " + active if active else "idle"


def stage_plan(task):
    return [(task, True, s) for s in ["export", "features", "evaluate"]] + [
        (task, False, s) for s in ["export", "features", "evaluate", "decision"]]


def completion_path(task, smoke, stage):
    out, _ = paths(task, smoke)
    return OUT / task / "decision.json" if stage == "decision" else out / {
        "export":"export", "features":"features", "evaluate":"selective"}[stage] / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true"); args = parser.parse_args()
    ensure_storage()
    if args.register:
        print("[REGISTERED] " + register(), flush=True); return
    reg = verify(full=True)
    lock = (ROOT / "status/VK_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    outcomes = {}; current = {}; gpu_lock = None; gpu = None; child = None
    state = dict(pid=os.getpid(), command=[__file__], completion=str(OUT / "summary.json"),
        log=str(ROOT / "logs/VK_queue.log"), protocol_sha256=reg,
        plan="wait for R16 GPU release; 3-image native smoke ->1000 SGCls gate ->SGDet only on pass; no fitting")

    def publish(status, **extra):
        state.update(status=status, current=current, outcomes=outcomes, gpu=[] if gpu is None else [gpu],
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
        atomic_json(STATE, state)

    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()

    def interrupted(signum, frame):
        raise InterruptedError("V-K queue received signal %d" % signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if (OUT / "summary.json").exists():
            value = read(OUT / "summary.json")
            if value.get("status") != "complete" or value.get("protocol_sha256") != reg:
                raise RuntimeError("Incompatible completed V-K queue")
            publish("complete", accepted=value["accepted"], reason=value["reason"]); return
        wait_deadline = time.monotonic() + SPEC["waiting_hours"] * 3600
        while gpu_lock is None:
            reasons = {}
            for candidate in SPEC["gpu_preference"]:
                ok, why = lane_available(candidate); reasons[str(candidate)] = why
                if not ok: continue
                lease = (ROOT / "status" / ("gpu%d.resource.lock" % candidate)).open("a")
                try: fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); reasons[str(candidate)] = "resource lock held"; continue
                ok, why = lane_available(candidate)
                if ok:
                    gpu, gpu_lock = candidate, lease; break
                lease.close(); reasons[str(candidate)] = why
            if gpu_lock is None:
                if time.monotonic() >= wait_deadline: raise TimeoutError("V-K queue wait budget reached")
                publish("waiting_gpu", reason="Waiting without reserving GPU or interrupting R16", waiting_on=reasons)
                print("[WAITING] " + __import__("json").dumps(reasons), flush=True); time.sleep(60)
        deadline = time.time() + SPEC["execution_hours"] * 3600
        publish("running", reason="Exclusive GPU acquired; frozen candidate only", waiting_on={},
            deadline_epoch=deadline, deadline_local=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)))
        passed = False; reason = "not_evaluated"
        for task in SPEC["tasks"]:
            for _, smoke, stage in stage_plan(task):
                if verify() != reg: raise RuntimeError("Registration changed while waiting")
                completion = completion_path(task, smoke, stage)
                name = "VK_%s_%s_%s" % (task, "smoke" if smoke else "gate", stage)
                if completion.exists():
                    record = read(completion)
                    if record.get("status") != "complete" or record.get("protocol_sha256") != reg:
                        raise RuntimeError("Incompatible V-K stage: " + name)
                    outcomes[name] = "complete"; continue
                modern = stage == "features"
                script = "vk_gate_features.py" if modern else "vk_gate_native.py"
                command = [os.environ["SGG_MODERN_PYTHON" if modern else "SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code" / script)]
                if not modern: command += [stage]
                command += ["--task", task, "--deadline", str(deadline)]
                if smoke: command += ["--smoke"]
                log = ROOT / "logs" / (name + ".log")
                current.clear(); current[str(gpu)] = name
                with log.open("a", buffering=1) as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(gpu_lock.fileno(),),
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1"))
                    publish("running", child_pid=child.pid, progress_file=str(completion.parent / "progress.json"),
                        stage_log=str(log), stage_completion=str(completion))
                    print("[START] %s GPU%d pid=%d log=%s" % (name, gpu, child.pid, log), flush=True)
                    code = child.wait(timeout=max(1, deadline - time.time()))
                good = code == 0 and completion.exists() and read(completion).get("status") == "complete"
                outcomes[name] = "complete" if good else "failed"
                print("[%s] %s" % (outcomes[name].upper(), name), flush=True)
                if not good: raise RuntimeError("V-K runtime failure; later stages stopped: " + name)
            verdict = read(OUT / task / "decision.json")
            passed = verdict["accepted"]; reason = task + ("_joint_gate_passed" if passed else "_joint_gate_rejected")
            if not passed: break
        value = dict(status="complete", accepted=passed, reason=reason, protocol_sha256=reg,
            checkpoint_sha256=read(OUT / "sgcls/decision.json")["checkpoint_sha256"],
            completed_stages=outcomes, test_evaluated=False, extra_seeds_run=False,
            final_efficacy_claim=False, no_automatic_search=True)
        atomic_json(OUT / "summary.json", value)
        current.clear(); publish("complete", accepted=passed, reason=reason, child_pid=None)
        print("[FINISHED] " + __import__("json").dumps(value), flush=True)
    except BaseException as exc:
        stop_child(); publish("failed", reason=type(exc).__name__ + ": " + str(exc), child_pid=None)
        raise
    finally:
        if gpu_lock is not None: gpu_lock.close()
        lock.close()


if __name__ == "__main__": main()
