"""Serial R15 lane with resource lock, observable progress and strict stage gates."""
import fcntl
import json
import os
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage
from r15_identity import OUT, verify


def main():
    ensure_storage()
    verify()
    lock = (ROOT / "status/R15_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = ROOT / "status/R15_queue.json"
    outcomes = {}
    for stage in ["smoke", "validation", "dev", "test"]:
        out = OUT / "transformer" / stage
        name = "R15_transformer_" + stage
        dest = ROOT / "status" / (name + ".json")
        row = dict(name=name, gpu=[0], status="waiting_gpu", completion=str(out / "summary.json"),
                   progress_file=str(out / "progress.json"))
        atomic_json(state, dict(status="running", pid=os.getpid(), current=stage, outcomes=outcomes))
        atomic_json(dest, row)
        gpu_lock = (ROOT / "status/gpu0.resource.lock").open("a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        try:
            verify()
            while subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                          "--format=csv,noheader,nounits"], text=True).strip():
                time.sleep(20)
            log = ROOT / "logs" / (name + "_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
            command = [os.environ["SGG_NATIVE_PYTHON"], "-u", str(ROOT / "code/r15_identity.py"),
                       "--family", "transformer", "--stage", stage]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            with log.open("w") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                row.update(status="running", pid=child.pid, command=command, log=str(log))
                atomic_json(dest, row)
                rc = child.wait()
            summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
            passed = rc == 0 and summary.get("status") == "complete" and summary.get("invariance_passed")
            if stage == "dev":
                passed = passed and summary.get("development_gate_passed")
            outcomes[stage] = "complete" if passed else "failed"
            atomic_json(dest, dict(row, status=outcomes[stage], returncode=rc))
            print("[%s] %s log=%s" % (outcomes[stage], stage, log), flush=True)
            if not passed:
                atomic_json(state, dict(status="needs_attention", outcomes=outcomes, failed_stage=stage))
                return
        finally:
            gpu_lock.close()
    atomic_json(state, dict(status="complete", outcomes=outcomes))


if __name__ == "__main__":
    main()
