"""Finite reproduction repair queue; completion and metric acceptance are separate."""
import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import time

from common import ROOT, atomic_json, ensure_storage, sha256
from repro_protocol import RUN, SPEC, sources, verify


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def jobs():
    entries = [
        ("sgcls_smoke", [1], ["evaluate", "--task", "sgcls", "--limit", "3"], "sgcls/eval_corrected/test_smoke", []),
        ("train_smoke", [0, 1], ["train", "--task", "sgdet", "--smoke_steps", "2"], "sgdet/training_smoke", []),
        ("sgcls_val", [1], ["evaluate", "--task", "sgcls", "--split", "val"], "sgcls/eval_corrected/val", ["sgcls_smoke"]),
        ("sgcls_test", [1], ["evaluate", "--task", "sgcls"], "sgcls/eval_corrected/test", ["sgcls_smoke"]),
        ("sgdet_train", [0, 1], ["train", "--task", "sgdet"], "sgdet/training", ["train_smoke"]),
        ("sgdet_val", [0], ["evaluate", "--task", "sgdet", "--variant", "retrained", "--split", "val"], "sgdet/eval_retrained/val", ["sgdet_train"]),
        ("sgdet_test", [0], ["evaluate", "--task", "sgdet", "--variant", "retrained"], "sgdet/eval_retrained/test", ["sgdet_train"]),
    ]
    return [dict(name="R9_"+n, gpu=g, arguments=a, completion=str(RUN/p/"summary.json"),
                 progress_file=str(RUN/p/"progress.json"), dependencies=["R9_"+d for d in deps])
            for n,g,a,p,deps in entries]


def read(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return {}


def register():
    path = ROOT / "manifests/R9_reproduction.json"
    if path.exists():
        verify()
    else:
        atomic_json(path, dict(spec=SPEC, sources=sources(), registered_at=now()))
    atomic_json(ROOT / "manifests/R9_jobs.json", dict(jobs=jobs(), registration_sha256=sha256(path)))
    print("[REGISTERED] Fixed repair protocol, no test-driven search", flush=True)


def run(job):
    ensure_storage(); verify()
    state = ROOT / "status" / (job["name"]+".json")
    if read(job["completion"]).get("status") == "complete":
        atomic_json(state, dict(job, status="complete", resumed=True))
        return True
    locks = []
    record = dict(job, status="waiting_gpu", updated_at=now())
    atomic_json(state, record)
    try:
        for gpu in sorted(job["gpu"]):
            lock = (ROOT / "status" / ("gpu%d.resource.lock" % gpu)).open("a")
            locks.append(lock); fcntl.flock(lock, fcntl.LOCK_EX)
        for gpu in job["gpu"]:
            while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                                          "--format=csv,noheader,nounits"], text=True).strip():
                time.sleep(20)
        command = [os.environ["SGG_NATIVE_PYTHON"], "-u"]
        if len(job["gpu"]) > 1:
            command += ["-m", "torch.distributed.launch", "--nproc_per_node=2", "--master_port=19629", "--use_env"]
        command += [str(ROOT / "code/repro_experiment.py")] + job["arguments"]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,job["gpu"]))
        log = ROOT / "logs" / (job["name"]+"_"+time.strftime("%Y%m%d_%H%M%S")+".log")
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            record.update(status="running", pid=process.pid, command=command, log=str(log), started_at=now())
            atomic_json(state, record)
            rc = process.wait()
        ok = rc == 0 and read(job["completion"]).get("status") == "complete"
        record.update(status="complete" if ok else "failed", returncode=rc, finished_at=now())
        atomic_json(state, record)
        print("[%s] %s log=%s" % (record["status"],job["name"],log), flush=True)
        return ok
    except Exception as exc:
        atomic_json(state, dict(record, status="failed", error=str(exc), finished_at=now()))
        print("[FAILED] %s: %s" % (job["name"],exc), flush=True)
        return False
    finally:
        for lock in reversed(locks):
            lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    args = parser.parse_args(); ensure_storage()
    if args.register:
        register(); return
    verify()
    lock = (ROOT / "status/R9_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError("At least 100 GiB free space required before this queue")
    sequence = jobs()[:2] if args.smoke_only else jobs()
    outcomes = {}
    for position, job in enumerate(sequence, 1):
        atomic_json(ROOT / "status/R9_queue.json", dict(status="running", pid=os.getpid(),
                    position=position, total=len(sequence), current=job["name"],
                    next=[x["name"] for x in sequence[position:]], outcomes=outcomes, updated_at=now()))
        failed = [d for d in job["dependencies"] if outcomes.get(d) != "complete"]
        if failed:
            outcomes[job["name"]] = "blocked"
            atomic_json(ROOT / "status" / (job["name"]+".json"),
                        dict(job, status="blocked", reason="Unsuccessful dependencies: " + ", ".join(failed)))
        else:
            outcomes[job["name"]] = "complete" if run(job) else "failed"
    ok = all(v == "complete" for v in outcomes.values())
    atomic_json(ROOT / "status/R9_queue.json", dict(status="complete" if ok else "needs_attention",
                outcomes=outcomes, smoke_only=args.smoke_only, finished_at=now(),
                note="Execution completion is not reference-metric acceptance"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
