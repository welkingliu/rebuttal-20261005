"""Run a dependency-checked finite queue; never treat a failed gate as completion."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from common import ROOT, ensure_storage, atomic_json, sha256


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def complete(path):
    if not Path(path).is_file():
        return False
    try:
        value=json.loads(Path(path).read_text())
        if value.get("smoke") or value.get("full_split") is False:
            return False
        return value.get("status") == "complete"
    except (OSError,ValueError):
        return False


def dependency_state(job):
    waiting=[]
    for dep in job.get("dependencies",[]):
        if complete(dep["completion"]):
            continue
        upstream=ROOT/"status"/(dep.get("job","")+".json")
        if upstream.is_file() and json.loads(upstream.read_text()).get("status") in ("failed","blocked"):
            return "blocked",dict(reason="Upstream failed",dependency=dep)
        if dep.get("pid"):
            try:
                os.kill(dep["pid"],0)
                command=Path("/proc/%d/cmdline"%dep["pid"]).read_bytes().decode().replace("\0"," ")
                alive=dep.get("expected_command", "") in command
            except (ProcessLookupError,FileNotFoundError):
                alive=False
            if not alive:
                return "blocked",dict(reason="Dependency process exited without completion",dependency=dep)
        waiting.append(dep)
    return ("waiting",dict(dependencies=waiting)) if waiting else ("ready",{})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane",required=True,choices=["gpu0","gpu1","cpu"])
    parser.add_argument("--job",help="Resume one named job without restarting the lane")
    args=parser.parse_args()
    ensure_storage()
    lane_key=args.lane+("_"+args.job if args.job else "")
    lock=(ROOT/"status"/(lane_key+".lock")).open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((ROOT/"manifests/queue.json").read_text())
    jobs=[j for j in config["jobs"] if j["lane"]==args.lane and (not args.job or j["id"]==args.job)]
    pending=list(jobs)
    while pending:
        ready=[]
        for job in list(pending):
            state=ROOT/"status"/(job["id"]+".json")
            if job.get("blocked_reason"):
                atomic_json(state,dict(status="blocked",reason=job["blocked_reason"],updated_at=now()))
                pending.remove(job)
            elif complete(job["completion"]):
                atomic_json(state,dict(status="complete",completion=job["completion"],observed_at=now()))
                pending.remove(job)
            else:
                status,detail=dependency_state(job)
                if status=="ready":
                    ready.append(job)
                else:
                    atomic_json(state,dict(status=status,updated_at=now(),**detail))
                    if status=="blocked":
                        pending.remove(job)
        if ready:
            job=ready[0]
            state=ROOT/"status"/(job["id"]+".json")
            ensure_storage()
            env=os.environ.copy()
            resource=None
            if args.lane.startswith("gpu"):
                resource=(ROOT/"status"/(args.lane+".resource.lock")).open("w")
                fcntl.flock(resource,fcntl.LOCK_EX)
                env["CUDA_VISIBLE_DEVICES"]=args.lane[-1]
                # Do not start over an unrelated CUDA workload.
                while True:
                    query=subprocess.check_output(["nvidia-smi","-i",args.lane[-1],
                        "--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                    if not query:
                        break
                    atomic_json(state,dict(status="waiting_gpu",compute_pids=query,updated_at=now()))
                    time.sleep(20)
            attempt=time.strftime("%Y%m%d_%H%M%S")
            log=ROOT/"logs"/(job["id"]+"_"+attempt+".log")
            command=job["command"]
            print("[START]",job["id"],now(),flush=True)
            with log.open("w") as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env,stdin=subprocess.DEVNULL)
                running=dict(status="running",pid=process.pid,command=command,log=str(log),started_at=now())
                atomic_json(state,running)
                code=process.wait()
            success=code==0 and complete(job["completion"])
            atomic_json(state,dict(running,status="complete" if success else "failed",
                                   returncode=code,finished_at=now(),completion=job["completion"]))
            print("[COMPLETE]" if success else "[FAILED]",job["id"],now(),flush=True)
            if resource:
                resource.close()
            pending.remove(job)
        elif pending:
            time.sleep(15)
    atomic_json(ROOT/"status"/(lane_key+"_queue.json"),dict(status="finished",finished_at=now(),
        states={j["id"]:json.loads((ROOT/"status"/(j["id"]+".json")).read_text())["status"] for j in jobs}))


if __name__=="__main__":
    main()
