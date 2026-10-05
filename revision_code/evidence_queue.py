"""Finite independent queues: mechanism GPU0, corrected Motifs GPU1, release CPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, atomic_json, ensure_storage
from evidence_completion import read


def jobs(lane):
    native = os.environ["SGG_NATIVE_PYTHON"]
    modern = os.environ["SGG_MODERN_PYTHON"]
    code = ROOT / "code"
    if lane == "R22":
        return [("R22_response_evidence", [native,str(code/"evidence_response_index.py"),"--wait"],
                 ROOT/"results/R22_response_evidence")]
    if lane == "R19":
        return [("R19_"+task, [native, str(code/"r19_vk_decomposition.py"), "--task", task],
                 ROOT/"results/R19_vk_mechanism"/task/"formal") for task in ("sgdet", "sgcls")]
    if lane == "R19A":
        return [("R19_exact_order_"+task, [native, str(code/"r19_fixed_order.py"), "--task", task],
                 ROOT/"results/R19_vk_fixed_order"/task) for task in ("sgdet", "sgcls")]
    if lane == "R20N":
        return [("R20_native_"+stage, [native, str(code/"r20_motifs_native_bridge.py"), stage],
                 ROOT/"results/R20_plain_motifs_native_bridge"/stage) for stage in ("validation", "identity", "visual")]
    if lane == "R20":
        return [("R20_"+stage, [native, str(code/"r20_motifs_bridge.py"), stage],
                 ROOT/"results/R20_plain_motifs_bridge"/stage) for stage in ("validation", "identity", "visual")]
    return [("R21_release", [modern, str(code/"r21_release_audit.py")], ROOT/"results/R21_release_execution")]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("lane", choices=["R19", "R19A", "R20", "R20N", "R21", "R22"])
    a = p.parse_args(); ensure_storage()
    # Keep duplicate launch attempts from competing for the same outputs/GPU.
    lock = (ROOT / "status" / (a.lane+".lock")).open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = jobs(a.lane); gpu = 0 if a.lane=="R19" else 1 if a.lane in ("R20", "R20N") else None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), PYTHONUNBUFFERED="1")
    outcomes = {}; state = ROOT / "status" / (a.lane+"_queue.json")
    if a.lane == "R19" and read(ROOT/"results/R19_vk_mechanism/sgdet/smoke/summary.json")["status"] != "complete":
        raise RuntimeError("V-K native replay smoke required")
    if a.lane == "R20" and not read(ROOT/"results/R20_plain_motifs_bridge/smoke/summary.json")["invariance_passed"]:
        raise RuntimeError("Corrected-Motifs intervention smoke required")
    if a.lane == "R20N":
        if not read(ROOT/"results/R20_plain_motifs_native_bridge/smoke/summary.json")["invariance_passed"]:
            raise RuntimeError("Full-native smoke required")
        if not read(ROOT/"results/R20_capture_audit/summary.json")["exact_native_reproduced"]:
            raise RuntimeError("Native cache amendment not verified")
        historical = {}
        for name in ("R20_identity", "R20_visual", "R20_queue"):
            path=ROOT/"status"/(name+".json")
            if path.exists():
                old=read(path); historical[name]=old
                old.update(status="stopped",reason="Superseded by separately registered native-all-pairs R20N; old partial outputs retained",superseded_by="R20N")
                atomic_json(path,old)
        atomic_json(ROOT/"results/R20_capture_audit/previous_queue_statuses.json",historical)
    if a.lane == "R19A":
        start=time.monotonic()
        while not all((ROOT/"results/R19_vk_mechanism"/task/"formal/summary.json").exists() for task in ("sgdet","sgcls")):
            dependency=read(ROOT/"status/R19_queue.json")
            if dependency.get("status")=="failed" or time.monotonic()-start>7200:
                atomic_json(state,dict(status="blocked",reason="R19 native export did not finish",pid=os.getpid()))
                return
            atomic_json(state,dict(status="waiting_dependency",pid=os.getpid(),current="R19 native cache export",plan="Exact native-rank replay on CPU after GPU export"))
            time.sleep(15)
    for name, command, folder in queue:
        summary = folder / "summary.json"
        status = ROOT / "status" / (name+".json")
        atomic_json(status, dict(status="pending", command=command, gpu=[] if gpu is None else [gpu],
            completion=str(summary), progress_file=str(folder/"progress.json")))
    for position, (name, command, folder) in enumerate(queue, 1):
        summary = folder/"summary.json"; status = ROOT/"status"/(name+".json")
        if summary.exists() and read(summary).get("status") == "complete":
            outcomes[name] = "complete"
            atomic_json(status, dict(status="complete", completion=str(summary), command=command))
            continue
        log = ROOT/"logs"/(name+".log")
        if gpu is not None:
            # Never evict other work. Wait if an external job has occupied this GPU.
            while True:
                used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
                if used < 1000: break
                atomic_json(status, dict(status="waiting_gpu", command=command, gpu=[gpu], completion=str(summary), reason="GPU occupied; no process will be stopped"))
                time.sleep(30)
        with log.open("a") as handle:
            child = subprocess.Popen(command, env=env, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
            atomic_json(status, dict(status="running", pid=child.pid, gpu=[] if gpu is None else [gpu],
                command=command, completion=str(summary), progress_file=str(folder/"progress.json"), log=str(log)))
            atomic_json(state, dict(status="running", pid=os.getpid(), current=name, position=position,
                total=len(queue), outcomes=outcomes, next=[q[0] for q in queue[position:]],
                plan="Finite evidence completion; no fitting/search, no test-driven selection, no automatic shutdown"))
            code = child.wait()
        valid = code == 0 and summary.exists() and read(summary).get("status") == "complete"
        outcomes[name] = "complete" if valid else "failed"
        atomic_json(status, dict(status=outcomes[name], pid=child.pid, gpu=[] if gpu is None else [gpu],
            command=command, completion=str(summary), log=str(log), returncode=code,
            reason="Execution completed; scientific verdict is in the summary" if valid else "Dependency lane stopped; inspect log"))
        if not valid:
            for blocked_name, blocked_command, blocked_folder in queue[position:]:
                atomic_json(ROOT/"status"/(blocked_name+".json"), dict(status="blocked", command=blocked_command,
                    completion=str(blocked_folder/"summary.json"), reason="Required stage failed: "+name))
            break
    atomic_json(state, dict(status="complete" if all(v=="complete" for v in outcomes.values()) and len(outcomes)==len(queue) else "failed",
        pid=os.getpid(), outcomes=outcomes, current=None))
    print(json.dumps(outcomes), flush=True)


if __name__ == "__main__": main()
