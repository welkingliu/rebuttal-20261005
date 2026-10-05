"""Server-owned overnight continuation and a bounded morning status snapshot.

The morning boundary stops reporting only; it never kills an experiment.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from common import ROOT,ensure_storage,atomic_json,output_path,sha256


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def alive(pid):
    if not pid:return False
    try:
        os.kill(pid,0)
        stat=Path("/proc/%s/stat"%pid)
        return not stat.exists() or stat.read_text().rsplit(")",1)[1].split()[0]!="Z"
    except (OSError,ValueError):return False


def dependency_gpu(queue):
    if queue.get("status") in ("complete","needs_attention"):
        return 0
    outcomes=queue.get("outcomes",{})
    for gpu,family in enumerate(("tde_motifs","transformer")):
        if outcomes.get(family+"/test")=="complete":return gpu
        if any(key.startswith(family) and value in ("failed","reference_gate_failed","blocked") for key,value in outcomes.items()):return gpu
    return None


def dependency_ready(queue):
    return dependency_gpu(queue) is not None


def atomic_text(path,text):
    path=output_path(path);temporary=path.with_suffix(".tmp")
    temporary.write_text(text,encoding="utf-8");temporary.replace(path)


def snapshot(final=False):
    queues={name:read(ROOT/"status"/(name+".json")) for name in ("R9_queue","R12_queue","R13_queue")}
    for name,row in queues.items():
        if row.get("status") in ("running","waiting_dependency") and row.get("pid") and not alive(row["pid"]):
            row["status"]="stale_process_missing"
    jobs=[]
    for p in sorted((ROOT/"status").glob("*.json")):
        row=read(p)
        if row.get("status") in ("running","waiting_gpu","failed","blocked") and row.get("completion"):
            if not p.stem.startswith(("R9_","R12_","R13_")):continue
            progress=read(row.get("progress_file",""))
            jobs.append(dict(name=p.stem,status=row["status"],pid=row.get("pid"),gpu=row.get("gpu"),
                             progress=progress,log=row.get("log"),reason=row.get("reason",row.get("error"))))
            if row["status"]=="running" and not alive(row.get("pid")):
                jobs[-1]["status"]="stale_process_missing"
    try:
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError) as exc:gpu=str(exc)
    completed={}
    for label,relative in [("R11 identity channels","R11_identity_channels/summary.json"),
                           ("Probe convergence","probe_convergence/summary.json"),
                           ("GQA disjoint expansion","R13_gqa_disjoint/summary.json"),
                           ("R12 TDE","R12_sgdet_identity/tde_motifs/test/summary.json"),
                           ("R12 Transformer","R12_sgdet_identity/transformer/test/summary.json")]:
        value=read(ROOT/"results"/relative)
        if value:completed[label]={key:value[key] for key in ("status","images","completed_runs","coverage_fraction","invariance_passed") if key in value}
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    result=dict(time=stamp,morning_snapshot=final,queues=queues,jobs=jobs,gpu=gpu,completed=completed,
                note="Execution completion is separate from metric acceptance. No experiment is stopped at the reporting deadline.")
    lines=["# Overnight experiment status", "",stamp,"", "## GPU", "```text",gpu,"```","", "## Queues"]
    for name,row in queues.items():lines.append("- %s: %s"%(name,row.get("status","not started")))
    lines += ["","## Jobs"]
    for job in jobs:
        progress=job["progress"];count=progress.get("iteration",progress.get("images","?"))
        lines.append("- %s: %s; %s/%s; log=%s"%(job["name"],job["status"],count,progress.get("total","?"),job.get("log")))
        if job.get("reason"):lines.append("  Reason: "+str(job["reason"]))
    lines += ["","## Completed result files",json.dumps(completed,indent=2),"",result["note"]]
    folder=ROOT/"reports/overnight_20260930"
    atomic_json(folder/"latest.json",result);atomic_text(folder/"latest.md","\n".join(lines)+"\n")
    if final:
        atomic_json(folder/"morning_20261001_1000.json",result)
        atomic_text(folder/"morning_20261001_1000.md","\n".join(lines)+"\n")


def run_stage(stage,gpu_index):
    gpu=stage in ("smoke","export")
    completion=ROOT/"results/R13_gqa_disjoint"/("summary.json" if stage=="evaluate" else stage+"/summary.json")
    state=ROOT/"status"/("R13_"+stage+".json")
    record=dict(status="waiting_gpu" if gpu else "starting",gpu=[gpu_index] if gpu else [],completion=str(completion),
                progress_file=str(completion.parent/"progress.json"))
    previous=read(completion)
    if previous.get("status")=="complete":
        if stage=="smoke" and not previous.get("historical_cache_match"):return False
        atomic_json(state,dict(record,status="complete",resumed=True));return True
    atomic_json(state,record);lock=None
    try:
        if gpu:
            lock=(ROOT/"status"/("gpu%d.resource.lock"%gpu_index)).open("a");fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(["nvidia-smi","-i",str(gpu_index),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip():time.sleep(20)
        command=[os.environ["SGG_NATIVE_PYTHON"],"-u",str(ROOT/"code/gqa_disjoint.py"),stage]
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu_index) if gpu else ""
        log=ROOT/"logs"/("R13_"+stage+"_"+time.strftime("%Y%m%d_%H%M%S")+".log")
        with log.open("w") as stream:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            record.update(status="running",pid=process.pid,command=command,log=str(log));atomic_json(state,record)
            rc=process.wait()
        ok=rc==0 and read(completion).get("status")=="complete"
        atomic_json(state,dict(record,status="complete" if ok else "failed",returncode=rc))
        return ok
    except Exception as exc:
        atomic_json(state,dict(record,status="failed",error=str(exc)));return False
    finally:
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--report_until",required=True);args=p.parse_args()
    ensure_storage();deadline=datetime.datetime.fromisoformat(args.report_until).timestamp()
    lock=(ROOT/"status/overnight_queue.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from gqa_disjoint import verify
    verify()
    reference_path=ROOT/"results/R1/tde_motifs/sgcls/summary.json"
    reference=read(reference_path);checks=reference.get("reproduction_checks",{})
    if not reference.get("full_split") or not checks or not all(v["passes_existing_tolerance"] for v in checks.values()):
        raise RuntimeError("TDE SGCls reference gate required before external transfer")
    source_files=[ROOT/"code/overnight_queue.py",ROOT/"code/gqa_disjoint.py",ROOT/"manifests/R13_gqa_disjoint.json",reference_path]
    registration=dict(report_until=args.report_until,sources={str(p):sha256(p) for p in source_files},
                      rule="Use the first terminal R12 GPU lane for independent R13 smoke/export/evaluate; never stop training at 10:00")
    manifest=ROOT/"manifests/overnight_20260930.json"
    if manifest.exists() and read(manifest)!=registration:raise RuntimeError("Overnight schedule already registered differently")
    atomic_json(manifest,registration)
    def reporter():
        while True:
            final=time.time()>=deadline
            snapshot(final=final)
            if final:return
            time.sleep(min(60,max(1,deadline-time.time())))
    reporter_thread=threading.Thread(target=reporter,daemon=True);reporter_thread.start()
    state=ROOT/"status/R13_queue.json"
    atomic_json(state,dict(status="waiting_dependency",pid=os.getpid(),dependency="first terminal R12 GPU lane",
                          plan="First released GPU: smoke -> full training-disjoint GQA export -> CPU historical-state comparison"))
    gpu_index=None
    while gpu_index is None:
        gpu_index=dependency_gpu(read(ROOT/"status/R12_queue.json"))
        if gpu_index is None:time.sleep(30)
    outcomes={}
    for stage in ("smoke","export","evaluate"):
        atomic_json(state,dict(status="running",pid=os.getpid(),gpu=gpu_index,current=stage,outcomes=outcomes))
        ok=run_stage(stage,gpu_index);outcomes[stage]="complete" if ok else "failed"
        if not ok:break
    atomic_json(state,dict(status="complete" if outcomes.get("evaluate")=="complete" else "needs_attention",outcomes=outcomes))
    snapshot()
    # Keep only the light status reporter alive until the requested morning window.
    reporter_thread.join()


if __name__=="__main__":main()
