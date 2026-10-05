"""Train on both GPUs first, then evaluate independent tasks on two lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import shutil
import threading

from common import ROOT, atomic_json, ensure_storage, sha256
from repro_protocol import verify
from repro_queue import jobs, now, run
from audit_reproduction_paths import main as audit_paths

PLAN = dict(training_first="R9_sgdet_train", smoke=["R9_sgcls_smoke","R9_train_smoke"],
            evaluation_lanes=[["R9_sgcls_val","R9_sgcls_test"],["R9_sgdet_val","R9_sgdet_test"]],
            rule="Failed training blocks only SGDet evaluation; evaluation error never discards training or cancels the other lane")


def fingerprint():
    return dict(plan=PLAN, reproduction_manifest_sha256=sha256(ROOT/"manifests/R9_reproduction.json"),
                sources={n:sha256(ROOT/"code"/n) for n in ["repro_pipeline.py","audit_reproduction_paths.py"]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--register",action="store_true")
    args=parser.parse_args();ensure_storage();verify()
    manifest=ROOT/"manifests/R9_schedule.json"
    if args.register:
        if manifest.exists():
            assert json.loads(manifest.read_text())["contract"]==fingerprint(), "Schedule already registered differently"
        else:
            atomic_json(manifest,dict(contract=fingerprint(),registered_at=now()))
        print("[REGISTERED] Dual-GPU training, then independent evaluation lanes",flush=True);return
    if json.loads(manifest.read_text())["contract"]!=fingerprint():
        raise RuntimeError("Schedule changed after registration")
    lock=(ROOT/"status/R9_queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if shutil.disk_usage(ROOT).free<100*1024**3:
        raise RuntimeError("Less than 100 GiB free")
    audit_paths()
    by_name={j["name"]:j for j in jobs()}
    order=PLAN["smoke"]+[PLAN["training_first"]]+sum(PLAN["evaluation_lanes"],[])
    outcomes={};running=set();mutex=threading.Lock()
    def publish():
        atomic_json(ROOT/"status/R9_queue.json",dict(status="running",pid=os.getpid(),
                    position=len(outcomes)+1,total=len(order),current=", ".join(sorted(running)) or "scheduling",
                    next=[n for n in order if n not in outcomes and n not in running],outcomes=outcomes.copy(),updated_at=now(),
                    schedule=str(manifest)))
    def execute(name):
        job=by_name[name]
        with mutex:
            failed=[d for d in job["dependencies"] if outcomes.get(d)!="complete"]
            if failed:
                atomic_json(ROOT/"status"/(name+".json"),dict(job,status="blocked",reason="Unsuccessful dependencies: "+", ".join(failed)))
                outcomes[name]="blocked";publish();return
            running.add(name);publish()
        ok=run(job)
        with mutex:
            outcomes[name]="complete" if ok else "failed";running.remove(name);publish()
    for name in PLAN["smoke"]+[PLAN["training_first"]]:
        execute(name)
    def lane(names):
        for name in names:
            execute(name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,names) for names in PLAN["evaluation_lanes"]]
        for future in futures:
            future.result()
    ok=all(v=="complete" for v in outcomes.values())
    atomic_json(ROOT/"status/R9_queue.json",dict(status="complete" if ok else "needs_attention",outcomes=outcomes,
                finished_at=now(),schedule=str(manifest),note="Execution completion is not reference-metric acceptance"))
    if not ok:
        raise SystemExit(1)


if __name__=="__main__":
    main()
