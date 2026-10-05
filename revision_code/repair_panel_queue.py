"""Detached three-strategy, two-GPU queue with bounded independent branches."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from common import ROOT, atomic_json, ensure_storage, output_path
from evidence_completion import read


def main():
    ensure_storage(); here = Path(__file__).resolve().parent
    guard = output_path(ROOT/"status/E5_panel_queue.lock").open("w")
    fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
    from vq_experiment import registration as q_registration
    from repair_panel_experiment import registration as p_registration, BRANCHES
    locations = dict(VQ="VQ_bounded_residual", VR=BRANCHES["VR"][0], VS=BRANCHES["VS"][0])
    for branch in locations:
        if branch == "VQ": _, reg, _ = q_registration(True)
        else: _, reg, _, _ = p_registration(branch, True)
        smoke = read(ROOT/"results"/locations[branch]/"smoke/fold0/summary.json")
        if (smoke.get("status") != "complete" or smoke.get("images") != 3
                or smoke.get("protocol_sha256") != reg): raise RuntimeError("Current smoke missing: "+branch)
        if branch == "VQ" and not smoke.get("smoke_gradients_exercised"): raise RuntimeError("Residual gradients not checked")
    plan = []

    def task(branch, stage, fold=None):
        script = "vq_experiment.py" if branch == "VQ" else "repair_panel_experiment.py"
        args = "--stage "+stage
        if branch != "VQ": args += " --branch "+branch
        if fold is not None: args += " --fold %d" % fold
        name = branch+"_"+("fold%d" % fold if fold is not None else stage)
        directory = ROOT/"results"/locations[branch]/"train3000"
        if stage != "summarize": directory /= "fold%d" % fold if fold is not None else stage
        return dict(name=name, branch=branch, stage=stage, fold=fold, script=script, arguments=args,
            completion=str(directory/"summary.json"), progress_file=str(directory/"progress.json"),
            log=str(ROOT/"logs"/(name+".log")), gpu=None)

    preparation = task("VQ", "prepare")
    for fold in range(5):
        for branch in locations: plan.append(task(branch, "fold", fold))
    summaries = [task(branch, "summarize") for branch in locations]
    all_tasks = [preparation]+plan+summaries
    mutex = threading.Lock(); outcomes = {}; active = {}; errors = []; failed = set()
    deadline = time.monotonic()+8*3600

    def update(row, status, **extra):
        with mutex:
            outcomes[row["name"]] = status
            if status in ["running", "waiting_gpu"]: active[row["name"]] = row.get("gpu")
            else: active.pop(row["name"], None)
            atomic_json(ROOT/"status"/(row["name"]+".json"), dict(row, status=status, **extra))
            atomic_json(ROOT/"status/E5_panel_queue.json", dict(status="running", pid=os.getpid(),
                current={(str(g) if g is not None else "CPU"): n for n,g in active.items()},
                outcomes=dict(outcomes), errors=list(errors),
                plan="VQ bounded visual residual / VR kernel residual / VS visual-constrained relation prior; TRAIN only"))

    def execute(row, gpu=None):
        row = dict(row, gpu=gpu)
        with mutex: blocked = row["branch"] in failed
        if blocked or time.monotonic() >= deadline:
            update(row, "blocked", detail="Branch prerequisite failed" if blocked else "Eight-hour dispatch budget reached")
            return False
        try:
            if gpu is not None:
                wait_until = min(deadline, time.monotonic()+3*3600)
                while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
                    update(row, "waiting_gpu", detail="Existing compute job; will not evict it")
                    if time.monotonic() > wait_until: raise TimeoutError("GPU wait/budget expired")
                    time.sleep(20)
            command = ["bash", "-lc", "source '%s/storage_runtime.sh' && exec env CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \"$SGG_NATIVE_PYTHON\" '%s/%s' %s" % (here, gpu if gpu is not None else "", here, row["script"], row["arguments"])]
            row["command"] = command; print("[start] %s GPU=%s" % (row["name"], gpu), flush=True)
            with output_path(row["log"]).open("a") as log:
                p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(row, "running", pid=p.pid, started=time.time())
                try: code = p.wait(timeout=2*3600)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    code = 124
            if code or not Path(row["completion"]).exists() or read(row["completion"]).get("status") != "complete":
                raise RuntimeError("Stage failed, exit="+str(code))
            update(row, "complete", finished=time.time()); print("[complete] "+row["name"], flush=True)
            return True
        except Exception as error:
            with mutex:
                failed.add(row["branch"]); errors.append(dict(task=row["name"], error=str(error)))
            update(row, "failed", detail=str(error)); print("[failed] %s: %s" % (row["name"], error), flush=True)
            return False

    for row in all_tasks: update(row, "queued")
    if execute(preparation):
        pending = deque(plan)
        def worker(gpu):
            while True:
                with mutex:
                    if not pending: return
                    row = pending.popleft()
                execute(row, gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            workers = [pool.submit(worker, gpu) for gpu in [0, 1]]
            for future in workers: future.result()
        for row in summaries:
            missing = [r for r in plan if r["branch"] == row["branch"] and outcomes.get(r["name"]) != "complete"]
            if missing: update(row, "blocked", detail="Not all five folds completed")
            else: execute(row)
    else:
        for row in plan+summaries: update(row, "blocked", detail="Shared preparation failed")
    report = dict(status="complete" if all(v == "complete" for v in outcomes.values()) else "incomplete",
        branches={}, outcomes=outcomes, errors=errors, validation_accessed=False, test_accessed=False,
        formal_gate_accepted=False, multiple_exploratory_hypotheses=True,
        note="All primaries and controls reported; per-strategy intervals not multiplicity-adjusted; no automatic promotion or gate.")
    for branch in locations:
        path = ROOT/"results"/locations[branch]/"train3000/summary.json"
        if not path.exists():
            report["branches"][branch] = dict(status="incomplete", expected_summary=str(path)); continue
        result = read(path)
        report["branches"][branch] = dict(status="complete", summary=str(path), primary=result["primary"],
            primary_training_screen_satisfied=result["primary_training_screen_satisfied"],
            arms=result["arms"], comparisons=result["comparisons"])
    atomic_json(ROOT/"results/EXPERIMENT_V_RESEARCH_PANEL/summary.json", report)
    atomic_json(ROOT/"status/E5_panel_queue.json", dict(status="complete" if report["status"] == "complete" else "failed",
        pid=os.getpid(), outcomes=outcomes, errors=errors, report=str(ROOT/"results/EXPERIMENT_V_RESEARCH_PANEL/summary.json"),
        note="All scheduled branches stopped; formal validation/test untouched; server remains on"))
    print("[panel-finished] "+report["status"], flush=True)
    if report["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
