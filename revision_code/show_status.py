"""Unfinished-work dashboard; --all includes historical results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT


FINISHED = {"complete", "completed", "finished", "done", "passed", "stopped",
            "rejected", "cancelled", "canceled", "skipped", "not_applicable"}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid,0)
        stat = Path("/proc/%s/stat" % pid)
        return not stat.exists() or stat.read_text().rsplit(")",1)[1].split()[0] != "Z"
    except (OSError, ValueError):
        return False


def duration(seconds):
    if seconds is None:
        return "estimating"
    minutes = int(max(seconds,0)) // 60
    return "%dh %02dm" % divmod(minutes,60) if minutes >= 60 else "%dm %02ds" % (minutes,int(seconds)%60)


def verdict(result):
    if isinstance(result.get("test"),dict) and "reproduction_passed" in result["test"]:
        return verdict(result["test"])
    if result.get("development_eligible") is False:
        return "STOPPED: development screening NOT met (gate/test not evaluated)"
    if result.get("accepted") is False:
        return "STOPPED: improvement gate NOT met"
    if result.get("accepted") is True:
        return "PASSED: validation gate (not a test claim)"
    if result.get("reproduction_passed") is False:
        return "FINISHED: reference gap remains"
    if result.get("reproduction_passed") is True:
        return "PASSED: registered reference tolerance"
    if result.get("training_completed"):
        return "TRAINING FINISHED: evaluation separate"
    return "FINISHED: results available"


def r16_progress_path(progress_path):
    """Validation/test write their own progress while training waits."""
    path = Path(progress_path)
    candidates = [p for p in [path, path.parent / "test/progress.json"] if p.exists()]
    candidates.extend(path.parent.glob("validation_*/progress.json"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else path


def snapshot():
    active = []
    for path in sorted((ROOT/"status").glob("*.json")):
        item = read(path)
        if item.get("status") not in ("running","waiting_gpu","waiting_dependency","pending",
                                      "queued","needs_attention","failed","blocked"):
            continue
        if not ("command" in item or "completion" in item):
            continue
        if item["status"] in ("failed","blocked") and not path.stem.startswith(("R9_","R10_","R12_","R13_","R14_","R15_","R16_","R17_","R18_","R19_","R20_","R21_","R22_","VE_","VF_","VG_","VH_","VI_","VK_","VN_","VO_","VP_","VQ_","VR_","VS_","VT_","VU_","VV_","E5_","probe_convergence")):
            continue
        item["name"] = path.stem
        if item["status"] == "running" and not alive(item.get("pid")):
            item["status"] = "STALE: process no longer running"
        progress_path = item.get("progress_file")
        if progress_path:
            if path.stem.startswith("R16_"):
                progress_path = r16_progress_path(progress_path)
            item["progress"] = read(progress_path)
            if Path(progress_path).exists():
                item["progress_age_seconds"] = time.time()-Path(progress_path).stat().st_mtime
            parent = Path(progress_path).parent.name
            if path.stem.startswith("R16_") and (parent.startswith("validation_") or parent == "test"):
                p = item["progress"]
                p["detail"] = ("Validation after training step " + parent.split("_")[-1]
                               if parent.startswith("validation_") else "Full test of validation-selected checkpoint")
                if p.get("images", 0) > 0 and p.get("seconds") is not None:
                    p["eta_seconds"] = p["seconds"] / p["images"] * max(0, p["total"]-p["images"])
        active.append(item)
    # Bounded V-L jobs write their own progress, without the historical queue.
    for label, folder, log in [
        ("VL_capacity", "VL_sgdet_feasibility/development200", "VL_sgdet_feasibility.log"),
        ("VL_router_fit", "VL_proposal_router/fit", "VL_router_fit.log"),
        ("VL_router_evaluate", "VL_proposal_router/evaluate", "VL_router_evaluate.log"),
        ("VM_localization_fit", "VM_localization_router/fit", "VM_localization_fit.log"),
        ("VM_localization_evaluate", "VM_localization_router/evaluate", "VM_localization_evaluate.log"),
    ]:
        result_dir = ROOT / "results" / folder
        p = read(result_dir / "progress.json")
        if not p or (result_dir / "summary.json").exists() or p.get("status") in FINISHED:
            continue
        active.append(dict(name=label, status=p.get("status", "running") if alive(p.get("pid")) else "STALE: process no longer running",
            pid=p.get("pid"), gpu=1, progress=p,
            progress_age_seconds=time.time()-(result_dir / "progress.json").stat().st_mtime,
            log=str(ROOT / "logs" / log)))
    vn_order = {"VN_" + stage: index for index, stage in enumerate(
        ["export", "features", "outcomes", "fit", "evaluate"])}
    active.sort(key=lambda item: (item["status"] != "running",
                                  vn_order.get(item["name"], 100), item["name"]))
    try:
        gpu = subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"],text=True,timeout=5).strip().splitlines()
    except (OSError,subprocess.SubprocessError):
        gpu = ["GPU telemetry unavailable"]
    paths = {"R17 original V gradient audit":"R17_original_v/summary.json",
             "R18 open-vocabulary audit":"R18_ovsgtr/formal/summary.json",
             "R16 plain Motifs SGCls":"R16_plain_motifs/sgcls/formal/summary.json",
             "R16 plain Motifs SGDet":"R16_plain_motifs/sgdet/formal/summary.json",
             "R16 plain Motifs PredCls":"R16_plain_motifs/predcls/formal/summary.json",
             "R6 SGTR live":"R6_sgtr_live/summary.json", "V-C":"VC/pilot_decision.json",
             "R11 identity decomposition":"R11_identity_channels/summary.json",
             "R13 disjoint GQA":"R13_gqa_disjoint/summary.json",
             "R14 SGDet pair-cap audit":"R14_pair_cap/sgdet/summary.json",
             "R14 SGCls pair-cap audit":"R14_pair_cap_sgcls_native/sgcls/summary.json",
             "R14 official zero-shot":"R14_official_zeroshot/summary.json",
             "R14 true batch memory":"R14_true_batch_memory/summary.json",
             "R14 corrected SGDet test":"R14_reference_sgdet_test/summary.json",
             "R15 Transformer identity":"R15_transformer_identity/transformer/test/summary.json",
             "V-E pilot gate":"VE/pilot_decision.json",
             "V-E failure mechanism audit":"VE_failure_audit_20261001/summary.json",
             "V-F exploratory pilot":"VF/summary.json",
             "V-G retrieval feasibility":"VG_retrieval_development/summary.json",
             "V-H independent visual expert":"VH_independent_identity/summary.json",
             "V-I selective fusion":"VI_selective_fusion/summary.json",
             "V-K native confirmation":"VK_native_confirmation/summary.json",
             "R10 probe capacity":"R10_probe_capacity/summary.json",
             "Probe convergence":"probe_convergence/summary.json",
             "GQA image-exposure audit":"external_overlap_audit/summary.json",
             "V-D":"VD/summary.json", "R9 Transformer SGCls":"R9_reproduction/sgcls/eval_corrected/test/summary.json",
             "R9 Transformer SGDet":"R9_reproduction/sgdet/eval_retrained/test/summary.json"}
    recent = {label:verdict(read(ROOT/"results"/path)) for label,path in paths.items() if (ROOT/"results"/path).exists()}
    queue = read(ROOT/"status/R9_queue.json")
    if queue.get("status") == "running" and not alive(queue.get("pid")):
        queue["status"] = "STALE: queue process missing"
    identity_queue = read(ROOT/"status/R12_queue.json")
    if identity_queue.get("status") in ("running","waiting_dependency") and not alive(identity_queue.get("pid")):
        identity_queue["status"] = "STALE: identity queue process missing"
    followups = {name: read(ROOT / "status" / (name + "_queue.json")) for name in ["R15", "VE", "VF", "VI", "VK", "VN", "VO", "VP", "E5_panel", "VT", "VU", "VV", "R19", "R19A", "R20", "R20N", "R21"]}
    for row in followups.values():
        if row.get("status") == "running" and not alive(row.get("pid")):
            row["status"] = "STALE: queue process missing"
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),gpu=gpu,active=active,queue=queue,
                identity_queue=identity_queue,external_queue=read(ROOT/"status/R13_queue.json"),recent=recent,
                followups=followups)


def unfinished_snapshot(data):
    """Filter the view only; preserve persisted statuses and scientific outcomes."""
    data = dict(data)
    recent = data.get("recent", {})
    identity_superseded = (
        recent.get("R14 corrected SGDet test") == "PASSED: registered reference tolerance"
        and "R15 Transformer identity" in recent
        and data.get("followups", {}).get("R15", {}).get("status") == "complete"
    )

    def unfinished(row):
        return bool(row) and row.get("status", "").lower() not in FINISHED

    data["active"] = [row for row in data.get("active", []) if unfinished(row)
                      and not (identity_superseded
                               and row.get("name") == "R12_transformer_eligibility")]
    data["followups"] = {name: row for name, row in data.get("followups", {}).items()
                         if unfinished(row)}
    for key in ("queue", "identity_queue", "external_queue"):
        row = data.get(key, {})
        data[key] = row if unfinished(row) else {}
    if identity_superseded:
        data["identity_queue"] = {}
    data["recent"] = {}
    return data


def show(data, include_history=False):
    print("SGG REBUTTAL | " + data["time"])
    print("="*78)
    for line in data["gpu"]:
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 4:
            print(line); continue
        index,util,used,total = fields
        tasks = []
        for item in data["active"]:
            assigned = item.get("gpu",[])
            assigned = assigned if isinstance(assigned,list) else [assigned]
            if int(index) in assigned and item["status"] == "running":
                tasks.append(item["name"])
        print("GPU %s | util %3s%% | %s/%s MiB | %s" % (index,util,used,total,", ".join(tasks) or "no tracked running job"))
    print("\nCURRENT" if include_history else "\nUNFINISHED TASKS")
    if not data["active"]:
        print("  No active tracked jobs; pending queues, if any, are listed below.")
    for item in data["active"]:
        print("  %s | %s | PID %s" % (item["name"],item["status"],item.get("pid","-")))
        progress = item.get("progress",{})
        if progress.get("stage"):
            print("    stage: " + str(progress["stage"]))
        count = progress.get("iteration",progress.get("images"))
        total = progress.get("total")
        if count is not None and total:
            timing = ("remaining %s | " % duration(progress.get("eta_seconds"))
                      if item["status"] == "running" else "")
            print("    %s/%s (%.1f%%) | %supdated %s ago" %
                  (count,total,100*count/total,timing,duration(item.get("progress_age_seconds"))))
            if item.get("progress_age_seconds",0)>600 and item["status"]=="running":
                print("    CHECK: progress unchanged >10 min; inspect log before diagnosing a stall.")
        elif item["status"]=="running":
            print("    Loading/checking inputs; no measured ETA yet.")
        if progress.get("detail"):
            print("    " + str(progress["detail"]))
        if item.get("log"):
            print("    log: " + item["log"])
        if item.get("stage_log"):
            print("    stage log: " + item["stage_log"])
        if item.get("reason"):
            print("    " + item["reason"])
        if not include_history:
            if item.get("plan"):
                print("    " + item["plan"])
            for gpu, reason in item.get("waiting_on", {}).items():
                print("    Waiting on GPU %s: %s" % (gpu, reason))
        if item["name"] == "R12_transformer_eligibility" and data.get("followups", {}).get("R15"):
            print("    Historical capped protocol. R14 corrected baseline passed; R15 is the new intervention run.")
    for name, row in data.get("followups", {}).items():
        if not row:
            continue
        if not include_history and any(item.get("name") == name + "_queue" for item in data["active"]):
            continue
        print("\nFOLLOW-UP %s | %s" % (name, row.get("status", "unknown")))
        if row.get("plan"):
            print("  " + row["plan"])
        if row.get("deadline_local"):
            print("  Budget boundary: %s (existing V-E jobs are not interrupted)" % row["deadline_local"])
        current = row.get("current")
        if isinstance(current, dict):
            for gpu, stage in current.items():
                print("  GPU %s: %s" % (gpu, stage))
        elif current:
            print("  Stage: " + str(current))
        counts = row.get("outcomes", {})
        if include_history:
            print("  Completed stages: %d | Failed stages: %d" %
                  (sum(x == "complete" for x in counts.values()), sum(x == "failed" for x in counts.values())))
        else:
            for stage, outcome in counts.items():
                if outcome.lower() not in FINISHED:
                    print("  %s: %s" % (stage, outcome))
        if row.get("reason"):
            print("  " + row["reason"])
        for gpu, reason in row.get("waiting_on", {}).items():
            print("  Waiting on GPU %s: %s" % (gpu, reason))
    queue = data["queue"]
    if queue:
        print("\nR9 TRAINING/REFERENCE QUEUE | " + queue.get("status","not started"))
        if queue.get("current"):
            print("  Stage %s/%s: %s" % (queue.get("position"),queue.get("total"),queue["current"]))
        print("  Pending: " + (", ".join(queue.get("next",[])) or "none"))
        if queue.get("schedule") and queue.get("status")=="running":
            print("  Plan: both GPUs train -> GPU 0 SGDet val/test + GPU 1 SGCls val/test")
    print("  ETA above is for the current task only, not the entire queue.")
    identity = data.get("identity_queue",{})
    if identity:
        print("\nNEXT QUEUE: SGDet IDENTITY | " + identity.get("status","unknown"))
        if identity.get("status")=="waiting_dependency":
            print("  Waiting for R9 training AND evaluation to finish; no GPU reserved.")
            print("  GPU 0: TDE-Motifs; GPU 1: corrected Transformer (reference gate required).")
            print("  Each lane: 3-image smoke -> 1000 validation -> 100 development -> 2000 test.")
        for gpu, task in identity.get("current",{}).items():
            print("  GPU %s: %s"%(gpu,task))
        if identity.get("outcomes"):
            for task,outcome in identity["outcomes"].items():
                if include_history or outcome.lower() not in FINISHED:
                    print("  %s: %s"%(task,outcome))
        if identity.get("reason"):
            print("  " + identity["reason"])
    external=data.get("external_queue",{})
    if external:
        print("\nNEXT QUEUE: DISJOINT GQA | " + external.get("status","unknown"))
        if external.get("status")=="waiting_dependency":
            print("  First released R12 GPU: historical-cache smoke -> 1674 eligible images.")
        if external.get("current"):
            print("  Stage: " + external["current"])
        for stage,outcome in external.get("outcomes",{}).items():
            if include_history or outcome.lower() not in FINISHED:
                print("  %s: %s"%(stage,outcome))
        print("  Server report: reports/overnight_20260930/latest.md (morning snapshot at 10:00 CST)")
    if data.get("recent"):
        print("\nRECENT RESULTS (execution and scientific outcome are separate)")
        for label,outcome in data["recent"].items():
            print("  %-25s %s" % (label,outcome))
    print("\nCtrl+C closes this viewer only; background experiments continue.",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true")
    parser.add_argument("--interval",type=float,default=5);parser.add_argument("--json",action="store_true")
    parser.add_argument("--all", action="store_true", help="Include finished tasks and historical results")
    args=parser.parse_args()
    try:
        while True:
            data=snapshot()
            if not args.all:
                data=unfinished_snapshot(data)
            if args.watch and sys.stdout.isatty() and not args.json:
                print("\033[2J\033[H",end="")
            if args.json:
                print(json.dumps(data),flush=True)
            else:
                show(data, include_history=args.all)
            if not args.watch:
                break
            time.sleep(max(1,args.interval))
    except KeyboardInterrupt:
        pass


if __name__=="__main__":
    main()
