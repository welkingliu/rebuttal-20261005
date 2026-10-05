"""Recover R16 without changing its sealed training code or optimizer recipe."""
import argparse
import fcntl
import json
import os
import random
import sys

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, sha256
import r16_motifs as r16


def training_complete(state, maximum=75000):
    return state["iteration"] >= maximum or state["scheduler"]["stage_count"] >= 3


def pending_is_newer(latest, pending):
    return bool(pending and pending.get("pending_validation") and
                (latest is None or pending["iteration"] > latest["iteration"]))


def verify_state(state, protocol):
    if state["protocol"] != protocol:
        raise RuntimeError("Recovery checkpoint does not match the sealed protocol")
    if protocol["driver_sha256"] != sha256(r16.__file__):
        raise RuntimeError("Sealed R16 driver changed; refusing automatic resume")
    for name, expected in protocol["sources"].items():
        if sha256(r16.REPO / name) != expected:
            raise RuntimeError("Native source changed: " + name)
    if sha256(ROOT / "data/R16_plain_motifs/upstream_statistics.pth") != protocol["statistics_sha256"]:
        raise RuntimeError("Frequency prior changed")


def restore_rng(state):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def finish_pending_validation(task, out, state, protocol):
    from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler
    import logging
    import math

    model, cfg = r16.build(task, out, weight=out / "pre_validation.pth")
    for module in (model.backbone, model.rpn, model.roi_heads.box):
        for p in module.parameters(): p.requires_grad_(False)
    optimizer = make_optimizer(cfg, model, logging.getLogger("r16_recovery"), slow_heads=[], rl_factor=8.)
    scheduler = make_lr_scheduler(cfg, optimizer, logging.getLogger("r16_recovery"))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    val = r16.dataset(cfg, "val")
    assert len(val) == 5000
    restore_rng(state)
    step = state["iteration"]
    result = r16.evaluate(model, cfg, val, task, out / ("validation_%06d" % step))
    value = result["R"]["100"]
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite validation metric: scheduler was not advanced")
    best = state["best"]
    if value > best:
        best = value
        r16.save_torch(out / "best.pth", dict(model=model.state_dict(), iteration=step, protocol=protocol))
    # The interrupted driver saved immediately after optimizer.step and before this call.
    scheduler.step(value, epoch=step)
    r16.save_torch(out / "latest.pth", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), iteration=step, best=best, protocol=protocol,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    atomic_json(out / "recovery_validation.json", dict(status="complete", iteration=step,
        optimizer_updates_replayed=0, scheduler_updates=1, validation_R100=value,
        recovery_driver_sha256=sha256(__file__)))


def finish_test(task, out, protocol):
    best = out / "best.pth"
    payload = torch.load(str(best), map_location="cpu")
    verify_state(payload, protocol)
    selected = payload["iteration"]
    del payload
    model, cfg = r16.build(task, out, weight=best)
    result = r16.evaluate(model, cfg, r16.dataset(cfg, "test"), task, out / "test")
    atomic_json(out / "summary.json", dict(status="complete", selected_iteration=selected, test=result,
        interpretation="Independent plain-Motifs reimplementation; TDE results cannot substitute",
        recovery=dict(test_only=True, extra_training_steps=0, checkpoint_sha256=sha256(best),
                      driver_sha256=sha256(__file__))))


def recover(task):
    out = r16.RUN / task / "formal"
    summary = out / "summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("status") == "complete":
        print("[ALREADY COMPLETE] " + task, flush=True)
        return
    protocol = json.loads((out / "protocol.json").read_text())
    latest = torch.load(str(out / "latest.pth"), map_location="cpu") if (out / "latest.pth").exists() else None
    pending = torch.load(str(out / "pre_validation.pth"), map_location="cpu") if (out / "pre_validation.pth").exists() else None
    for state in [latest, pending]:
        if state is not None: verify_state(state, protocol)
    action = "pending_validation" if pending_is_newer(latest, pending) else "latest_checkpoint"
    atomic_json(out / "recovery_entry.json", dict(status="running", action=action,
        latest_iteration=latest["iteration"] if latest else None,
        pending_iteration=pending["iteration"] if pending else None,
        sealed_driver_sha256=protocol["driver_sha256"], recovery_driver_sha256=sha256(__file__)))
    if action == "pending_validation":
        finish_pending_validation(task, out, pending, protocol)
        latest = torch.load(str(out / "latest.pth"), map_location="cpu")
        torch.cuda.empty_cache()
    del pending
    if latest is None:
        raise RuntimeError("No saved state to recover; this command never starts over")
    done = training_complete(latest)
    del latest
    if done:
        finish_test(task, out, protocol)
    else:
        # Execute the unchanged driver; it restores sampler, optimizer and all RNG states.
        sys.argv = [r16.__file__, "--task", task]
        r16.main()
    atomic_json(out / "recovery_complete.json", dict(status="complete", task=task,
        driver_sha256=sha256(__file__), original_protocol_preserved=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["predcls", "sgcls", "sgdet"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    ensure_storage()
    lock = (ROOT / "status" / ("gpu%d.resource.lock" % args.gpu)).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("[BUSY] Existing queue owns this GPU; no training state was changed")
    # Do not allow recovery of a task running on a different GPU, either.
    for path in (ROOT / "status").glob("R16_gpu*.json"):
        state = json.loads(path.read_text())
        pid = state.get("child_pid")
        if state.get("task") == args.task and pid and os.path.exists("/proc/%d" % pid):
            raise SystemExit("[BUSY] This task is already running; no state was changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(4)
    recover(args.task)


if __name__ == "__main__": main()
