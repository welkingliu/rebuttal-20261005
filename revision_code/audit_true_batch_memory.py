"""One real batch-8 training backward, no optimizer update or checkpoint search."""
import json
import os
from pathlib import Path
import time
import fcntl

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage
from repro_experiment import build, dataset


def main():
    ensure_storage()
    torch.set_num_threads(4)
    torch.manual_seed(666)
    np.random.seed(666)
    out = ROOT / "results/R14_true_batch_memory"
    state = ROOT / "status/R14_true_batch_memory.json"
    record = dict(status="waiting_gpu", pid=os.getpid(), gpu=[0],
                  command=[str(Path(__file__).resolve())], completion=str(out / "summary.json"),
                  log=str(ROOT / "logs/R14_true_batch_memory.log"))
    atomic_json(state, record)
    lock = (ROOT / "status/gpu0.resource.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    atomic_json(state, dict(record, status="running"))
    start = time.monotonic()
    try:
        model, cfg, _, provenance = build("sgdet", out, fresh=True)
        for module in [model.backbone, model.rpn, model.roi_heads.box]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        from pysgg.data.transforms import build_transforms
        from pysgg.data.collate_batch import BatchCollator
        ds = dataset(cfg, "train")
        ds.filter_duplicate_rels = True
        ds.transforms = build_transforms(cfg, is_train=True)
        # Reference loader groups aspect ratios. Use the first portrait group.
        indices = [i for i, info in enumerate(ds.img_info) if info["height"] >= info["width"]][:8]
        if len(indices) != 8:
            raise RuntimeError("Incomplete memory probe batch")
        images, targets, _ = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)([ds[i] for i in indices])
        model.train()
        torch.cuda.reset_peak_memory_stats()
        losses = model(images.to("cuda"), [target.to("cuda") for target in targets], logger=None)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite true-batch loss")
        loss.backward()
        grad = sum(float(p.grad.square().sum()) for n, p in model.named_parameters()
                   if "context_layer.out_obj" in n and p.grad is not None) ** .5
        result = dict(status="complete", batch8_backward_passed=grad > 0, per_gpu_batch=8,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                      padded_image_shape=list(images.tensors.shape), loss_terms={k: float(v.detach()) for k, v in losses.items()},
                      object_head_gradient_norm=grad, training_indices=indices,
                      note="One grouped batch only; not a guarantee that every training batch fits. No optimizer update was made.")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result = dict(status="complete", batch8_backward_passed=False, reason="cuda_out_of_memory", error=str(exc),
                          note="No training run was launched; a memory-safe exact or declared alternative is needed")
        else:
            atomic_json(state, dict(record, status="failed", reason=str(exc)))
            raise
    result["seconds"] = time.monotonic() - start
    atomic_json(out / "summary.json", result)
    atomic_json(state, dict(record, status="complete"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
