"""R20 amendment: retain native all-pair execution before selecting GT metric rows."""
from pathlib import Path

import numpy as np
import torch

from common import ROOT, sha256
import r20_motifs_bridge as original

SOURCE = Path(__file__).resolve()


def capture(model, cfg, transform, ds, index, image=None):
    from maskrcnn_benchmark.structures.image_list import to_image_list
    target = ds.get_groundtruth(index, evaluation=True)
    gt = target.get_field("relation_tuple").long()
    if image is None: image = ds[index][0]
    tensor, resized = transform(image, target)
    holder = {}
    def before(module, args): holder["inputs"] = args
    def after(module, args, output): holder["output"] = output
    predictor = model.roi_heads.relation.predictor
    hooks = [predictor.register_forward_pre_hook(before), predictor.register_forward_hook(after)]
    try:
        with torch.no_grad():
            model(to_image_list([tensor.cuda()], size_divisible=32), [resized.to("cuda")])
    finally:
        for hook in hooks: hook.remove()
    pairs = holder["inputs"][1][0].detach().cpu()
    n = len(target)
    expected = {(s, o) for s in range(n) for o in range(n) if s != o}
    mapping = {tuple(p): i for i, p in enumerate(pairs.tolist())}
    if len(mapping) != len(pairs) or set(mapping) != expected:
        raise RuntimeError("Native SGCls all-pair set/order contract failed")
    if len(holder["inputs"][0][0]) != n: raise RuntimeError("GT node order/cardinality changed")
    rows = torch.tensor([mapping[tuple(p)] for p in gt[:, :2].tolist()], device="cuda")
    return holder, target, gt, rows


def run():
    original.OUT = ROOT / "results/R20_plain_motifs_native_bridge"
    original.capture = capture
    old_lock = original.lock
    def lock(path, protocol):
        protocol["version"] = "r20_native_all_pairs_v2"
        protocol["sources"][SOURCE.name] = sha256(SOURCE)
        protocol["inference"] = "Unmodified native all-directed-pairs execution; metrics select annotated GT rows after inference"
        protocol["support"] = "GT boxes; native object ROI features, geometry and all candidate pairs fixed for identity intervention"
        protocol["amendment"] = dict(previous="results/R20_plain_motifs_bridge", outcome="stopped at native cache audit",
            reason="Restricted GT-pair execution changed union-feature convolution batch shapes; preserve exact native execution rather than relax tolerance",
            original_tolerance=1e-4, new_tolerance=1e-4, old_partial_results_not_pooled=True)
        return old_lock(path, protocol)
    original.lock = lock
    original.main()


if __name__ == "__main__": run()
