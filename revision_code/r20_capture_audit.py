"""Compare restricted-pair and full-native execution on the first blocked image."""
from pathlib import Path

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, sha256
from evidence_completion import read
from r16_motifs import build, dataset, RUN
from r20_motifs_bridge import capture as restricted
from r20_motifs_native_bridge import capture as native


def main():
    ensure_storage(); torch.set_num_threads(2); torch.manual_seed(666)
    out = ROOT/"results/R20_capture_audit"
    previous = ROOT/"results/R20_plain_motifs_bridge/identity"
    protocol = read(previous/"protocol.json")
    pending = [i for i in protocol["image_ids"] if not (previous/"images"/(i+".json")).exists()]
    iid = pending[0]
    model, cfg = build("sgcls", out, RUN/"sgcls/formal/best.pth"); model.eval()
    from maskrcnn_benchmark.data.transforms import build_transforms
    from reproduction_pair_audit import ChunkUnion
    chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
    transform = build_transforms(cfg, is_train=False); ds = dataset(cfg,"test")
    mapping = {Path(p).stem: i for i,p in enumerate(ds.filenames)}
    values = {}; comparisons = {}
    with np.load(RUN/"sgcls/formal/test/predictions"/(iid+".npz")) as z:
        stored_pairs = {tuple(p):j for j,p in enumerate(z["pred_rel_pairs"].tolist())}
        reference = z["pred_rel_scores"].copy()
    for name, method in [("restricted_gt_pairs", restricted), ("native_all_pairs", native)]:
        holder, target, rel, ix = method(model,cfg,transform,ds,mapping[iid])
        probabilities = holder["output"][1][0][ix].softmax(-1).cpu().numpy()
        rows = [stored_pairs[tuple(p)] for p in rel[:,:2].tolist()]
        comparisons[name] = dict(max_abs_error=float(np.abs(probabilities-reference[rows]).max()),
            predicate_top1_agreement=float((probabilities[:,1:].argmax(1)==reference[rows,1:].argmax(1)).mean()),
            native_candidate_pairs=len(holder["inputs"][1][0]),gt_relation_rows=len(rel))
        values[name]=probabilities
    chunk.close()
    atomic_json(out/"summary.json",dict(status="complete",image_id=iid,comparisons=comparisons,
        exact_native_reproduced=comparisons["native_all_pairs"]["max_abs_error"]==0.,
        same_checkpoint=True, checkpoint_sha256=sha256(RUN/"sgcls/formal/best.pth"),
        threshold_not_relaxed=True))
    print(comparisons,flush=True)


if __name__=="__main__": main()
