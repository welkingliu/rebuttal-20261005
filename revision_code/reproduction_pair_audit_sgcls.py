"""SGCls audit: compare pair-keyed probabilities instead of floating-point tie order."""
from pathlib import Path
import torch

import reproduction_pair_audit as audit
from common import ROOT, sha256, atomic_json

SOURCE = Path(__file__).resolve()
audit.OUT = ROOT / "results/R14_pair_cap_sgcls_keyed"
audit.SPEC = dict(audit.SPEC, version="r14_sgcls_pair_keyed_noop_v1", tasks=["sgcls"],
                  noop="same object outputs, same pair set, pair-keyed scores within 1e-5; control R@K must match saved R9 on every validation image")
original_fingerprints = audit.fingerprints
audit.fingerprints = lambda: dict(original_fingerprints(), **{str(SOURCE): sha256(SOURCE)})
checks = []


def keyed_noop(a, b):
    result = audit.compare_shared(a, b)
    pa, pb = a.get_field("rel_pair_idxs"), b.get_field("rel_pair_idxs")
    if len(pa) != len(pb):
        raise RuntimeError("No-op changed candidate count")
    if result["shared_probability_max_error"] > 1e-5:
        raise RuntimeError("No-op changed pair-keyed scores beyond 1e-5")
    result["rank_positions_changed"] = int((pa != pb).any(1).sum())
    lookup = {tuple(p): j for j, p in enumerate(pb.cpu().tolist())}
    index = torch.tensor([lookup[tuple(p)] for p in pa.cpu().tolist()], device=pa.device)
    left = a.get_field("pred_rel_scores")[:, 1:].argmax(1)
    right = b.get_field("pred_rel_scores")[index, 1:].argmax(1)
    if not torch.equal(left, right):
        raise RuntimeError("No-op changed the predicate argmax for a candidate")
    checks.append(result)
    atomic_json(audit.OUT / "sgcls/keyed_noop.json", dict(checks=checks))
    return result["shared_probability_max_error"]


audit.compare_predictions = keyed_noop

if __name__ == "__main__":
    audit.main()
