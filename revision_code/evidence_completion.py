"""Shared, auditable summaries for the bounded October 3 evidence completion."""
import hashlib
import json
from pathlib import Path

import numpy as np

from common import atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def lock(path, value):
    if Path(path).exists() and read(path) != value:
        raise RuntimeError("Registered evidence protocol changed: " + str(path))
    atomic_json(path, value)
    return sha256(path)


def sources(names):
    folder = Path(__file__).resolve().parent
    return {name: sha256(folder / name) for name in names}


def selected_ids(ids, count, salt):
    return sorted(ids, key=lambda x: hashlib.sha256((salt + str(x)).encode()).digest())[:count]


def paired_ratio(correct, baseline, denominators, draws=2000, seed=17029):
    a, b, d = (np.asarray(x, dtype=float) for x in (correct, baseline, denominators))
    if a.shape != b.shape or a.shape != d.shape or a.ndim != 1 or not len(a) or d.sum() <= 0:
        raise ValueError("Invalid paired evidence denominator")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        ix = rng.integers(0, len(a), len(a))
        if d[ix].sum() > 0:
            boot.append(float((a[ix] - b[ix]).sum() / d[ix].sum()))
    return dict(value=float(a.sum() / d.sum()), baseline=float(b.sum() / d.sum()),
                delta=float((a-b).sum() / d.sum()), paired_image_95ci=np.quantile(boot, [.025, .975]).tolist(),
                images=len(a), denominator=float(d.sum()), bootstrap_draws=draws,
                interpretation="descriptive paired interval; no multiplicity-adjusted discovery claim")


def transitions(native_pre, repaired_pre, native_post, repaired_post, target):
    arrays = [np.asarray(x) for x in (native_pre, repaired_pre, native_post, repaired_post, target)]
    if any(x.shape != arrays[-1].shape for x in arrays):
        raise ValueError("Proposal order/cardinality mismatch")
    valid = arrays[-1] > 0
    truth = arrays[-1][valid]
    flags = [x[valid] == truth for x in arrays[:4]]
    code = sum(flag.astype(int) << j for j, flag in enumerate(flags))
    table = np.bincount(code, minlength=16)
    return dict(positive_proposals=int(valid.sum()), correctness_pattern_counts=table.tolist(),
                bit_order=["native_pre", "repaired_pre", "native_post", "repaired_post"],
                pre_corrections=int((~flags[0] & flags[1]).sum()),
                pre_regressions=int((flags[0] & ~flags[1]).sum()),
                post_corrections=int((~flags[2] & flags[3]).sum()),
                post_regressions=int((flags[2] & ~flags[3]).sum()),
                pre_corrections_correct_after_repair_nms=int((~flags[0] & flags[1] & flags[3]).sum()))
