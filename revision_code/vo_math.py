"""Training-only grouped diagnostics. Selection functions never accept GT."""
import hashlib

import numpy as np


def image_folds(ids, folds=5):
    ids = [str(x) for x in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    order = sorted(ids, key=lambda x: hashlib.sha256(("vo_grouped_v1:"+x).encode()).digest())
    return {iid: i % folds for i, iid in enumerate(order)}


def protection_targets(gain, delta_r, delta_class, support):
    gain, delta_r = np.asarray(gain), np.asarray(delta_r)
    dc, support = np.asarray(delta_class), np.asarray(support, dtype=bool)
    if dc.ndim != 2 or support.shape != (dc.shape[1],) or len(gain) != len(dc):
        raise ValueError("Wrong protection-target shape")
    if not np.isfinite(dc).all() or not np.isfinite(delta_r).all():
        raise ValueError("Non-finite deltas")
    if np.any(np.abs(dc[:, ~support]) > 1e-12):
        raise ValueError("Unsupported predicates cannot change")
    strict_relation = (delta_r >= -1e-12) & np.all(dc[:, support] >= -1e-12, axis=1)
    image_relation = (delta_r >= -1e-12) & (dc[:, support].sum(1) >= -1e-12)
    return dict(effect=(gain != 0) | ~strict_relation,
                strict=(gain > 0) & strict_relation,
                image_guard=(gain > 0) & image_relation,
                identity_only=gain > 0)


def choose_top(groups, indices, score, eligible, n_images):
    """Return action-row indices, not labels. Ties use original proposal index."""
    groups, indices, score, eligible = map(np.asarray, (groups, indices, score, eligible))
    if not (groups.shape == indices.shape == score.shape == eligible.shape):
        raise ValueError("Unaligned candidate rows")
    if not np.isfinite(score).all():
        raise ValueError("Non-finite selection score")
    selected = np.full(n_images, -1, dtype=int)
    for j in np.flatnonzero(eligible):
        image = groups[j]
        old = selected[image]
        if old < 0 or (-score[j], indices[j]) < (-score[old], indices[old]):
            selected[image] = j
    return selected


def macro_delta(dc, support, weights=None):
    """Native fixed-50 macro: mean over per-predicate image means."""
    dc, support = np.asarray(dc, dtype=float), np.asarray(support, dtype=float)
    if dc.shape != support.shape or dc.ndim != 2 or dc.shape[1] != 50:
        raise ValueError("Native mR requires aligned image-by-50 arrays")
    if weights is None:
        sums, counts = dc.sum(0), support.sum(0)
    else:
        sums, counts = weights @ dc, weights @ support
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).mean(-1)


def policy_summary(selection, groups, gain, delta_r, delta_class, support,
                   positive_count, folds, draws=2000):
    n = len(selection)
    di, dr, dm = np.zeros(n), np.zeros(n), np.zeros((n, 50))
    changed = selection >= 0
    rows = selection[changed]
    if not np.array_equal(np.asarray(groups)[rows], np.flatnonzero(changed)):
        raise ValueError("Selection crosses image boundaries")
    di[changed], dr[changed], dm[changed] = gain[rows], delta_r[rows], delta_class[rows]
    pos = np.asarray(positive_count, dtype=float)
    if pos.sum() <= 0:
        raise ValueError("No identity support")
    point = [float(di.sum()/pos.sum()), float(dr.mean()), float(macro_delta(dm, support))]
    rng = np.random.default_rng(17029)
    boot = []
    for start in range(0, draws, 100):
        count = min(100, draws-start)
        ix = rng.integers(0, n, (count, n))
        weights = np.stack([np.bincount(row, minlength=n) for row in ix]).astype(float)
        denom = weights @ pos
        if np.any(denom == 0):
            raise ValueError("Empty bootstrap denominator")
        boot.append(np.column_stack([(weights @ di)/denom, (weights @ dr)/n,
                                     macro_delta(dm, support, weights)]))
    boot = np.concatenate(boot)
    fold_rows = []
    for fold in sorted(set(folds)):
        take = np.asarray(folds) == fold
        fold_rows.append(dict(fold=int(fold), images=int(take.sum()),
            selected_images=int(changed[take].sum()), net_correct=int(di[take].sum()),
            identity_delta=float(di[take].sum()/pos[take].sum()),
            R50_delta=float(dr[take].mean()), mR50_delta=float(macro_delta(dm[take], support[take]))))
    names = ["object", "R50", "mR50"]
    lower = dict(zip(names, np.quantile(boot, .05, axis=0).tolist()))
    return dict(images=n, selected_images=int(changed.sum()), positive_objects=int(pos.sum()),
        net_correct=int(di.sum()), positive_gain_images=int((di > 0).sum()),
        negative_gain_images=int((di < 0).sum()), zero_gain_selected_images=int((changed & (di == 0)).sum()),
        gross_correct_count_gain=int(di[di > 0].sum()), gross_correct_count_loss=int(-di[di < 0].sum()),
        delta=dict(zip(names, point)), image_bootstrap_95ci=dict(zip(names, np.quantile(boot, [.025, .975], axis=0).T.tolist())),
        image_bootstrap_one_sided_95_lower=lower, folds=fold_rows,
        descriptive_original_numerical_checks=dict(identity_point=point[0] >= .005,
            identity_lower=lower["object"] > 0, R_noninferior=lower["R50"] >= -.005,
            mR_noninferior=lower["mR50"] >= -.005),
        formal_gate_accepted=False, independent_confirmation=False,
        interval_caveat="Image bootstrap conditions on cross-fitted predictions; models not refitted; no multiplicity correction")


def ranking_metrics(y, score, mean_prediction=None):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, score = np.asarray(y), np.asarray(score)
    positive = y > 0
    mixed = positive.any() and not positive.all()
    corr = spearmanr(y, score)[0] if np.unique(y).size > 1 and np.unique(score).size > 1 else None
    result = dict(actions=len(y), positive_actions=int(positive.sum()), prevalence=float(positive.mean()),
        positive_average_precision=float(average_precision_score(positive, score)) if mixed else None,
        positive_auc=float(roc_auc_score(positive, score)) if mixed else None,
        signed_gain_spearman=float(corr) if corr is not None and np.isfinite(corr) else None)
    bins = []
    for rows in np.array_split(np.argsort(score, kind="stable"), 10):
        if len(rows):
            bins.append(dict(actions=len(rows), score_mean=float(score[rows].mean()),
                gain_mean=float(y[rows].mean()), positive_rate=float(positive[rows].mean()),
                harmful_rate=float((y[rows] < 0).mean())))
    result["score_deciles_low_to_high"] = bins
    if mean_prediction is not None:
        mse, reference = float(np.mean((score-y)**2)), float(np.mean((mean_prediction-y)**2))
        result.update(mse=mse, fold_training_mean_mse=reference, mse_skill=1-mse/reference if reference else None)
    return result
