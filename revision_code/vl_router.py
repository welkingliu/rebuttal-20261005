"""GT-free proposal routing features and a fixed, two-stage utility selector."""
import numpy as np
import torch
from torch.nn import functional as F


FEATURE_NAMES = ["legacy_%02d" % i for i in range(14)] + [
    "native_nms_matches_argmax", "changed_proposal_fraction", "changed_top50_fraction",
    "top50_tuple_change_fraction", "expert_log_gain_mean", "expert_log_gain_min",
    "expert_log_gain_max", "prototype_cos_gain_mean", "prototype_cos_gain_min",
    "old_label_native_conf_mean", "candidate_background_probability", "candidate_area",
    "class_box_iou_min"]


def box_iou_aligned(a, b):
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2]) + 1).clamp_min(0).prod(-1)
    aa = (a[:, 2:] - a[:, :2] + 1).clamp_min(0).prod(-1)
    bb = (b[:, 2:] - b[:, :2] + 1).clamp_min(0).prod(-1)
    return inter / (aa + bb - inter).clamp_min(1e-12)


def action_features(index, legacy, baseline, features, expert, centers, base, trial):
    """Only model outputs, image geometry and training-derived prototypes enter X."""
    base, trial = base.to("cpu"), trial.to("cpu")
    old, new = base.get_field("pred_labels"), trial.get_field("pred_labels")
    changed = (old != new).nonzero(as_tuple=False).flatten()
    p = baseline.softmax(-1)
    pairs = base.get_field("rel_pair_idxs")[:50]
    top_nodes = torch.unique(pairs.flatten())
    endpoint_changes = int(torch.isin(top_nodes, changed).sum()) if hasattr(torch, "isin") else sum(int(i in changed) for i in top_nodes)
    newpairs = trial.get_field("rel_pair_idxs")[:50]
    if len(pairs) != len(newpairs):
        raise ValueError("Native pair count changed")
    if len(pairs):
        # Changes in order, endpoint identity or selected class box all affect matching.
        tuple_changed = ((pairs != newpairs).any(-1)
            | (old[pairs] != new[newpairs]).any(-1)
            | (base.bbox[pairs] != trial.bbox[newpairs]).any(-1).any(-1))
        pair_fraction = float(tuple_changed.float().mean())
    else:
        pair_fraction = 0.
    if len(changed):
        log_gain = (expert[changed, new[changed]-1].clamp_min(1e-12).log()
                    - expert[changed, old[changed]-1].clamp_min(1e-12).log())
        cosine = (features[changed] * (centers[new[changed]-1] - centers[old[changed]-1])).sum(-1)
        gains = [float(log_gain.mean()), float(log_gain.min()), float(log_gain.max()),
                 float(cosine.mean()), float(cosine.min()), float(p[changed, old[changed]].mean()),
                 float(box_iou_aligned(base.bbox[changed], trial.bbox[changed]).min())]
    else:
        gains = [0., 0., 0., 0., 0., 0., 1.]
    wh = (base.bbox[index, 2:] - base.bbox[index, :2] + 1).clamp_min(0)
    extra = [float(old[index] == baseline[index, 1:].argmax()+1), len(changed)/max(len(old),1),
             endpoint_changes/max(len(top_nodes),1), pair_fraction] + gains[:6] + [
             float(p[index,0]), float(wh.prod()/(base.size[0]*base.size[1])), gains[6]]
    result = np.r_[legacy[index].numpy(), extra].astype(np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Routing feature contract failed")
    return result


def fit_logistic(x, y):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both outcome classes are required")
    mean = x.mean(0); scale = np.maximum(x.std(0), .001)
    estimator = LogisticRegression(C=.1, solver="lbfgs", max_iter=2000, random_state=17)
    estimator.fit((x-mean)/scale, y)
    if int(estimator.n_iter_.max()) >= 2000:
        raise RuntimeError("Router fitting did not converge")
    return dict(mean=mean.tolist(), scale=scale.tolist(), weight=estimator.coef_[0].tolist(),
                bias=float(estimator.intercept_[0]), iterations=int(estimator.n_iter_[0]), samples=len(y))


def predict_logistic(model, x):
    z = ((x-np.asarray(model["mean"]))/np.asarray(model["scale"])) @ np.asarray(model["weight"]) + model["bias"]
    return 1./(1.+np.exp(-np.clip(z,-60,60)))


def fit_selector(x, effect, safe):
    effect, safe = np.asarray(effect,dtype=bool), np.asarray(safe,dtype=bool)
    if np.any(safe & ~effect):
        raise ValueError("Safe utility must be an effective action")
    if int(safe.sum()) < 10 or int((effect & ~safe).sum()) < 10:
        raise ValueError("Insufficient positive/negative development outcomes")
    return dict(effect=fit_logistic(x,effect), safety=fit_logistic(x[effect],safe[effect]),
                feature_names=FEATURE_NAMES, probability_min=.6, max_actions_per_image=1)


def select_action(model, x, indices):
    if len(x) == 0:
        return None, []
    pe = predict_logistic(model["effect"],x)
    ps = predict_logistic(model["safety"],x)
    utility = pe*(2*ps-1)
    rows = [dict(proposal=int(i), effect_probability=float(a), safety_probability=float(b),
                 expected_signed_utility=float(u)) for i,a,b,u in zip(indices,pe,ps,utility)]
    eligible = [r for r in rows if r["safety_probability"] >= model["probability_min"] and r["expected_signed_utility"]>0]
    chosen = min(eligible,key=lambda r:(-r["expected_signed_utility"],r["proposal"]))["proposal"] if eligible else None
    return chosen, rows
