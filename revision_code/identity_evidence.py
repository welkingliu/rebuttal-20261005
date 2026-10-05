"""Image-clustered, denominator-explicit summaries of identity interventions."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def geometry_assignment(boxes, gt_boxes, threshold=.5):
    """Class-blind one-to-one matching: maximize valid matches, then pixel IoU."""
    boxes, gt_boxes = np.asarray(boxes), np.asarray(gt_boxes)
    mapping = np.full(len(boxes), -1, dtype=np.int64)
    assigned_iou = np.zeros(len(boxes), dtype=float)
    if not len(boxes) or not len(gt_boxes):
        return mapping, assigned_iou
    lo = np.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
    inter = np.maximum(hi-lo+1, 0).prod(-1)
    area = lambda b: np.maximum(b[:, 2:]-b[:, :2]+1, 0).prod(-1)
    overlap = inter/np.maximum(area(boxes)[:, None]+area(gt_boxes)[None, :]-inter, 1e-12)
    valid = overlap >= threshold
    # A single extra threshold-valid match dominates any possible sum-IoU gain.
    utility = valid*(min(len(boxes), len(gt_boxes))+1) + np.where(valid, overlap, 0.)
    rows, cols = linear_sum_assignment(-utility)
    good = valid[rows, cols]
    mapping[rows[good]] = cols[good]
    assigned_iou[rows[good]] = overlap[rows[good], cols[good]]
    return mapping, assigned_iou


def paired_micro(rows, seed=17, trials=2000):
    """Rows contain numerator and denominator per image, including zero-support images."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    support = int(rows[:, 1].sum())
    eligible = rows[:, 1] > 0
    if not support:
        return dict(value=None, bootstrap_95ci=None, denominator=0, images=0, bootstrap_unit="image")
    # Zero-support images have no data for this conditional estimand.
    active = rows[eligible]
    ci = None
    if len(active) >= 2:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(active), size=(trials, len(active)))
        totals = active[samples].sum(1)
        ci = np.quantile(totals[:, 0]/totals[:, 1], [.025, .975]).tolist()
    return dict(value=float(rows[:, 0].sum()/support), bootstrap_95ci=ci, denominator=support,
                images=int(eligible.sum()), bootstrap_unit="image", bootstrap_trials=trials)


def cell_counts(clean, prediction, target, selected):
    clean, prediction, target, selected = map(np.asarray, (clean, prediction, target, selected))
    if not (clean.shape == prediction.shape == target.shape == selected.shape):
        raise ValueError("Per-relation prediction or subset shape mismatch")
    before, after = clean == target, prediction == target
    take = lambda values: int(np.count_nonzero(values & selected))
    return dict(support=int(selected.sum()), clean_correct=take(before), correct=take(after),
                repaired=take(~before & after), damaged=take(before & ~after),
                changed=take(clean != prediction), clean_wrong=take(~before))


def summarize_cells(cells):
    totals = {key: sum(row[key] for row in cells) for key in cells[0]} if cells else {}
    def measure(numerator, denominator):
        return paired_micro([[numerator(row), row[denominator]] for row in cells])
    return dict(counts=totals,
        clean_hit1=measure(lambda r: r["clean_correct"], "support"),
        hit1=measure(lambda r: r["correct"], "support"),
        delta_hit1=measure(lambda r: r["correct"]-r["clean_correct"], "support"),
        repair_rate_given_clean_wrong=measure(lambda r: r["repaired"], "clean_wrong"),
        damage_rate_given_clean_correct=measure(lambda r: r["damaged"], "clean_correct"),
        predicate_change_rate=measure(lambda r: r["changed"], "support"))


def image_record(payload):
    truth, clean = payload["relation_gt"], payload["clean_prediction"]
    pairs = payload["relation_pairs"].astype(int)
    correct_object = payload["object_gt"] == payload["object_pred"]
    if pairs.size and (pairs.min() < 0 or pairs.max() >= len(correct_object)):
        raise ValueError("Relation endpoint outside object array")
    identity_wrong = ~(correct_object[pairs[:, 0]] & correct_object[pairs[:, 1]])
    summaries = {}
    for key in sorted(k for k in payload if k.startswith("prediction_")):
        name = key[len("prediction_"):]
        incident = payload["incident_"+name].astype(bool)
        subsets = dict(all=np.ones(len(truth), dtype=bool), touched=incident,
                       untouched=~incident, native_endpoint_wrong=identity_wrong,
                       native_endpoint_correct=~identity_wrong)
        summaries[name] = {subset: cell_counts(clean, payload[key], truth, selected)
                           for subset, selected in subsets.items()}
    # Oracle output labels change matching without claiming changed reasoning.
    correctness = clean == truth
    bound = dict(relations=len(truth), predicate_correct=int(correctness.sum()),
                 identity_and_predicate_correct=int((correctness & ~identity_wrong).sum()),
                 lost_to_identity_with_predicate_correct=int((correctness & identity_wrong).sum()))
    return dict(conditions=summaries, fixed_pair_label_accounting=bound)


def summarize_records(records):
    names = list(records[0]["conditions"]) if records else []
    if any(set(record["conditions"]) != set(names) for record in records):
        raise ValueError("Condition set differs between images")
    results = {name: {subset: summarize_cells([record["conditions"][name][subset] for record in records])
                     for subset in records[0]["conditions"][name]} for name in names}
    interaction = None
    if all(key in names for key in ["oracle", "oracle_frequency", "oracle_both"]):
        terms = []
        for record in records:
            c = record["conditions"]
            row = c["oracle_both"]["all"]
            value = row["correct"]-c["oracle"]["all"]["correct"]-c["oracle_frequency"]["all"]["correct"]+row["clean_correct"]
            terms.append([value, row["support"]])
        interaction = paired_micro(terms)
    bound = [record["fixed_pair_label_accounting"] for record in records]
    return dict(conditions=results, oracle_nonadditivity_delta=interaction,
                fixed_pair_identity_matching_loss=paired_micro([[r["lost_to_identity_with_predicate_correct"],r["relations"]] for r in bound]),
                fixed_pair_accounting_scope="Frozen predicate predictions on evaluable annotated pairs; not SGDet recall or mitigation",
                estimand="Relation-micro statistics, clustered by image; seeds are corruption seeds, not checkpoint seeds")
