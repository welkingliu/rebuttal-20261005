"""Decision rules for a privileged SGDet feasibility audit, not a repair gate."""
import math


def improves_identity(current, trial):
    if trial["positive_objects"] != current["positive_objects"]:
        raise ValueError("The identity denominator changed")
    return trial["post_nms_correct"] > current["post_nms_correct"]


def preserves_relations(current, trial, k_index):
    a, b = current["recalls"][k_index], trial["recalls"][k_index]
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Nonfinite relation recall")
    if b < a - 1e-12:
        return False
    for a, b in zip(current["class_recalls"][k_index], trial["class_recalls"][k_index]):
        if (a is None) != (b is None):
            raise ValueError("Predicate support changed")
        if a is not None and (not math.isfinite(b) or b < a - 1e-12):
            return False
    return True


def candidate_order(indices, confidence):
    """Fixed, label-free order. Ties are broken by original proposal index."""
    return sorted(indices, key=lambda i: (float(confidence[i]), i))

