"""Apply the frozen V-K primary router without labels or any fitting."""
import torch
from torch.nn import functional as F

import vj_mac as frozen


def repair(features, baseline, state):
    if (features.ndim != 2 or features.shape[1] != 768
            or baseline.shape != (len(features), 151)
            or not torch.isfinite(features).all() or not torch.isfinite(baseline).all()):
        raise ValueError("Invalid native proposal features/logits")
    if not torch.allclose(features.norm(dim=-1), torch.ones_like(features[:, 0]), atol=2e-5):
        raise ValueError("DINO feature normalization differs")
    state = {k: {n: v.to(baseline) for n, v in state[k].items()} if isinstance(state[k], dict)
             else state[k].to(device=baseline.device) for k in ["expert_head", "router", "centers", "counts"]}
    expert = F.linear(features, state["expert_head"]["weight"], state["expert_head"]["bias"]).softmax(-1)
    _, foreground, candidate = frozen.probabilities(baseline, expert)
    visual, supported = frozen.visual_features(features, foreground, candidate, state["centers"], state["counts"])
    inputs = torch.cat([frozen.confidence_features(foreground, expert), visual], -1)
    probabilities, selected, confidence = frozen.route(baseline, expert, inputs, state["router"], supported)
    # Leave unselected native logits bit-identical, not merely softmax-equivalent.
    logits = torch.where(selected[:, None], probabilities.clamp_min(1e-30).log(), baseline)
    if not torch.allclose(logits.softmax(-1), probabilities, atol=2e-7, rtol=1e-5):
        raise RuntimeError("Repaired probabilities changed during logit conversion")
    return dict(logits=logits, probabilities=probabilities, selected=selected,
                repair_probability=confidence)


def validate_fixture(fixture, state):
    actual = repair(fixture["features"], fixture["baseline"], state)
    errors = {}
    for key in ["probabilities", "repair_probability"]:
        errors[key] = float((actual[key] - fixture[key]).abs().max())
        if not torch.allclose(actual[key], fixture[key], atol=2e-6, rtol=1e-5):
            raise RuntimeError("Frozen Mac/native numerical parity failed: " + key)
    if not torch.equal(actual["selected"], fixture["selected"]):
        raise RuntimeError("Frozen V-K selection mask differs")
    if not torch.equal(actual["logits"][~actual["selected"]], fixture["baseline"][~actual["selected"]]):
        raise RuntimeError("Unselected proposals changed")
    return dict(objects=len(fixture["features"]), selected=int(actual["selected"].sum()), max_errors=errors)
