"""Generate a numerical compatibility fixture from old development inputs only."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

import vj_mac as original
from vk_gate_math import validate_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2)
    data=torch.load(str(a.bundle),map_location="cpu",weights_only=True)["data"]["development"]
    state=torch.load(str(a.checkpoint),map_location="cpu",weights_only=True)
    # Fixed first512 old-development regions, without correctness-based selection.
    x,logits=data["features"][:512],data["baseline"][:512]
    with torch.inference_mode():
        q=F.linear(x,state["expert_head"]["weight"],state["expert_head"]["bias"]).softmax(-1)
        _,fg,candidate=original.probabilities(logits,q)
        visual,eligible=original.visual_features(x,fg,candidate,state["centers"],state["counts"])
        inputs=torch.cat([original.confidence_features(fg,q),visual],-1)
        probabilities,selected,confidence=original.route(logits,q,inputs,state["router"],eligible)
    fixture=dict(features=x.clone(),baseline=logits.clone(),probabilities=probabilities.clone(),
        selected=selected.clone(),repair_probability=confidence.clone(),
        checkpoint_sha256=original.digest(a.checkpoint),source_sha256=original.digest(Path(__file__)),
        scope="Numerical replay of first512 old-development regions; no gate data or fitting")
    check=validate_fixture(fixture,state)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise RuntimeError("Preserve existing parity fixture")
    original.save(a.output,fixture)
    original.write(a.output.with_suffix(".json"),dict(status="complete",sha256=original.digest(a.output),**check))
    print(check,flush=True)


if __name__=="__main__":main()
