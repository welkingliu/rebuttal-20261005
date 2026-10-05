"""Synthetic nonzero route test only; never a scientific repair candidate."""
from pathlib import Path
import torch

AUDIT_SOURCE = Path(__file__).resolve()

from common import atomic_json, sha256
from vu_experiment import bundle_for
from vv_protocol import load, OUT
from vv_math import make_head, predict, project_channels
from vv_experiment import records_for
from vn_train_proposals import make_post, replay
from vk_gate_native import configure_backend


def main():
    configure_backend()
    torch.set_num_threads(2)
    protocol, reg, lane = load(True)
    bundle, _ = bundle_for(protocol["vu_protocol_sha256"], lane)
    records = records_for(protocol, bundle, protocol["image_ids"][:3])
    _, post = make_post(OUT/lane/"route_audit/runtime")
    head = make_head("cuda")
    results = []
    with torch.no_grad():
        head.weight.normal_(0, .2)
        head.bias.normal_(0, .2)
        for rec in records:
            native = rec["raw"]["native_logits"].float().cuda()
            updated = predict(rec["visual"].cuda(), native, head, "joint_visual")
            baseline = replay(post, rec["raw"], native)
            row = dict(image_id=rec["iid"])
            for name,logits in [("joint",updated),
                    ("foreground",project_channels(native,updated,"foreground")),
                    ("background",project_channels(native,updated,"background"))]:
                prediction = replay(post,rec["raw"],logits)
                changes = {field:not torch.equal(prediction.get_field(field),baseline.get_field(field))
                    for field in ["pred_labels","pred_scores","rel_pair_idxs","pred_rel_scores"]}
                changes["boxes"] = not torch.equal(prediction.bbox,baseline.bbox)
                delta = float((logits.softmax(-1)-native.softmax(-1)).abs().max())
                if not delta>0 or not any(changes.values()):
                    raise RuntimeError("Nonzero synthetic update did not reach emitted predictions")
                row[name] = dict(probability_max_delta=delta,emitted_fields_changed=changes)
            results.append(row)
    value = dict(status="complete",protocol_sha256=reg,audit_source_sha256=sha256(AUDIT_SOURCE),
        synthetic_nonzero_head_discarded=True,efficacy_claim=False,images=len(results),records=results)
    atomic_json(OUT/lane/"route_audit/summary.json",value)
    print("[route-audit-ok]",reg,flush=True)


if __name__=="__main__":
    main()
