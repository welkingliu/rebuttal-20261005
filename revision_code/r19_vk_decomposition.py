"""Frozen V-K postprocessing decomposition; never a second acceptance gate."""
import argparse
import itertools
import json
import time

import numpy as np
import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import lock, read, sources, paired_ratio, transitions
from native_runtime import infer
from repro_experiment import dataset, save_torch
from ve_experiment import lookup
from vi_native import setup
from vb_native import OfficialMetrics, compare_predictions, targets
from vc_protocol import summarize_rows, KS
from vk_gate_native import configure_backend, tensor_digest
from vk_gate_protocol import OUT as VK, CACHE, MANIFEST, ids


OUT = ROOT / "results/R19_vk_mechanism"


def crossed_prediction(base, changed, labels, scores, boxes):
    from pysgg.structures.bounding_box import BoxList
    result = BoxList((changed if boxes else base).bbox.clone(), base.size, "xyxy")
    lab = (changed if labels else base).get_field("pred_labels")
    score = (changed if scores else base).get_field("pred_scores")
    pairs = base.get_field("rel_pair_idxs")
    rel = base.get_field("pred_rel_scores")
    rank = score[pairs[:, 0]] * score[pairs[:, 1]] * rel[:, 1:].max(-1)[0]
    order = torch.argsort(rank, descending=True)
    result.add_field("pred_labels", lab.clone())
    result.add_field("pred_scores", score.clone())
    result.add_field("rel_pair_idxs", pairs[order])
    result.add_field("pred_rel_scores", rel[order])
    return result


def cached_prediction_check(pred, path):
    from export_pysgg_vg_task import convert_prediction
    current = convert_prediction(pred.to("cpu"))
    with np.load(path) as stored:
        for key, value in current.items():
            if key not in stored or value.shape != stored[key].shape:
                raise RuntimeError("Prediction cache shape/key mismatch: " + key)
            if not np.allclose(value, stored[key], atol=1e-5, rtol=1e-5):
                raise RuntimeError("Prediction cache differs: " + key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["sgcls", "sgdet"], required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); ensure_storage(); torch.set_num_threads(2); configure_backend()
    out = OUT / a.task / ("smoke" if a.smoke else "formal")
    selected = ids()[:3] if a.smoke else ids()
    protocol = dict(version="r19_frozen_postprocessing_v1", task=a.task, image_ids=selected,
        old_gate_sha256=sha256(MANIFEST), original_gate_accepted=read(VK / "summary.json")["accepted"],
        semantics="Post-hoc descriptive replay of an already inspected validation gate; no fitting, selection or gate change",
        factorial="2x2x2 native/repaired emitted labels, emitted scores, class-specific boxes; fixed predicate probabilities and pair set",
        caveat="Crossed label/score/box arms are diagnostic counterfactual outputs, not deployable repair methods",
        sources=sources(["r19_vk_decomposition.py", "evidence_completion.py", "vi_native.py", "vf_native.py",
                         "vk_gate_native.py", "vk_gate_protocol.py", "vb_native.py", "vc_protocol.py", "ve_experiment.py"]))
    reg = lock(out / "protocol.json", protocol)
    model, cfg, transform, provenance, patch = setup(a.task, out)
    if provenance["checkpoint_sha256"] != read(MANIFEST)["baseline_assets"][a.task]["sha256"]:
        raise RuntimeError("Native model checkpoint changed")
    post = model.roi_heads.relation.post_processor
    if post.use_relness_ranking or post.BCE_loss or post.attribute_on:
        raise RuntimeError("Unsupported native ranking protocol")
    ds = dataset(cfg, "val"); mapping = lookup(ds); metric = OfficialMetrics(a.task)
    records = []; start = time.monotonic()
    from pysgg.structures.bounding_box import BoxList
    from export_pysgg_vg_task import convert_prediction
    for n, iid in enumerate(selected, 1):
        row_path = out / "images" / (iid + ".json")
        if row_path.exists():
            row = read(row_path)
            if row["protocol_sha256"] != reg: raise RuntimeError("Resume protocol mismatch")
            records.append(row)
            continue
        item = torch.load(str(CACHE / a.task / "gate/export" / (iid + ".pt")), map_location="cpu")
        update = torch.load(str(CACHE / a.task / "gate/updates" / (iid + ".pt")), map_location="cpu")
        if item["protocol_sha256"] != protocol["old_gate_sha256"] or update["protocol_sha256"] != item["protocol_sha256"]:
            raise RuntimeError("Frozen gate cache provenance changed")
        patch.enabled = False
        with torch.no_grad():
            prediction, _ = infer(model, cfg, transform, ds, mapping[iid])
            c = patch.capture
            if not torch.equal(c["baseline"].cpu(), item["baseline"]):
                raise RuntimeError("Native logits differ from frozen gate")
            if tensor_digest(c["relation_logits"]) != item["predicate_sha256"] or tensor_digest(c["pairs"]) != item["pair_sha256"]:
                raise RuntimeError("Predicate/pair drift")
            if not torch.equal(update["native_logits"], item["baseline"]): raise RuntimeError("Update source drift")
            cached_prediction_check(prediction, VK / a.task / "gate/native/predictions" / (iid + ".npz"))
            def replay(logits):
                box = BoxList(c["proposal_boxes"].clone(), c["size"], "xyxy")
                if "boxes_per_cls" in c: box.add_field("boxes_per_cls", c["boxes_per_cls"].clone())
                return post.forward(([c["relation_logits"]], [logits]), [c["pairs"]], [box])[0]
            base = replay(c["baseline"])
            changed_logits = update["logits"].to(c["baseline"])
            repaired = replay(changed_logits)
            compare_predictions(base, prediction)
            cached_prediction_check(repaired, VK / a.task / "gate/selective/predictions" / (iid + ".npz"))
            gt = ds.get_groundtruth(mapping[iid], evaluation=True); y = targets(c, gt, a.task)
            flags = transitions(c["baseline"][:, 1:].argmax(-1).cpu().numpy()+1,
                changed_logits[:, 1:].argmax(-1).cpu().numpy()+1,
                base.get_field("pred_labels").cpu().numpy(), repaired.get_field("pred_labels").cpu().numpy(), y)
            rows = {}
            # Use native endpoint outputs directly: a second sort can reorder ties.
            for l, s, b in itertools.product([0, 1], repeat=3):
                name = "L%d_S%d_B%d" % (l, s, b)
                pred = base if (l,s,b)==(0,0,0) else repaired if (l,s,b)==(1,1,1) else crossed_prediction(base, repaired, l,s,b)
                rows[name] = metric.row(iid, pred, gt, dict(logits=changed_logits if s else c["baseline"]), y)
            record = dict(image_id=iid, protocol_sha256=reg, transitions=flags, arms=rows,
                          native_cache_reproduced=True, repaired_cache_reproduced=True,
                          fixed_predicate_and_pairs=True)
            raw = dict(image_id=iid, native_logits=c["baseline"].cpu(), repaired_logits=changed_logits.cpu(),
                       target=torch.from_numpy(y), proposal_boxes=c["proposal_boxes"].cpu(), size=c["size"],
                       pairs=c["pairs"].cpu(), relation_logits=c["relation_logits"].cpu(), protocol_sha256=reg)
            if "boxes_per_cls" in c: raw["boxes_per_cls"] = c["boxes_per_cls"].cpu()
            save_torch(ROOT / "cache/R19_vk_mechanism" / a.task / (iid + ".pt"), raw)
            atomic_json(row_path, record); records.append(record)
        if n % 10 == 0 or n == len(selected):
            elapsed = time.monotonic()-start
            progress = dict(stage="frozen_native_postprocessing_replay", images=n, total=len(selected), seconds=elapsed,
                            eta_seconds=elapsed/n*(len(selected)-n))
            atomic_json(out / "progress.json", progress); print(json.dumps(progress), flush=True)
    patch.close()
    aggregates = {}; ref = [r["arms"]["L0_S0_B0"] for r in records]
    for name in records[0]["arms"]:
        rows = [r["arms"][name] for r in records]
        aggregates[name] = dict(metrics=summarize_rows(rows),
            R50=paired_ratio([r["recalls"][KS.index(50)] for r in rows], [r["recalls"][KS.index(50)] for r in ref], np.ones(len(rows))),
            post_nms_identity=paired_ratio([r["post_nms_correct"] for r in rows], [r["post_nms_correct"] for r in ref], [r["positive_objects"] for r in ref]))
    transition_counts = np.sum([r["transitions"]["correctness_pattern_counts"] for r in records], axis=0).tolist()
    atomic_json(out / "summary.json", dict(status="complete", images=len(records), smoke=a.smoke,
        protocol_sha256=reg, results=aggregates, correctness_pattern_counts=transition_counts,
        bit_order=records[0]["transitions"]["bit_order"], original_joint_gate_accepted=False,
        new_repair_trained=False, confirmatory_claim=False, elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__": main()
