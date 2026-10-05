"""Audit VG exposure in GQA and re-score fixed historical predictions on held-out IDs.

No fitting, image resampling, label remapping, or checkpoint selection is performed.
The excluded raw VG training partition is deliberately broader than relation training.
"""
import json
import inspect
from pathlib import Path

import h5py
import numpy as np
import torch

from common import ROOT, OLD, ensure_storage, atomic_json, sha256


def correctness(payload, objects, relations):
    target = payload["object_targets"]
    obj = objects.argmax(1) == target
    rel = relations.argmax(1) == payload["relation_targets"]
    subject, endpoint = payload["relation_subject"], payload["relation_object"]
    if not np.array_equal(payload["object_image_ids"][subject], payload["relation_image_ids"]):
        raise ValueError("Subject image does not match relation image")
    if not np.array_equal(payload["object_image_ids"][endpoint], payload["relation_image_ids"]):
        raise ValueError("Object image does not match relation image")
    both = obj[subject] & obj[endpoint]
    return dict(object_top1=obj, predicate_hit_at_1=rel, triplet_hit_at_1=both & rel,
                endpoint_identity_disagreement_rate=~both)


def micro_interval(values, image_ids, seed=17):
    """Resample images, preserving all rows of each selected image and micro weighting."""
    keys, index = np.unique(image_ids, return_inverse=True)
    if not len(keys):
        return dict(value=None, bootstrap_95ci=None, images=0, denominator=0)
    counts = np.bincount(index)
    totals = np.bincount(index, weights=values.astype(float))
    interval = None
    if len(keys) > 1:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(keys), size=(2000, len(keys)))
        boots = totals[draws].sum(1) / counts[draws].sum(1)
        interval = np.quantile(boots, [.025, .975]).tolist()
    return dict(value=float(np.mean(values)), bootstrap_95ci=interval,
                images=len(keys), denominator=len(values))


def summarize(payload, values, baseline, ids):
    result = {}
    for key, correct in values.items():
        images = payload["object_image_ids" if key == "object_top1" else "relation_image_ids"]
        selected = np.isin(images, list(ids))
        result[key] = micro_interval(correct[selected], images[selected])
        delta = correct.astype(float) - baseline[key].astype(float)
        result[key]["paired_change_from_base"] = micro_interval(delta[selected], images[selected])
    osel = np.isin(payload["object_image_ids"], list(ids))
    rsel = np.isin(payload["relation_image_ids"], list(ids))
    result["support"] = dict(images=len(ids), objects=int(osel.sum()), relations=int(rsel.sum()),
                             object_classes=len(np.unique(payload["object_targets"][osel])),
                             predicate_classes=len(np.unique(payload["relation_targets"][rsel])))
    return result


def main():
    ensure_storage()
    torch.set_num_threads(2)
    cache = OLD / "data/derived/predictions/experiment_5_external/gqa/pysgg_tde_motifs_vg_live"
    archive = OLD / "artifacts/experiment_5/tde_motifs_causal_external_exp5_20260727/external/gqa/TDE_Neural_Motifs"
    image_path = OLD / "data/vg/image_data.json"
    h5_path = OLD / "data/vg/VG-SGG-with-attri.h5"
    gqa_path = OLD / "data/gqa/val_sceneGraphs.json"
    images = json.loads(image_path.read_text())
    images = [row for row in images if row["image_id"] not in {1592, 1722, 4616, 4617}]
    ids = np.asarray([str(row["image_id"]) for row in images])
    with h5py.File(h5_path, "r") as handle:
        split = handle["split"][:]
        valid = (handle["img_to_first_box"][:] >= 0) & (handle["img_to_first_rel"][:] >= 0)
    if len(ids) != len(split) or len(set(ids)) != len(ids):
        raise ValueError("VG image metadata cannot be aligned unambiguously")
    relation_train_indices = np.flatnonzero((split == 0) & valid)[5000:]
    vg_sets = dict(raw_train=set(ids[split == 0]), relation_train=set(ids[relation_train_indices]),
                   heldout_test=set(ids[split == 2]), all_vg=set(ids))
    gqa_ids = set(json.loads(gqa_path.read_text()))
    metadata = json.loads((cache / "metadata.json").read_text())
    with np.load(cache / "predictions.npz", allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    for key in ["object_image_ids", "relation_image_ids"]:
        payload[key] = payload[key].astype(str)
    cache_ids = set(payload["object_image_ids"]) | set(payload["relation_image_ids"])
    if not cache_ids <= gqa_ids:
        raise ValueError("Cached image IDs are not GQA validation IDs")
    subsets = dict(published_cached_overlap=cache_ids,
                   exclude_raw_vg_train=cache_ids - vg_sets["raw_train"],
                   known_vg_test_only=cache_ids & vg_sets["heldout_test"])
    base = correctness(payload, payload["object_scores"], payload["relation_scores"])
    results = {key: {"base": summarize(payload, base, base, selected)} for key, selected in subsets.items()}
    original = json.loads((archive / "supervised_control/seed_17/summary.json").read_text())
    for key, values in base.items():
        if not np.isclose(float(values.mean()), original["base"][key], atol=1e-6):
            raise ValueError("Base metric failed to reproduce: " + key)
    state_paths = sorted((ROOT / "imported/experiment5").rglob("mitigated_state_dict.pth"))
    if len(state_paths) != 6:
        raise ValueError("Expected six historical adaptation states")
    sources = [Path(__file__), Path(__file__).with_name("common.py"), image_path, h5_path,
               gqa_path, cache / "metadata.json", cache / "predictions.npz"]
    for path in state_paths:
        load_options = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
        checkpoint = torch.load(path, map_location="cpu", **load_options)
        if checkpoint["base_checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("Adaptation base checkpoint mismatch")
        state = checkpoint["grounding_state_dict"]
        outputs = []
        for name, prefix in [("object_scores", "entity_calibrator"), ("relation_scores", "relation_calibrator")]:
            scores = torch.from_numpy(payload[name]).float()
            outputs.append(torch.nn.functional.linear(scores, state[prefix + ".weight"].float(),
                           state[prefix + ".bias"].float()).numpy())
        values = correctness(payload, *outputs)
        mode, seed = path.parents[2].name, path.parent.name
        summary_path = archive / mode / seed / "summary.json"
        reference = json.loads(summary_path.read_text())
        if sha256(path) != reference["mitigation_state"]["sha256"]:
            raise ValueError("Historical state hash differs from published summary")
        for key, correct in values.items():
            if not np.isclose(float(correct.mean()), reference["evaluated"][key], atol=1e-6):
                raise ValueError("Adapted metric failed to reproduce: " + key)
        for subset, selected in subsets.items():
            results[subset][mode + "/" + seed] = summarize(payload, values, base, selected)
        sources.extend([path, summary_path])
    record = dict(status="complete", protocol="historical_GQA_prediction_exposure_audit_v1",
        source_sha256={str(path): sha256(path) for path in sources},
        original_cache_metadata=metadata,
        full_gqa=dict(images=len(gqa_ids), vg_relation_train_overlap=len(gqa_ids & vg_sets["relation_train"]),
                      vg_raw_train_overlap=len(gqa_ids & vg_sets["raw_train"]),
                      known_vg_test=len(gqa_ids & vg_sets["heldout_test"])),
        cached_gqa=dict(images=len(cache_ids), vg_relation_train_overlap=len(cache_ids & vg_sets["relation_train"]),
                        vg_raw_train_overlap=len(cache_ids & vg_sets["raw_train"]),
                        unknown_vg_split=sorted(cache_ids-vg_sets["all_vg"])),
        subsets={name: sorted(selected) for name, selected in subsets.items()}, results=results,
        interpretation=["Cross-annotation evaluation is not automatically unseen-image transfer.",
            "Known VG-test-only is the conservative subset; unknown VG IDs are not assumed unseen.",
            "No guarantee about ImageNet or foundation pretraining overlap is made.",
            "This audit concerns VG-trained TDE-Motifs, not PSG-trained identity probes.",
            "GT-box, exact shared-label Hit@1 is not native GQA SGDet R@K.",
            "Historical affine calibration is re-evaluated, not newly optimized.",
            "Image-bootstrap intervals do not include training-seed uncertainty."])
    atomic_json(ROOT / "results/external_overlap_audit/summary.json", record)
    print(json.dumps({key: record[key] for key in ["status", "full_gqa", "cached_gqa"]}), flush=True)


if __name__ == "__main__":
    main()
