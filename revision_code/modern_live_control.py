"""SGTR live, end-to-end paired spatial controls on a fixed VG subset."""
import argparse
import json
import time

import numpy as np
from PIL import Image
import torch

from common import OLD, ROOT, atomic_json, ensure_storage, sha256, output_path
from extension_protocol import SPEC, verify, immutable
from paired_visual_control import mask, selected_nodes


def metrics(prediction, batch):
    from sgg_core.audits.standard_sgg_eval import _build_ranked_triplets, _ground_truth, _matched_gt_indices
    pred = {k: torch.from_numpy(v) for k, v in prediction.items()}
    pred["pred_rel_score_mode"] = "independent_probabilities"
    ranked = _build_ranked_triplets(pred, batch, "sgdet", True)
    gt = _ground_truth(batch)
    labels = gt["predicates"].numpy()
    count = np.bincount(labels, minlength=51)[1:]
    rr, mr = [], []
    for k in SPEC["R6"]["Ks"]:
        matched = sorted(_matched_gt_indices(ranked.top(k), gt, .5))
        hits = np.bincount(labels[matched], minlength=51)[1:]
        rr.append(len(matched) / max(len(labels), 1))
        mr.append([float(h / c) if c else None for h, c in zip(hits, count)])
    return dict(R=rr, class_recall=mr, relations=len(labels))


def aggregate(rows):
    means = {}
    conditions = list(rows[0]["conditions"])
    for condition in conditions:
        rr = np.array([r["conditions"][condition]["R"] for r in rows])
        mr = np.array([r["conditions"][condition]["class_recall"] for r in rows], dtype=float)
        count = np.isfinite(mr).sum(0)
        macro = np.divide(np.nansum(mr, axis=0), count, out=np.zeros_like(count, dtype=float), where=count > 0).mean(1)
        means[condition] = dict(R=rr.mean(0).tolist(), mR=macro.tolist())
    rng = np.random.default_rng(17)
    contrasts = {}
    for strength in SPEC["R6"]["strengths"]:
        key, control = "key_%.2f" % strength, "unrelated_%.2f" % strength
        values = np.array([np.array(r["conditions"][key]["R"]) - np.array(r["conditions"][control]["R"]) for r in rows])
        draws = [values[rng.integers(0, len(rows), len(rows))].mean(0) for _ in range(SPEC["R6"]["bootstrap"])]
        contrasts[str(strength)] = dict(key_minus_unrelated_R=values.mean(0).tolist(),
                                       paired_image_95CI=np.quantile(draws, [.025, .975], axis=0).tolist())
    return dict(status="complete", images=len(rows), relations=sum(r["conditions"]["clean"]["relations"] for r in rows),
                means=means, contrasts=contrasts, Ks=SPEC["R6"]["Ks"],
                interpretation=SPEC["R6"]["interpretation"], class_mean="official-style image-mean per predicate then all 50 predicates")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--integration_only", action="store_true")
    args = p.parse_args(); ensure_storage(); verify(); torch.set_num_threads(4)
    import export_sgtr_vg_predictions as exporter
    from sgg_core.data.data_utils import build_vg_test_loader
    exporter.PROJECT_ROOT = ROOT
    out = ROOT / "results/R6_sgtr_live"
    weight_dir = OLD / "checkpoints/sgg/weights/sgtr/vg/sgtr_vg_new_pth"
    weight = weight_dir / "model_0095999.pth"
    digest = sha256(weight)
    if digest != "73a96a8a7fd33ccca4b841411c2101c93d29185da4b8f69cb04076a9620e05fc":
        raise RuntimeError("SGTR checkpoint provenance mismatch")
    ids = json.loads((ROOT / "results/R4_paired/tde_motifs/protocol.json").read_text())["candidate_images"]
    if len(ids) != SPEC["R6"]["candidate_images"]:
        raise RuntimeError("Unexpected R4 candidate denominator")
    protocol = dict(spec=SPEC["R6"], image_ids=ids, checkpoint_sha256=digest,
                    config_sha256=sha256(weight_dir / "config.json"), evaluator_sha256=sha256(OLD / "sgg_core/audits/standard_sgg_eval.py"),
                    exporter_sha256=sha256(OLD / "scripts/export_sgtr_vg_predictions.py"),
                    source_marker=json.loads((OLD / "external/official_repos/SGTR/.official_source.json").read_text()))
    pf = out / "protocol.json"; immutable(pf, protocol)
    ds = build_vg_test_loader(str(OLD / "data/vg/v1.4"), num_samples=1000000000, batch_size=1,
                             split=2, include_proxy_features=False, include_raw_images=True).dataset
    lookup = {str(ds.index_to_image_meta[int(index)]["image_id"]): i for i, index in enumerate(ds.image_indices)}
    model, _ = exporter.load_model(OLD / "external/official_repos/SGTR", weight_dir / "config.json", weight, torch.device("cuda"), "vg")
    def infer(batch):
        inputs, h, w, _ = exporter.prepare_input(batch, torch.device("cuda"), 150, "vg")
        with torch.no_grad():
            output = model(inputs)[0]
        return exporter.convert_output(output, h, w, 150, 50)
    checks = []
    for iid in ids[:3]:
        batch = ds[lookup[iid]]
        value = infer(batch)
        cache = OLD / "artifacts/prediction_cache/sgtr_vg/predictions/sgdet" / (iid + ".npz")
        with np.load(cache) as archived:
            errors = {}
            for k, a in value.items():
                b = archived[k]
                if a.shape != b.shape or not np.allclose(a, b, atol=1e-4, rtol=1e-4):
                    raise RuntimeError("Live SGTR differs from archived clean output: %s %s" % (iid, k))
                errors[k] = float(np.max(np.abs(a - b))) if a.size else 0.
        checks.append(dict(image_id=iid, max_errors=errors))
    atomic_json(out / "integration.json", dict(status="complete", checks=checks, source_sha256=sha256(pf)))
    print("[INTEGRATION OK] clean live SGTR equals archived output", flush=True)
    if args.integration_only:
        return
    rows, excluded = [], []
    start = time.monotonic()
    for n, iid in enumerate(ids):
        path = out / "images" / (iid + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["protocol_sha256"] != sha256(pf):
                raise RuntimeError("Stale modern-control record")
            rows.append(row); continue
        batch = ds[lookup[iid]]
        _, h, w = batch["image"].shape
        boxes = batch["boxes"].numpy() * [w, h, w, h]
        nodes, reason = selected_nodes(boxes, batch["rel_pairs"].numpy())
        if nodes is None:
            excluded.append(dict(image_id=iid, reason=reason)); continue
        image = Image.fromarray((batch["image"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))
        payload = {}; conditions = {}
        clean = infer(batch)
        conditions["clean"] = metrics(clean, batch)
        payload.update({"clean_" + k: v for k, v in clean.items()})
        for mode, node in [("key", nodes[0]), ("unrelated", nodes[1])]:
            for strength in SPEC["R6"]["strengths"]:
                changed = mask(image, boxes[node], strength)
                if np.array_equal(np.asarray(image), np.asarray(changed)):
                    raise RuntimeError("Perturbation is a no-op")
                altered = dict(batch, image=torch.from_numpy(np.asarray(changed).copy()).permute(2, 0, 1).float() / 255.)
                pred = infer(altered)
                name = "%s_%.2f" % (mode, strength)
                conditions[name] = metrics(pred, batch)
                payload.update({name + "_" + k: v for k, v in pred.items()})
        np.savez_compressed(output_path(out / "predictions" / (iid + ".npz")), **payload)
        row = dict(image_id=iid, key=nodes[0], unrelated=nodes[1], area_ratio=nodes[2],
                   conditions=conditions, protocol_sha256=sha256(pf))
        atomic_json(path, row); rows.append(row)
        if len(rows) % 10 == 0:
            progress = dict(candidates=n + 1, total=len(ids), evaluated=len(rows), seconds=time.monotonic() - start)
            atomic_json(out / "progress.json", progress); print(json.dumps(progress), flush=True)
    if len(rows) < SPEC["R6"]["minimum_images"]:
        raise RuntimeError("Insufficient paired support")
    atomic_json(out / "summary.json", dict(aggregate(rows), excluded=excluded))
    print("[COMPLETE] R6 modern live intervention", flush=True)


if __name__ == "__main__":
    main()
