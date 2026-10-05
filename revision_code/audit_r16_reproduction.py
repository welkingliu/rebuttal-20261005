"""Read-only R16 audit; all audit products are separate from training runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import logging
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
from PIL import Image
import torch

from common import ROOT, OLD, atomic_json, output_path, sha256
import r16_motifs as r16

OUT = ROOT / "results/R16_code_audit_20261002"


def snapshot(task):
    source = r16.RUN / task / "formal/latest.pth"
    dest = output_path(OUT / "snapshots" / (task + ".pth"))
    if not dest.exists():
        # Atomic checkpoint replacement means this hard link pins a complete file.
        os.link(str(source), str(dest))
    return dest, torch.load(str(dest), map_location="cpu")


def static_audit():
    from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
    from repro_experiment import DeterministicImages
    from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler

    cfg = r16.configure("sgcls", OUT / "static")
    splits = {name: r16.dataset(cfg, name) for name in ["train", "val", "test"]}
    ids = {k: [Path(p).stem for p in ds.filenames] for k, ds in splits.items()}
    overlap = {a + "/" + b: len(set(ids[a]) & set(ids[b]))
               for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    assert not any(overlap.values())
    atomic_json(OUT / "split_ids.json", ids)
    count = {k: len(v) for k, v in ids.items()}
    assert count["val"] == 5000 and count["test"] == 26446
    comparison = {}
    with h5py.File(str(OLD / "data/vg/VG-SGG-with-attri.h5"), "r") as a, \
            h5py.File(str(OLD / "data/vg/v1.4/VG-SGG.h5"), "r") as b:
        for key in ["split", "img_to_first_box", "img_to_last_box", "img_to_first_rel",
                    "img_to_last_rel", "labels", "relationships", "predicates", "boxes_1024"]:
            comparison[key] = np.array_equal(a[key][:], b[key][:])
    assert all(comparison.values()), comparison
    ds = splits["train"]
    expected = DatasetCatalog.get(cfg.DATASETS.TRAIN[0], cfg)["args"]
    assert ds.filter_non_overlap == expected["filter_non_overlap"] == False
    assert ds.filter_duplicate_rels and not ds.flip_aug
    order = list(DeterministicImages(17, 0, 104, 666))
    assert order[40:] == list(DeterministicImages(17, 40, 104, 666))
    assert sorted(order[:17]) == list(range(17))
    dummy = torch.nn.Linear(2, 2)
    opt = make_optimizer(cfg, dummy, logging.getLogger("audit"), slow_heads=[], rl_factor=8.)
    scheduler = make_lr_scheduler(cfg, opt)
    for i in range(1, 701):
        opt.step(); scheduler.step(None, epoch=i)
    saved = copy.deepcopy(scheduler.state_dict())
    restored = make_lr_scheduler(cfg, opt)
    restored.load_state_dict(saved)
    seq = [(3000, .3), (6000, .301), (9000, .301), (12000, .301)]
    lr = []
    for step, value in seq:
        scheduler.step(value, epoch=step)
        a = (scheduler.stage_count, scheduler.get_lr())
        restored.step(value, epoch=step)
        assert a == (restored.stage_count, restored.get_lr())
        lr.append(dict(iteration=step, metric=value, stage=a[0], lr=a[1][0]))
    states = {}
    detector = torch.load(str(OLD / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"), map_location="cpu")["model"]
    for task in ["sgcls", "sgdet"]:
        path, state = snapshot(task)
        protocol = json.loads((r16.RUN / task / "formal/protocol.json").read_text())
        assert state["protocol"] == protocol
        assert protocol["driver_sha256"] == sha256(Path(r16.__file__))
        changed_frozen = [k for k, v in state["model"].items()
                          if not k.startswith("roi_heads.relation.") and not torch.equal(v, detector[k])]
        assert not changed_frozen, changed_frozen
        decoder = {k: v for k, v in state["model"].items() if "decoder_rnn.out_obj" in k}
        assert decoder and all(torch.isfinite(v).all() for v in decoder.values())
        gradient = json.loads((r16.RUN / task / "formal/object_gradient.json").read_text())
        assert gradient["norm"] > 0
        assert state["scheduler"]["last_epoch"] == state["iteration"]
        states[task] = dict(iteration=state["iteration"], sha256=sha256(path),
                            changed_frozen_tensors=changed_frozen, object_gradient=gradient,
                            scheduler_stage=state["scheduler"]["stage_count"],
                            source_hashes_match=all(sha256(r16.REPO / k) == v for k, v in protocol["sources"].items()))
        assert states[task]["source_hashes_match"]
    report = dict(status="complete", splits=count, split_overlap=overlap, h5_equivalence=comparison,
                  states=states, scheduler_resume=lr, image_sampler_resume_exact=True,
                  upstream_filter_non_overlap=expected["filter_non_overlap"],
                  documented_deviations=["single GPU true batch8 rather than example global12",
                    "FP32 instead of Apex", "deterministic ungrouped image shuffle",
                    "75000 maximum steps and 3000 validation period scaled by image budget",
                    "warmup remains upstream 500 optimizer steps; not exposure-scaled"])
    atomic_json(OUT / "static.json", report)
    print(json.dumps(report), flush=True)


def check_image(item):
    filename, expected = item
    try:
        with Image.open(filename) as image:
            image = image.convert("RGB")
            image.load()
            return None if image.size == expected else dict(path=filename, kind="size", actual=image.size, expected=expected)
    except Exception as error:
        return dict(path=filename, kind="decode", error=str(error))


def image_audit():
    cfg = r16.configure("sgcls", OUT / "image_scan")
    rows = {}; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in ["val", "test", "train"]:
            ds = r16.dataset(cfg, name)
            items = [(p, (m["width"], m["height"])) for p, m in zip(ds.filenames, ds.img_info)]
            failures = []
            for i, result in enumerate(pool.map(check_image, items), 1):
                if result: failures.append(result)
                if i % 1000 == 0:
                    progress = dict(split=name, checked=i, total=len(items), issues=len(failures), seconds=time.monotonic()-started)
                    atomic_json(OUT / "image_progress.json", progress)
                    print(json.dumps(progress), flush=True)
            rows[name] = dict(images=len(items), issues=failures)
            atomic_json(OUT / "images.json", dict(status="running", splits=rows))
    atomic_json(OUT / "images.json", dict(status="complete", splits=rows))


def detector_audit():
    shared_path = OLD / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth"
    shared = torch.load(str(shared_path), map_location="cpu")["model"]
    comparisons = {}
    for task, filename in [("predcls", "model_0030000.pth"), ("sgcls", "model_final.pth"),
                           ("sgdet", "model_0028000.pth")]:
        path = OLD / "checkpoints/sgg/weights/causal_motifs_sum/vg" / task / filename
        payload = torch.load(str(path), map_location="cpu")
        other = {k[7:] if k.startswith("module.") else k: v
                 for k, v in payload["model"].items()}
        differences = []
        for key, value in shared.items():
            if key not in other or value.shape != other[key].shape:
                differences.append(dict(key=key, reason="missing_or_shape"))
            elif not torch.equal(value, other[key].to(value.dtype)):
                differences.append(dict(key=key, reason="values",
                    shared_dtype=str(value.dtype), reference_dtype=str(other[key].dtype),
                    equal_after_fp16_rounding=torch.equal(value.half(), other[key].half()),
                    max_absolute_error=float((value.float()-other[key].float()).abs().max())))
        comparisons[task] = dict(path=str(path), sha256=sha256(path), tensors=len(shared),
                                exact_equal=not differences,
                                same_after_fp16_rounding=all(x.get("equal_after_fp16_rounding", False) for x in differences),
                                differences=differences)
        print(json.dumps(dict(task=task, different_tensors=len(differences))), flush=True)
        del other, payload
    atomic_json(OUT / "detector.json", dict(status="complete", shared=str(shared_path),
        shared_sha256=sha256(shared_path), comparisons=comparisons,
        interpretation="Compare detector tensors only. No TDE relation weights are transferred."))


def native_metrics(pred, gt, task, ds):
    from maskrcnn_benchmark.data.datasets.evaluation.vg import sgg_eval as s
    from maskrcnn_benchmark.data.datasets.evaluation.vg.vg_eval import evaluate_relation_of_one_image
    result = {}
    mapping = dict(eval_recall=s.SGRecall, eval_nog_recall=s.SGNoGraphConstraintRecall,
                   eval_zeroshot_recall=s.SGZeroShotRecall, eval_ng_zeroshot_recall=s.SGNGZeroShotRecall,
                   eval_pair_accuracy=s.SGPairAccuracy)
    evaluators = {k: cls(result) for k, cls in mapping.items()}
    evaluators.update(eval_mean_recall=s.SGMeanRecall(result, 51, ds.ind_to_predicates),
                      eval_ng_mean_recall=s.SGNGMeanRecall(result, 51, ds.ind_to_predicates))
    for e in evaluators.values(): e.register_container(task)
    zero = torch.load(str(r16.REPO / "maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch"))
    global_values = dict(mode=task, iou_thres=.5, zeroshot_triplet=zero.numpy(), num_rel_category=51,
                         multiple_preds=False, attribute_on=False, num_attributes=201)
    evaluate_relation_of_one_image(gt, pred.resize(gt.size), global_values, evaluators)
    evaluators["eval_mean_recall"].calculate_mean_recall(task)
    return result


def compare(a, b, align_pairs=False):
    errors = dict(boxes=float((a.bbox-b.bbox).abs().max()))
    assert a.size == b.size and torch.allclose(a.bbox, b.bbox, atol=1e-5, rtol=1e-5), errors
    for field in ["pred_labels", "pred_scores"]:
        x, y = a.get_field(field), b.get_field(field)
        assert x.shape == y.shape, (field, x.shape, y.shape)
        errors[field] = float((x-y).abs().max())
        assert torch.allclose(x.float(), y.float(), atol=1e-5, rtol=1e-5), errors
    pa, pb = a.get_field("rel_pair_idxs"), b.get_field("rel_pair_idxs")
    sa, sb = a.get_field("pred_rel_scores"), b.get_field("pred_rel_scores")
    assert pa.shape == pb.shape and sa.shape == sb.shape
    errors["pair_order_identical"] = torch.equal(pa, pb)
    if align_pairs:
        n = len(a)
        ka, kb = pa[:,0]*n+pa[:,1], pb[:,0]*n+pb[:,1]
        ia, ib = ka.argsort(), kb.argsort()
        assert torch.equal(ka[ia], kb[ib])
        errors["topk_pair_symmetric_difference"] = {str(k):len(set(ka[:k].tolist()) ^ set(kb[:k].tolist())) for k in r16.KS}
        mismatch = (pa != pb).any(1).nonzero().flatten()
        errors["first_changed_rank_one_based"] = int(mismatch[0])+1 if len(mismatch) else None
        sa, sb = sa[ia], sb[ib]
    else:
        assert torch.equal(pa, pb), errors
    errors["pred_rel_scores"] = float((sa-sb).abs().max())
    assert torch.allclose(sa, sb, atol=1e-5, rtol=1e-5), errors
    return errors


@torch.no_grad()
def native_audit(unchunked=False):
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.data.transforms import build_transforms
    from maskrcnn_benchmark.structures.image_list import to_image_list
    from maskrcnn_benchmark.structures.bounding_box import BoxList
    from reproduction_pair_audit import ChunkUnion
    from export_pysgg_vg_task import convert_prediction
    reports = {}
    for task in ["sgcls", "sgdet", "predcls"]:
        cfg = r16.configure(task, OUT / task)
        cfg.defrost(); cfg.MODEL.DEVICE="cpu"; cfg.freeze()
        stats = torch.load(str(ROOT / "data/R16_plain_motifs/upstream_statistics.pth"))
        r16.save_torch(Path(cfg.OUTPUT_DIR) / "VG_stanford_filtered_with_attribute_train_statistics.cache", stats)
        model = build_detection_model(cfg).cpu().eval()
        path, state = snapshot(task if task != "predcls" else "sgcls")
        model.load_state_dict(state["model"], strict=True)
        del state
        ds = r16.dataset(cfg, "val")
        transform = build_transforms(cfg, is_train=False)
        dense_boxes = BoxList(torch.arange(47*4).reshape(47,4).float(), (1000,1000), "xyxy")
        pairs = model.roi_heads.relation.samp_processor.prepare_test_pairs(torch.device("cpu"), [dense_boxes])[0]
        assert pairs.shape == (47*46, 2), pairs.shape
        rows = []
        for idx in ([0] if unchunked else [0, 1]):
            image, _, _ = ds[idx]
            gt = ds.get_groundtruth(idx, evaluation=True)
            tensor, target = transform(image, gt)
            images = to_image_list([tensor], size_divisible=32)
            chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
            try:
                pred = model(images, [target])[0]
            finally:
                chunk.close()
            if unchunked:
                alternate = model(images, [target])[0]
            else:
                chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 128)
                try: alternate = model(images, [target])[0]
                finally: chunk.close()
            invariance = compare(pred, alternate, align_pairs=unchunked)
            if unchunked:
                x, y = r16.recall_row(pred, gt, task), r16.recall_row(alternate, gt, task)
                for field in ["R", "class_recall", "zR"]:
                    a, b = np.asarray(x[field], dtype=float), np.asarray(y[field], dtype=float)
                    assert np.allclose(a, b, equal_nan=True, atol=1e-10, rtol=0), (field, x, y)
                invariance["all_reported_metrics_identical"] = True
                print(json.dumps(dict(task=task, unchunked_comparison=invariance)), flush=True)
            if task != "predcls":
                changed = copy.deepcopy(target)
                changed.add_field("labels", (changed.get_field("labels") % 150) + 1)
                changed.add_field("relation", torch.zeros_like(changed.get_field("relation")), is_triplet=True)
                chunk = ChunkUnion(model.roi_heads.relation.union_feature_extractor, 256)
                try: perturbed = model(images, [changed])[0]
                finally: chunk.close()
                label_leak = compare(pred, perturbed)
            else:
                assert torch.equal(pred.get_field("pred_labels"), target.get_field("labels"))
                label_leak = "GT labels intentionally supplied"
            row = r16.recall_row(pred, gt, task)
            native = native_metrics(pred, gt, task, ds)
            for k in [20,50,100]:
                j = r16.KS.index(k)
                assert abs(row["R"][j] - native[task+"_recall"][k][0]) < 1e-10
                per_class = row["class_recall"][j]
                macro = sum(v if v is not None else 0. for v in per_class)/50.
                assert abs(macro-native[task+"_mean_recall"][k]) < 1e-10
                expected = native[task+"_zeroshot_recall"][k]
                assert (not expected and row["zR"][j] is None) or abs(row["zR"][j]-expected[0]) < 1e-10
            data = convert_prediction(copy.deepcopy(pred))
            assert np.isfinite(data["pred_rel_scores"]).all()
            assert np.allclose(data["pred_boxes"], pred.bbox.numpy()/np.array([*pred.size,*pred.size]))
            pairs = pred.get_field("rel_pair_idxs")
            expected_pairs = len(pred)*(len(pred)-1)
            assert len(pairs) == expected_pairs
            scores = pred.get_field("pred_scores")
            ranking = scores[pairs].prod(1)*pred.get_field("pred_rel_scores")[:,1:].max(1)[0]
            assert torch.all(ranking[:-1] >= ranking[1:])
            rows.append(dict(image_id=Path(ds.filenames[idx]).stem, objects=len(pred), pairs=len(pairs),
                             metrics=row, native_evaluator_parity=True, union_chunk_errors=invariance,
                             input_label_invariance=label_leak, export_finite=True))
            print(json.dumps(dict(task=task, image=idx, objects=len(pred), pairs=len(pairs), status="ok")), flush=True)
        reports[task] = dict(checkpoint=str(path), diagnostic_only=True, source_iteration=json.loads((OUT / "static.json").read_text())["states"][task if task!="predcls" else "sgcls"]["iteration"],
                             predcls_note="SGCls weights used only to check PredCls execution, not performance" if task=="predcls" else None,
                             all_pairs_above_2048=True, comparison="unchunked vs 256" if unchunked else "128 vs 256", images=rows)
        atomic_json(OUT / ("native_unchunked.json" if unchunked else "native.json"), dict(status="running", tasks=reports))
        del model
    atomic_json(OUT / ("native_unchunked.json" if unchunked else "native.json"), dict(status="complete", tasks=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["static", "native", "images", "detector", "unchunked"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(16002); np.random.seed(16002); torch.manual_seed(16002)
    {"static":static_audit,"native":native_audit,"images":image_audit,"detector":detector_audit,
     "unchunked":lambda: native_audit(unchunked=True)}[args.stage]()
