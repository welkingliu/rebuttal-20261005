"""Inference-only GQA/VRD transfer audit for Experiment V.

The evaluator exports a model-family cache once and applies each learned
grounding calibrator offline. External labels are restricted to normalized
exact matches with VG-150 and mapping coverage is always reported. This is a
GT-box SGCls-aligned diagnostic, not a native external-dataset SGG benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sgg_core.data.gqa_psg_data_utils import build_gqa_loader
from sgg_core.data.shared_vg_ontology import (
    build_exact_mapping,
    load_vg_ontology,
    project_batch_to_vg,
)
from sgg_core.data.vrd_data_utils import build_vrd_loader
from sgg_core.models.official_adapter import OfficialSGGAdapter


SCHEMA_VERSION = "experiment_5_external_shared_vg_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(args):
    if args.dataset == "gqa":
        return build_gqa_loader(
            args.eval_ann, num_samples=args.eval_samples,
            vocabulary_path=args.eval_ann, image_root=args.image_root,
            include_proxy_features=False, include_raw_images=True,
        )
    return build_vrd_loader(
        args.data_root, "test", args.eval_samples,
        include_proxy_features=False, include_raw_images=True,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _align_relation_scores(output: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    score = output["pred_rel_scores"].detach().float().cpu()
    output_pairs = output.get("pred_rel_pairs")
    if output_pairs is None:
        if score.size(0) != batch["rel_pairs"].size(0):
            raise RuntimeError("Relation scores do not align with GT relation pairs")
        return score, torch.arange(score.size(0), dtype=torch.long)
    pair_to_rows = defaultdict(list)
    for row, pair in enumerate(output_pairs.detach().cpu().tolist()):
        pair_to_rows[tuple(map(int, pair))].append(row)
    rows = []
    for pair in batch["rel_pairs"].tolist():
        candidates = pair_to_rows.get(tuple(map(int, pair)), [])
        if not candidates:
            raise RuntimeError(f"Missing relation pair in live output: {pair}")
        rows.append(candidates.pop(0))
    return score[rows], torch.tensor(rows, dtype=torch.long)


def _export_cache(args, cache_dir: Path) -> dict:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("execution_mode", "live_adapter") != "live_adapter":
        raise RuntimeError("External Experiment V requires a live adapter")
    if str(manifest.get("reference_dataset", "")).lower() != "vg":
        raise RuntimeError("External shared-ontology transfer requires a VG model")
    loader = _loader(args)
    vg = load_vg_ontology(args.vg_dict)
    mapping = build_exact_mapping(loader.dataset, vg)
    device = torch.device(args.device)
    model = OfficialSGGAdapter(str(manifest_path), device=device)
    if not model.supports_mitigation:
        raise RuntimeError("Manifest does not expose forward_grounding")

    object_scores, object_targets, object_images = [], [], []
    relation_scores, relation_targets, relation_images = [], [], []
    relation_subject, relation_object = [], []
    coverage = Counter()
    skip_reasons = Counter()
    object_offset = 0
    model.eval()
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            projected, report = project_batch_to_vg(raw_batch, mapping, vg)
            coverage["source_images"] += 1
            for key in ("source_objects", "source_relations", "retained_objects", "retained_relations"):
                coverage[key] += int(report.get(key, 0))
            skip_reasons.update(report.get("skipped_relations", {}))
            if projected is None:
                coverage["skipped_images"] += 1
                if report.get("missing_image_tensor"):
                    skip_reasons["missing_image_tensor"] += 1
                continue
            batch = _move(projected, device)
            output = model.forward_grounding(batch)
            entity = output["pred_entity_scores"].detach().float().cpu()
            if entity.ndim != 2 or entity.size(0) != batch["entity_labels"].numel():
                raise RuntimeError("Object logits do not align with projected GT entities")
            relation, _ = _align_relation_scores(output, batch)
            labels = batch["entity_labels"].detach().cpu()
            predicates = batch["rel_labels"].detach().cpu()
            pairs = batch["rel_pairs"].detach().cpu()
            if relation.size(0) != predicates.numel():
                raise RuntimeError("Predicate logits do not align with projected GT relations")
            image_id = str(batch["image_id"])
            object_scores.append(entity)
            object_targets.append(labels)
            object_images.extend([image_id] * labels.numel())
            relation_scores.append(relation)
            relation_targets.append(predicates)
            relation_images.extend([image_id] * predicates.numel())
            relation_subject.extend((pairs[:, 0] + object_offset).tolist())
            relation_object.extend((pairs[:, 1] + object_offset).tolist())
            object_offset += labels.numel()
            coverage["retained_images"] += 1
            print(json.dumps({
                "dataset": args.dataset, "processed": index + 1,
                "total": len(loader.dataset), "retained_images": coverage["retained_images"],
            }), flush=True)

    if not object_scores or not relation_scores:
        raise RuntimeError("No shared-ontology external predictions were exported")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        object_scores=torch.cat(object_scores).numpy(),
        object_targets=torch.cat(object_targets).numpy(),
        object_image_ids=np.asarray(object_images, dtype=str),
        relation_scores=torch.cat(relation_scores).numpy(),
        relation_targets=torch.cat(relation_targets).numpy(),
        relation_image_ids=np.asarray(relation_images, dtype=str),
        relation_subject=np.asarray(relation_subject, dtype=np.int64),
        relation_object=np.asarray(relation_object, dtype=np.int64),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "model": model.name,
        "architecture_family": model.architecture_family,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": model.checkpoint_status["sha256"],
        "source_ontology_id": loader.dataset.ontology_id,
        "target_ontology_id": vg["ontology_id"],
        "vg_dictionary": {key: vg[key] for key in ("path", "sha256", "ontology_id")},
        "mapping": {key: value for key, value in mapping.items() if not key.endswith("_map")},
        "coverage": dict(coverage),
        "skip_reasons": dict(skip_reasons),
        "protocol": "GT-box shared-VG exact-overlap inference only",
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _linear(scores: torch.Tensor, state: dict, prefix: str) -> torch.Tensor:
    weight = state.get(f"{prefix}.weight")
    bias = state.get(f"{prefix}.bias")
    if weight is None or bias is None:
        raise KeyError(f"Mitigation state is missing {prefix}.weight/bias")
    return F.linear(scores.float(), weight.float(), bias.float())


def _micro_ci(correct: np.ndarray, image_ids: np.ndarray, seed: int, trials: int = 2000) -> list[float]:
    grouped = defaultdict(list)
    for value, image_id in zip(correct.astype(np.float64), image_ids.tolist()):
        grouped[str(image_id)].append(float(value))
    keys = sorted(grouped)
    if len(keys) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(trials)):
        sampled = rng.integers(0, len(keys), len(keys))
        rows = [value for index in sampled for value in grouped[keys[index]]]
        values.append(float(np.mean(rows)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _macro_accuracy(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    values = [
        float(np.mean(prediction[target == label] == label))
        for label in np.unique(target)
    ]
    return (float(np.mean(values)) if values else float("nan"), len(values))


def _ece(scores: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    probability = scores.softmax(dim=1)
    confidence, prediction = probability.max(dim=1)
    correct = prediction.eq(target).float()
    edges = torch.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = confidence.gt(lower) & confidence.le(upper)
        if selected.any():
            value += float(selected.float().mean() * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs())
    return value


def _metrics(object_scores: torch.Tensor, relation_scores: torch.Tensor, payload, seed: int) -> dict:
    object_target = torch.from_numpy(payload["object_targets"]).long()
    relation_target = torch.from_numpy(payload["relation_targets"]).long()
    object_prediction = object_scores.argmax(dim=1)
    relation_prediction = relation_scores.argmax(dim=1)
    subject = torch.from_numpy(payload["relation_subject"]).long()
    obj = torch.from_numpy(payload["relation_object"]).long()
    object_correct = object_prediction.eq(object_target)
    relation_correct = relation_prediction.eq(relation_target)
    endpoint_correct = object_correct[subject] & object_correct[obj]
    triplet_correct = endpoint_correct & relation_correct
    object_macro, object_classes = _macro_accuracy(
        object_prediction.numpy(), object_target.numpy()
    )
    relation_macro, relation_classes = _macro_accuracy(
        relation_prediction.numpy(), relation_target.numpy()
    )
    object_images = payload["object_image_ids"]
    relation_images = payload["relation_image_ids"]
    return {
        "object_top1": float(object_correct.float().mean()),
        "object_top1_bootstrap_95ci": _micro_ci(object_correct.numpy(), object_images, seed),
        "object_macro_accuracy": object_macro,
        "object_macro_supported_classes": object_classes,
        "object_ece_15_uncalibrated": _ece(object_scores, object_target),
        "predicate_hit_at_1": float(relation_correct.float().mean()),
        "predicate_hit_at_1_bootstrap_95ci": _micro_ci(relation_correct.numpy(), relation_images, seed + 1),
        "predicate_macro_accuracy": relation_macro,
        "predicate_macro_supported_classes": relation_classes,
        "triplet_hit_at_1": float(triplet_correct.float().mean()),
        "triplet_hit_at_1_bootstrap_95ci": _micro_ci(triplet_correct.numpy(), relation_images, seed + 2),
        "endpoint_identity_disagreement_rate": float((~endpoint_correct).float().mean()),
        "endpoint_identity_disagreement_bootstrap_95ci": _micro_ci((~endpoint_correct).numpy(), relation_images, seed + 3),
        "objects": int(object_target.numel()),
        "relations": int(relation_target.numel()),
    }


def _paired_delta_ci(before: np.ndarray, after: np.ndarray, image_ids: np.ndarray, seed: int, trials: int = 2000) -> list[float]:
    delta = after.astype(np.float64) - before.astype(np.float64)
    return _micro_ci(delta, image_ids, seed, trials)


def _evaluate_cache(args, cache_dir: Path, metadata: dict) -> dict:
    with np.load(cache_dir / "predictions.npz", allow_pickle=False) as stored:
        payload = {key: np.asarray(stored[key]) for key in stored.files}
    base_object = torch.from_numpy(payload["object_scores"]).float()
    base_relation = torch.from_numpy(payload["relation_scores"]).float()
    evaluated_object, evaluated_relation = base_object, base_relation
    state_metadata = None
    if args.state:
        state_path = Path(args.state).expanduser().resolve()
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        if checkpoint.get("base_checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise RuntimeError("Mitigation state/base-checkpoint SHA256 mismatch")
        state = checkpoint["grounding_state_dict"]
        evaluated_object = _linear(base_object, state, "entity_calibrator")
        evaluated_relation = _linear(base_relation, state, "relation_calibrator")
        state_metadata = {
            "path": str(state_path), "sha256": _sha256(state_path),
            "training_mode": checkpoint.get("training_args", {}).get("training_mode"),
            "seed": checkpoint.get("training_args", {}).get("seed"),
            "selected_epoch": checkpoint.get("selected_epoch"),
        }
    base = _metrics(base_object, base_relation, payload, args.seed)
    evaluated = _metrics(evaluated_object, evaluated_relation, payload, args.seed)
    object_target = torch.from_numpy(payload["object_targets"]).long()
    relation_target = torch.from_numpy(payload["relation_targets"]).long()
    subject = torch.from_numpy(payload["relation_subject"]).long()
    obj = torch.from_numpy(payload["relation_object"]).long()
    base_object_correct = base_object.argmax(1).eq(object_target)
    eval_object_correct = evaluated_object.argmax(1).eq(object_target)
    base_relation_correct = base_relation.argmax(1).eq(relation_target)
    eval_relation_correct = evaluated_relation.argmax(1).eq(relation_target)
    base_triplet = base_relation_correct & base_object_correct[subject] & base_object_correct[obj]
    eval_triplet = eval_relation_correct & eval_object_correct[subject] & eval_object_correct[obj]
    deltas = {
        "object_top1": evaluated["object_top1"] - base["object_top1"],
        "object_top1_bootstrap_95ci": _paired_delta_ci(
            base_object_correct.numpy(), eval_object_correct.numpy(),
            payload["object_image_ids"], args.seed + 10,
        ),
        "predicate_hit_at_1": evaluated["predicate_hit_at_1"] - base["predicate_hit_at_1"],
        "predicate_hit_at_1_bootstrap_95ci": _paired_delta_ci(
            base_relation_correct.numpy(), eval_relation_correct.numpy(),
            payload["relation_image_ids"], args.seed + 11,
        ),
        "triplet_hit_at_1": evaluated["triplet_hit_at_1"] - base["triplet_hit_at_1"],
        "triplet_hit_at_1_bootstrap_95ci": _paired_delta_ci(
            base_triplet.numpy(), eval_triplet.numpy(),
            payload["relation_image_ids"], args.seed + 12,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "V_external_inference_only",
        "paper_evidence_tier": "shared_vg_gt_box_external",
        "claim_scope": (
            "Inference-only transfer on normalized exact VG label overlap. "
            "This is SGCls-aligned and is not native GQA/VRD SGDet."
        ),
        "cache": str(cache_dir),
        "cache_metadata": metadata,
        "mitigation_state": state_metadata,
        "base": base,
        "evaluated": evaluated,
        "paired_change_from_base": deltas,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True, choices=("gqa", "vrd"))
    parser.add_argument("--data_root")
    parser.add_argument("--eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--vg_dict", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state")
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    metadata_path = cache_dir / "metadata.json"
    prediction_path = cache_dir / "predictions.npz"
    if not metadata_path.is_file() or not prediction_path.is_file():
        metadata = _export_cache(args, cache_dir)
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest_path = Path(args.manifest).expanduser().resolve()
        if metadata["manifest_sha256"] != _sha256(manifest_path):
            raise RuntimeError("Cache/manifest SHA256 mismatch")
        if metadata["dataset"] != args.dataset:
            raise RuntimeError("Cache/dataset mismatch")
    if args.export_only:
        print(f"External cache ready: {cache_dir}")
        return
    summary = _evaluate_cache(args, cache_dir, metadata)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Experiment V external complete: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
