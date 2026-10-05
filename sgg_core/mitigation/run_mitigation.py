"""Fine-tune a live official SGG adapter with an object-grounding objective.

Model selection is performed on validation data. VG-150 additionally receives
one untouched split-2 evaluation after training; the test result is never used
by the optimizer or the mitigation acceptance rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sgg_core.audits.graph_audit import GraphLevelAudit
from sgg_core.audits.error_decomposition import GroundingErrorDecompositionAudit
from sgg_core.audits.live_sgcls_validation import LiveSGClsValidationAudit
from sgg_core.audits.pair_audit import PairLevelAudit, VisualPerturbation
from sgg_core.audits.standard_sgg_eval import StandardSGGAudit
from sgg_core.mitigation.grounding_regularizer import GroundingDependencyRegularizer
from sgg_core.models.official_adapter import OfficialSGGAdapter
from sgg_core.data.data_utils import build_vg_test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding-aware SGG mitigation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True, choices=["vg"])
    parser.add_argument("--data_root")
    parser.add_argument("--train_ann")
    parser.add_argument("--eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--panoptic_root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Maximum number of mitigation epochs.",
    )
    parser.add_argument(
        "--minimum_epochs", type=int, default=3,
        help="Do not apply validation early stopping before this epoch.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=1,
        help="Stop after this many non-improving validation epochs.",
    )
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument(
        "--test_samples", type=int, default=1_000_000_000,
        help="VG uses the complete untouched split-2 test by default.",
    )
    parser.add_argument(
        "--skip_test", action="store_true",
        help="Run the validation gate without evaluating the untouched test split.",
    )
    parser.add_argument(
        "--before_test_cache",
        help=(
            "Optional family-level cache for the unchanged pretrained test "
            "evaluation. Formal matrix runs share this baseline across modes "
            "and seeds while always recomputing the post-training test result."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mild_noise", type=float, default=0.05)
    parser.add_argument(
        "--training_mode", choices=("supervised_control", "grounding"),
        default="grounding",
        help=(
            "supervised_control uses the same relation/object supervision and "
            "training budget without the proposed regularizers."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recall_ks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--max_mr_drop", type=float, default=0.01)
    parser.add_argument(
        "--stop_on_mr_drop", action="store_true",
        help=(
            "Stop after validation exceeds max_mr_drop and restore the best "
            "task-preserving epoch."
        ),
    )
    parser.add_argument("--minimum_object_top1_gain", type=float, default=0.005)
    parser.add_argument("--max_object_ece_increase", type=float, default=0.01)
    parser.add_argument("--minimum_validation_objects", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--object_weight", type=float, default=1.0)
    parser.add_argument("--object_consistency_weight", type=float, default=0.25)
    parser.add_argument("--object_calibration_weight", type=float, default=0.05)
    parser.add_argument(
        "--object_focal_gamma", type=float, default=0.0,
        help="Focal exponent for low-confidence object supervision.",
    )
    parser.add_argument(
        "--object_margin_weight", type=float, default=0.0,
        help="Weight for hard-object logit margin supervision.",
    )
    parser.add_argument("--object_margin", type=float, default=0.1)
    parser.add_argument(
        "--hard_object_fraction", type=float, default=1.0,
        help="Fraction of objects with the largest margin violations to optimize.",
    )
    parser.add_argument(
        "--validation_interval_images", type=int, default=0,
        help=(
            "Run checkpoint selection after this many training images inside "
            "each epoch; zero keeps epoch-only selection."
        ),
    )
    parser.add_argument(
        "--progress_interval_images", type=int, default=100,
        help=(
            "Emit a structured training progress event after this many valid "
            "training images; zero disables interval events."
        ),
    )
    parser.add_argument(
        "--task_driven_object_focus", action="store_true",
        help=(
            "Upweight objects participating in task-relevant relations while "
            "retaining supervision for every annotated relation."
        ),
    )
    parser.add_argument("--task_object_weight", type=float, default=2.0)
    parser.add_argument(
        "--task_predicate_ids", nargs="+", type=int,
        default=[
            7, 11, 14, 19, 21, 22, 24, 26, 28, 31, 32,
            35, 38, 40, 41, 43, 44, 45, 46, 48, 49,
        ],
        help="VG predicate ids treated as downstream interaction/affordance tasks.",
    )
    parser.add_argument(
        "--freeze_relation_parameters", action="store_true",
        help="Update only the object parameter group during mitigation.",
    )
    parser.add_argument(
        "--object_bias_only", action="store_true",
        help=(
            "Within the object parameter group, freeze matrix parameters and "
            "optimize only the one-dimensional class bias."
        ),
    )
    parser.add_argument(
        "--object_weight_delta_scale",
        type=float,
        default=1.0,
        help="Fixed post-selection scale for object-head non-bias deltas.",
    )
    parser.add_argument(
        "--object_bias_delta_scale",
        type=float,
        default=1.0,
        help="Fixed post-selection scale for object-head bias deltas.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--run_appendix_audits", action="store_true",
        help="Also run the expensive BRR and motif audits before/after training.",
    )
    return parser.parse_args()


def _loaders(args):
    if args.dataset == "vg":
        # The canonical VG-SGG release used by this project contains only
        # split 0 (train) and split 2 (test). Build a deterministic, disjoint
        # validation holdout from split 0 instead of assuming split 1 exists.
        train_count = int(args.train_samples)
        validation_count = int(args.eval_samples)
        pool_loader = build_vg_test_loader(
            args.data_root, train_count + validation_count, split=0
        )
        pool = pool_loader.dataset
        required = train_count + validation_count
        if len(pool) < required:
            raise RuntimeError(
                "VG train split cannot satisfy the requested disjoint "
                f"train/validation holdout: available={len(pool)} "
                f"requested={required}"
            )

        def subset_loader(start, stop):
            subset = Subset(pool, range(start, stop))
            for name in (
                "dataset_name", "ontology_id", "num_entity_classes",
                "num_predicate_classes",
            ):
                if hasattr(pool, name):
                    setattr(subset, name, getattr(pool, name))
            return DataLoader(
                subset, batch_size=1, shuffle=False,
                collate_fn=pool_loader.collate_fn, num_workers=0,
            )

        test_loader = (
            None if args.skip_test
            else build_vg_test_loader(args.data_root, args.test_samples, split=2)
        )
        return (
            subset_loader(0, train_count),
            subset_loader(train_count, required),
            test_loader,
            (
                "deterministic_disjoint_split0_holdout_gate_without_test"
                if test_loader is None
                else "deterministic_disjoint_split0_holdout_and_untouched_split2_test"
            ),
        )
    raise AssertionError("Experiment V training is registered only on VG-150")


def _move(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _deduplicate_training_relations(batch, seed):
    """Match PySGG's train-time one-predicate-per-entity-pair protocol."""
    pairs = batch.get("rel_pairs")
    labels = batch.get("rel_labels")
    if not isinstance(pairs, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return batch, 0
    if pairs.ndim != 2 or pairs.shape[1] != 2 or labels.ndim != 1:
        raise ValueError("Relation annotations must be shaped [R,2] and [R]")
    if pairs.shape[0] != labels.shape[0]:
        raise ValueError("Relation pairs and labels are not aligned")

    grouped = {}
    for index, pair in enumerate(pairs.detach().cpu().tolist()):
        grouped.setdefault((int(pair[0]), int(pair[1])), []).append(index)
    removed = int(pairs.shape[0] - len(grouped))
    if removed == 0:
        return batch, 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = []
    for indices in grouped.values():
        choice = int(torch.randint(len(indices), (1,), generator=generator).item())
        selected.append(indices[choice])
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=pairs.device)

    result = dict(batch)
    result["rel_pairs"] = pairs.index_select(0, selected_tensor)
    result["rel_labels"] = labels.index_select(
        0, selected_tensor.to(labels.device)
    )
    union_features = batch.get("union_features")
    if (
        isinstance(union_features, torch.Tensor)
        and union_features.ndim >= 1
        and union_features.shape[0] == pairs.shape[0]
    ):
        result["union_features"] = union_features.index_select(
            0, selected_tensor.to(union_features.device)
        )
    return result, removed


def _training_metadata(loader):
    seen = set()
    object_support = Counter()
    subset = getattr(loader, "dataset", None)
    pool = getattr(subset, "dataset", None)
    subset_indices = getattr(subset, "indices", None)
    if (
        pool is not None
        and subset_indices is not None
        and hasattr(pool, "annot_file")
        and hasattr(pool, "image_indices")
    ):
        annotations = pool.annot_file
        for subset_index in subset_indices:
            image_index = int(pool.image_indices[int(subset_index)])
            first_box = int(annotations["img_to_first_box"][image_index])
            last_box = int(annotations["img_to_last_box"][image_index])
            labels = annotations["labels"][first_box:last_box + 1, 0]
            object_support.update(int(value) for value in labels.tolist())

            first_rel = int(annotations["img_to_first_rel"][image_index])
            last_rel = int(annotations["img_to_last_rel"][image_index])
            if first_rel < 0 or first_rel > last_rel:
                continue
            relationships = annotations["relationships"][first_rel:last_rel + 1]
            predicates = annotations["predicates"][first_rel:last_rel + 1, 0]
            for pair, predicate in zip(relationships, predicates):
                subject = int(pair[0]) - first_box
                obj = int(pair[1]) - first_box
                predicate = int(predicate)
                if (
                    predicate > 0
                    and 0 <= subject < len(labels)
                    and 0 <= obj < len(labels)
                ):
                    seen.add(
                        (int(labels[subject]), predicate, int(labels[obj]))
                    )
        return seen, dict(object_support)

    for batch in loader:
        entities = batch.get("entity_labels")
        pairs = batch.get("rel_pairs")
        predicates = batch.get("rel_labels")
        if entities is None or pairs is None or predicates is None:
            continue
        object_support.update(int(value) for value in entities.tolist())
        for pair, predicate in zip(pairs.tolist(), predicates.tolist()):
            subject, obj = map(int, pair)
            predicate = int(predicate)
            if predicate > 0 and subject < entities.numel() and obj < entities.numel():
                seen.add((int(entities[subject]), predicate, int(entities[obj])))
    return seen, dict(object_support)


def _task_object_weights(batch, predicate_ids, task_weight):
    entities = batch.get("entity_labels")
    pairs = batch.get("rel_pairs")
    predicates = batch.get("rel_labels")
    if not all(isinstance(value, torch.Tensor) for value in (
        entities, pairs, predicates
    )):
        return None, 0
    weights = torch.ones(
        entities.numel(), dtype=torch.float32, device=entities.device
    )
    task_mask = torch.zeros_like(predicates, dtype=torch.bool)
    for predicate_id in predicate_ids:
        task_mask |= predicates == int(predicate_id)
    selected_pairs = pairs[task_mask]
    if selected_pairs.numel():
        object_indices = selected_pairs.reshape(-1).unique()
        object_indices = object_indices[
            (object_indices >= 0) & (object_indices < entities.numel())
        ]
        weights[object_indices] = float(task_weight)
        focused = int(object_indices.numel())
    else:
        focused = 0
    return weights, focused


def _evaluate(model, loader, motif_loader, recall_ks, device, seen_triplets,
              object_support, run_appendix_audits=False):
    models = {model.name: model}
    results = {
        "standard_sgg": StandardSGGAudit(
            ks=recall_ks, device=device, seen_triplets=seen_triplets,
        ).run(models, loader)[model.name],
        "grounding_error_decomposition": GroundingErrorDecompositionAudit(
            device=device, class_frequency=object_support,
        ).run(models, loader)[model.name],
    }
    if run_appendix_audits:
        results["pair_audit"] = PairLevelAudit(
            recall_ks=(1, 5, 10), primary_k=5, device=device,
        ).run(models, loader)[model.name]
        results["graph_audit"] = GraphLevelAudit(device=device).run(
            models, loader, motif_loader=motif_loader
        )[model.name]
    return results


def _evaluate_selection(model, loader, recall_ks, device, seen_triplets,
                        object_support):
    """Cache-free validation endpoints used for epoch selection."""
    del object_support
    return LiveSGClsValidationAudit(
        ks=recall_ks, device=device, seen_triplets=seen_triplets,
    ).run(model, loader)


def _nested_metric(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _finite(value):
    return bool(
        isinstance(value, (int, float, np.number)) and np.isfinite(value)
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _validation_contract(payload, minimum_objects):
    object_count = _nested_metric(
        payload, "grounding_error_decomposition", "object_identity",
        "localized_objects",
    )
    top1 = _nested_metric(
        payload, "grounding_error_decomposition", "object_identity",
        "top1_accuracy_given_localized",
    )
    ece = _nested_metric(
        payload, "grounding_error_decomposition", "object_identity", "ece_15"
    )
    sgcls_mr = _nested_metric(
        payload, "standard_sgg", "tasks", "sgcls", "metrics", "mR@50"
    )
    errors = (
        _nested_metric(payload, "grounding_error_decomposition", "errors") or []
    )
    checks = {
        "minimum_objects": (
            isinstance(object_count, (int, float))
            and object_count >= minimum_objects
        ),
        "object_top1_finite": _finite(top1),
        "object_ece_finite": _finite(ece),
        "sgcls_mR@50_finite": _finite(sgcls_mr),
        "no_live_errors": not errors,
    }
    return {
        "protocol": "live_sgcls_gt_boxes_on_disjoint_split0_holdout",
        "object_count": object_count,
        "required_object_count": minimum_objects,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _parameter_groups(model):
    raw_groups = model.grounding_parameter_groups()
    groups = {
        str(name): [parameter for parameter in values if parameter.requires_grad]
        for name, values in raw_groups.items()
    }
    groups = {name: values for name, values in groups.items() if values}
    if "object" not in groups:
        raise RuntimeError(
            "Mitigation adapter must expose a non-empty 'object' parameter group"
        )
    return groups


def _restrict_object_parameters_to_bias(model):
    object_parameters = list(
        model.grounding_parameter_groups().get("object", ())
    )
    bias_parameters = [
        parameter for parameter in object_parameters if parameter.ndim == 1
    ]
    if len(bias_parameters) != 1:
        raise RuntimeError(
            "Bias-only mitigation requires exactly one one-dimensional "
            "parameter in the object group"
        )
    bias_parameter = bias_parameters[0]
    for parameter in object_parameters:
        parameter.requires_grad_(parameter is bias_parameter)


def _parameter_snapshot(groups):
    return {
        name: [parameter.detach().cpu().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }


def _parameter_update_audit(groups, initial):
    result = {}
    for name, parameters in groups.items():
        squared = 0.0
        count = 0
        for parameter, baseline in zip(parameters, initial[name]):
            difference = parameter.detach().cpu() - baseline
            squared += float(difference.float().square().sum().item())
            count += int(difference.numel())
        result[name] = {
            "l2_delta": float(squared ** 0.5),
            "parameter_count": count,
            "updated": bool(squared > 0.0),
        }
    return {
        "groups": result,
        "object_parameters_updated": bool(
            result.get("object", {}).get("updated", False)
        ),
    }


def _acceptance(before, after, args, parameter_audit=None):
    before_contract = _validation_contract(
        before, args.minimum_validation_objects
    )
    after_contract = _validation_contract(
        after, args.minimum_validation_objects
    )
    task_checks = {}
    for task in ("sgcls", "sgdet"):
        old = _nested_metric(
            before, "standard_sgg", "tasks", task, "metrics", "mR@50"
        )
        new = _nested_metric(
            after, "standard_sgg", "tasks", task, "metrics", "mR@50"
        )
        task_checks[task] = {
            "before_mR@50": old,
            "after_mR@50": new,
            "applicable": _finite(old) and _finite(new),
            "passed": (
                new >= old - args.max_mr_drop
                if _finite(old) and _finite(new) else None
            ),
        }
    old_top1 = _nested_metric(
        before, "grounding_error_decomposition", "object_identity",
        "top1_accuracy_given_localized",
    )
    new_top1 = _nested_metric(
        after, "grounding_error_decomposition", "object_identity",
        "top1_accuracy_given_localized",
    )
    old_ece = _nested_metric(
        before, "grounding_error_decomposition", "object_identity", "ece_15"
    )
    new_ece = _nested_metric(
        after, "grounding_error_decomposition", "object_identity", "ece_15"
    )
    object_improved = (
        _finite(old_top1) and _finite(new_top1)
        and new_top1 - old_top1 >= args.minimum_object_top1_gain
    )
    calibration_preserved = (
        _finite(old_ece) and _finite(new_ece)
        and new_ece <= old_ece + args.max_object_ece_increase
    )
    applicable_standard = [
        item for item in task_checks.values() if item["applicable"]
    ]
    standard_preserved = bool(applicable_standard) and all(
        item["passed"] for item in applicable_standard
    )

    old_brr = _nested_metric(before, "pair_audit", "BRR")
    new_brr = _nested_metric(after, "pair_audit", "BRR")
    old_mar = _nested_metric(before, "graph_audit", "MAR")
    new_mar = _nested_metric(after, "graph_audit", "MAR")
    parameters_updated = (
        True if parameter_audit is None
        else bool(parameter_audit.get("object_parameters_updated"))
    )
    protocol_valid = before_contract["passed"] and after_contract["passed"]
    return {
        "selection_split": "validation",
        "validation_protocol": "live_sgcls_gt_boxes_on_disjoint_split0_holdout",
        "protocol_valid": protocol_valid,
        "before_contract": before_contract,
        "after_contract": after_contract,
        "primary_endpoint": "gt_box_object_top1_accuracy",
        "object_top1": {
            "before": old_top1,
            "after": new_top1,
            "minimum_gain": args.minimum_object_top1_gain,
            "passed": object_improved,
        },
        "object_ece_15": {
            "before": old_ece,
            "after": new_ece,
            "maximum_increase": args.max_object_ece_increase,
            "passed": calibration_preserved,
        },
        "standard_task_non_degradation": task_checks,
        "standard_preserved": standard_preserved,
        "parameter_update_audit": parameter_audit,
        "object_parameters_updated": parameters_updated,
        "secondary_diagnostics": {
            "BRR_change": (
                new_brr - old_brr if _finite(old_brr) and _finite(new_brr)
                else None
            ),
            "MAR_change": (
                new_mar - old_mar if _finite(old_mar) and _finite(new_mar)
                else None
            ),
            "not_acceptance_endpoints": True,
        },
        "passed": (
            protocol_valid
            and object_improved
            and calibration_preserved
            and standard_preserved
            and parameters_updated
        ),
    }


def _selection_key(acceptance: dict) -> tuple:
    object_top1 = acceptance["object_top1"].get("after")
    object_ece = acceptance["object_ece_15"].get("after")
    return (
        int(bool(acceptance.get("protocol_valid"))),
        int(bool(acceptance.get("object_parameters_updated"))),
        int(bool(acceptance.get("standard_preserved"))),
        int(bool(acceptance["object_ece_15"].get("passed"))),
        float(object_top1) if _finite(object_top1) else -1e30,
        -float(object_ece) if _finite(object_ece) else -1e30,
    )


def _cpu_grounding_state(model) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.grounding_state_dict().items()
    }


def main():
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not 1 <= args.minimum_epochs <= args.epochs:
        raise ValueError("--minimum_epochs must be between 1 and --epochs")
    if args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be positive")
    if args.minimum_validation_objects < 1:
        raise ValueError("--minimum_validation_objects must be positive")
    if not 0.0 < args.hard_object_fraction <= 1.0:
        raise ValueError("--hard_object_fraction must be in (0, 1]")
    if args.validation_interval_images < 0:
        raise ValueError("--validation_interval_images cannot be negative")
    if args.progress_interval_images < 0:
        raise ValueError("--progress_interval_images cannot be negative")
    if args.object_weight_delta_scale < 0:
        raise ValueError("--object_weight_delta_scale must be non-negative")
    if args.object_bias_delta_scale < 0:
        raise ValueError("--object_bias_delta_scale must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = OfficialSGGAdapter(args.manifest, device=args.device)
    if not model.supports_mitigation:
        raise RuntimeError(
            "The official adapter must expose differentiable forward_grounding(batch)"
        )
    train_loader, validation_loader, test_loader, split_protocol = _loaders(args)
    dataset = getattr(validation_loader, "dataset", None)
    if not model.supports_dataset(args.dataset, getattr(dataset, "ontology_id", None)):
        raise RuntimeError("Manifest ontology does not match the evaluation loader")

    seen_triplets, object_support = _training_metadata(train_loader)
    print(json.dumps({
        "event": "validation_start",
        "phase": "before_training",
        "images": len(validation_loader),
    }, sort_keys=True), flush=True)
    before_validation = _evaluate_selection(
        model, validation_loader, args.recall_ks, args.device,
        seen_triplets, object_support,
    )
    print(json.dumps({
        "event": "validation_complete",
        "phase": "before_training",
        "images": len(validation_loader),
    }, sort_keys=True), flush=True)
    before_contract = _validation_contract(
        before_validation, args.minimum_validation_objects
    )
    if not before_contract["passed"]:
        raise RuntimeError(
            "Live mitigation validation preflight failed: "
            + json.dumps(before_contract, sort_keys=True)
        )
    before_test = None
    if test_loader is not None:
        cache_path = (
            Path(args.before_test_cache).expanduser().resolve()
            if args.before_test_cache else None
        )
        cache_contract = {
            "schema": "experiment5_pretrained_test_cache_v1",
            "dataset": args.dataset,
            "model": model.name,
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "base_checkpoint_sha256": model.checkpoint_status["sha256"],
            "train_samples": int(args.train_samples),
            "test_samples": int(args.test_samples),
            "recall_ks": [int(value) for value in args.recall_ks],
            "run_appendix_audits": bool(args.run_appendix_audits),
        }
        if cache_path is not None and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("contract") != cache_contract:
                raise RuntimeError(
                    f"Experiment V pretrained-test cache contract mismatch: "
                    f"{cache_path}"
                )
            before_test = cached["evaluation"]
            print(json.dumps({
                "event": "reuse_pretrained_test_cache",
                "path": str(cache_path),
            }, sort_keys=True))
        else:
            before_test = _evaluate(
                model, test_loader, train_loader, args.recall_ks, args.device,
                seen_triplets, object_support, args.run_appendix_audits,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(
                    cache_path.name + f".{os.getpid()}.tmp"
                )
                temporary.write_text(
                    json.dumps(
                        {
                            "contract": cache_contract,
                            "evaluation": before_test,
                        },
                        indent=2,
                        default=_json_default,
                    ) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                print(json.dumps({
                    "event": "write_pretrained_test_cache",
                    "path": str(cache_path),
                }, sort_keys=True))
    initial_grounding_state = _cpu_grounding_state(model)
    if args.freeze_relation_parameters:
        for name, group_parameters in model.grounding_parameter_groups().items():
            if str(name) != "object":
                for parameter in group_parameters:
                    parameter.requires_grad_(False)
    if args.object_bias_only:
        _restrict_object_parameters_to_bias(model)
    parameter_groups = _parameter_groups(model)
    initial_parameters = _parameter_snapshot(parameter_groups)
    parameters = []
    seen_parameter_ids = set()
    for group_parameters in parameter_groups.values():
        for parameter in group_parameters:
            if id(parameter) not in seen_parameter_ids:
                parameters.append(parameter)
                seen_parameter_ids.add(id(parameter))
    if not parameters:
        raise RuntimeError("No trainable grounding parameters were exposed by the adapter")
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    grounding_mode = args.training_mode == "grounding"
    objective = GroundingDependencyRegularizer(
        mild_weight=0.5 if grounding_mode else 0.0,
        dependency_weight=0.5 if grounding_mode else 0.0,
        uncertainty_weight=0.5 if grounding_mode else 0.0,
        object_weight=args.object_weight,
        object_consistency_weight=(
            args.object_consistency_weight if grounding_mode else 0.0
        ),
        object_calibration_weight=(
            args.object_calibration_weight if grounding_mode else 0.0
        ),
        object_focal_gamma=args.object_focal_gamma,
        object_margin_weight=args.object_margin_weight,
        object_margin=args.object_margin,
        hard_object_fraction=args.hard_object_fraction,
        detach_clean_consistency_target=True,
    )
    perturb = VisualPerturbation(noise_std=1.0)
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    selection_history = []
    best_state = None
    best_epoch = None
    best_image_step = None
    best_key = None
    epochs_without_improvement = 0
    stopped_early = False
    stopping_reason = "maximum_epochs_reached"

    def consider_selection(epoch_number, image_step, count_non_improvement):
        nonlocal best_state, best_epoch, best_image_step, best_key
        nonlocal epochs_without_improvement
        model.eval()
        selection_evaluation = _evaluate_selection(
            model, validation_loader, args.recall_ks, args.device,
            seen_triplets, object_support,
        )
        selection_parameter_audit = _parameter_update_audit(
            parameter_groups, initial_parameters
        )
        selection_acceptance = _acceptance(
            before_validation, selection_evaluation, args,
            parameter_audit=selection_parameter_audit,
        )
        candidate_key = _selection_key(selection_acceptance)
        improved = best_key is None or candidate_key > best_key
        if improved:
            best_key = candidate_key
            best_epoch = epoch_number
            best_image_step = image_step
            best_state = _cpu_grounding_state(model)
            epochs_without_improvement = 0
        elif count_non_improvement:
            epochs_without_improvement += 1
        selection_history.append({
            "epoch": epoch_number,
            "image_step": image_step,
            "global_image_step": (
                (epoch_number - 1) * int(args.train_samples) + image_step
            ),
            "selection_key": list(candidate_key),
            "acceptance": selection_acceptance,
            "improved": improved,
            "counts_for_early_stopping": count_non_improvement,
            "epochs_without_improvement": epochs_without_improvement,
        })
        print(json.dumps({
            "event": "validation_selection",
            "epoch": epoch_number,
            "image_step": image_step,
            "improved": improved,
            "selected_epoch": best_epoch,
            "selected_image_step": best_image_step,
            "acceptance_passed": selection_acceptance["passed"],
            "object_top1_before": (
                selection_acceptance["object_top1"]["before"]
            ),
            "object_top1_after": (
                selection_acceptance["object_top1"]["after"]
            ),
            "object_top1_gain": (
                selection_acceptance["object_top1"]["after"]
                - selection_acceptance["object_top1"]["before"]
            ),
            "object_ece_after": (
                selection_acceptance["object_ece_15"]["after"]
            ),
            "sgcls_mR50_before": (
                selection_acceptance["standard_task_non_degradation"]["sgcls"][
                    "before_mR@50"
                ]
            ),
            "sgcls_mR50_after": (
                selection_acceptance["standard_task_non_degradation"]["sgcls"][
                    "after_mR@50"
                ]
            ),
        }, sort_keys=True), flush=True)
        return selection_acceptance

    for epoch in range(args.epochs):
        model.train(True)
        epoch_started_at = time.monotonic()
        sums = {}
        steps = 0
        optimizer_steps = 0
        object_gradient_steps = 0
        accumulation_count = 0
        duplicate_relation_annotations_removed = 0
        task_focused_objects = 0
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({
            "event": "training_start",
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "total_images": len(train_loader),
        }, sort_keys=True), flush=True)
        for step, raw_batch in enumerate(train_loader):
            raw_batch, removed = _deduplicate_training_relations(
                raw_batch, args.seed + epoch * 100000 + step
            )
            duplicate_relation_annotations_removed += removed
            batch = _move(raw_batch, args.device)
            if batch.get("rel_labels") is None or batch["rel_labels"].numel() == 0:
                continue
            object_sample_weights = None
            if args.task_driven_object_focus:
                object_sample_weights, focused = _task_object_weights(
                    batch, args.task_predicate_ids, args.task_object_weight
                )
                task_focused_objects += focused
            mild = perturb.inject_visual_noise(
                batch, strength=args.mild_noise, seed=args.seed + epoch * 100000 + step
            )
            ablated = perturb.attenuate_union_features(batch, strength=1.0)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                clean_output = model.forward_grounding(batch)
                clean_logits = clean_output["pred_rel_scores"]
                if grounding_mode:
                    mild_logits = model.forward_grounding(mild)["pred_rel_scores"]
                    ablated_logits = model.forward_grounding(ablated)["pred_rel_scores"]
                else:
                    mild_logits = clean_logits
                    ablated_logits = clean_logits
                losses = objective(
                    clean_logits, mild_logits, ablated_logits, batch["rel_labels"],
                    object_logits=clean_output["pred_entity_scores"],
                    object_targets=batch["entity_labels"],
                    mask_object_logits=clean_output.get("mask_entity_scores"),
                    object_sample_weights=object_sample_weights,
                )
            scaler.scale(
                losses["loss"] / args.gradient_accumulation_steps
            ).backward()
            if any(
                parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad.detach()).item())
                for parameter in parameter_groups["object"]
            ):
                object_gradient_steps += 1
            accumulation_count += 1
            if accumulation_count == args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                optimizer_steps += 1
            steps += 1
            for key, value in losses.items():
                if key not in {"num_relations", "num_objects"}:
                    sums[key] = sums.get(key, 0.0) + float(value.item())
            if (
                args.progress_interval_images > 0
                and (
                    steps == 1
                    or steps % args.progress_interval_images == 0
                    or steps == len(train_loader)
                )
            ):
                elapsed = max(time.monotonic() - epoch_started_at, 1e-9)
                images_per_second = steps / elapsed
                remaining = max(len(train_loader) - steps, 0)
                print(json.dumps({
                    "event": "training_progress",
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "image_step": steps,
                    "total_images": len(train_loader),
                    "percent": round(100.0 * steps / len(train_loader), 2),
                    "optimizer_steps": optimizer_steps,
                    "images_per_second": images_per_second,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (
                        remaining / images_per_second
                        if images_per_second > 0.0 else None
                    ),
                    "mean_loss": sums.get("loss", 0.0) / steps,
                }, sort_keys=True), flush=True)
            if (
                args.validation_interval_images > 0
                and steps % args.validation_interval_images == 0
                and accumulation_count == 0
                and steps < len(train_loader)
            ):
                consider_selection(
                    epoch + 1, steps, count_non_improvement=False
                )
                model.train(True)
        if accumulation_count:
            correction = args.gradient_accumulation_steps / accumulation_count
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if not steps:
            raise RuntimeError("No valid relation batches were available for mitigation")
        epoch_record = {
            "epoch": epoch + 1,
            "image_steps": steps,
            "optimizer_steps": optimizer_steps,
            "object_gradient_steps": object_gradient_steps,
            "image_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "amp": amp_enabled,
            "training_mode": args.training_mode,
            "consistency_gradient": "clean_target_detached_mild_branch_trainable",
            "relation_pair_policy": "one_seeded_predicate_per_entity_pair",
            "duplicate_relation_annotations_removed": (
                duplicate_relation_annotations_removed
            ),
            "task_driven_object_focus": args.task_driven_object_focus,
            "task_predicate_ids": args.task_predicate_ids,
            "task_object_weight": args.task_object_weight,
            "task_focused_objects": task_focused_objects,
            "freeze_relation_parameters": args.freeze_relation_parameters,
            "object_bias_only": args.object_bias_only,
            "optimized_parameter_groups": sorted(parameter_groups),
        }
        epoch_record.update({key: value / steps for key, value in sums.items()})
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))

        selection_acceptance = consider_selection(
            epoch + 1, steps, count_non_improvement=True
        )
        mr_boundary_exceeded = not bool(
            selection_acceptance.get("standard_preserved")
        )
        if (
            args.stop_on_mr_drop
            and mr_boundary_exceeded
            and epoch + 1 >= args.minimum_epochs
            and epoch + 1 < args.epochs
        ):
            stopped_early = True
            stopping_reason = "validation_mR_drop_boundary_exceeded"
            print(json.dumps({
                "event": "early_stopping",
                "epoch": epoch + 1,
                "selected_epoch": best_epoch,
                "selected_image_step": best_image_step,
                "max_mr_drop": args.max_mr_drop,
                "reason": stopping_reason,
            }, sort_keys=True))
            break
        if (
            epoch + 1 >= args.minimum_epochs
            and epochs_without_improvement >= args.early_stopping_patience
            and epoch + 1 < args.epochs
        ):
            stopped_early = True
            stopping_reason = (
                "validation_selection_key_not_improved_for_"
                f"{epochs_without_improvement}_epoch"
            )
            print(json.dumps({
                "event": "early_stopping",
                "epoch": epoch + 1,
                "selected_epoch": best_epoch,
                "selected_image_step": best_image_step,
                "minimum_epochs": args.minimum_epochs,
                "patience": args.early_stopping_patience,
                "reason": stopping_reason,
            }, sort_keys=True))
            break

    if best_state is None or best_epoch is None or best_image_step is None:
        raise RuntimeError("Mitigation did not produce a selectable epoch")
    for key, selected_value in tuple(best_state.items()):
        if not key.startswith("entity_calibrator."):
            continue
        initial_value = initial_grounding_state.get(key)
        if initial_value is None:
            continue
        scale = (
            args.object_bias_delta_scale
            if key.endswith(".bias")
            else args.object_weight_delta_scale
        )
        best_state[key] = (
            initial_value + float(scale) * (selected_value - initial_value)
        )
    print(json.dumps({
        "event": "fixed_object_delta_scaling_complete",
        "object_weight_delta_scale": args.object_weight_delta_scale,
        "object_bias_delta_scale": args.object_bias_delta_scale,
    }, sort_keys=True), flush=True)
    model.load_grounding_state_dict(best_state)

    model.eval()
    after_validation = _evaluate_selection(
        model, validation_loader, args.recall_ks, args.device,
        seen_triplets, object_support,
    )
    after_test = (
        _evaluate(
            model, test_loader, train_loader, args.recall_ks, args.device,
            seen_triplets, object_support, args.run_appendix_audits,
        ) if test_loader is not None else None
    )
    parameter_update_audit = _parameter_update_audit(
        parameter_groups, initial_parameters
    )
    acceptance = _acceptance(
        before_validation, after_validation, args,
        parameter_audit=parameter_update_audit,
    )
    checkpoint_path = output_dir / "mitigated_state_dict.pth"
    torch.save({
        "grounding_state_dict": model.grounding_state_dict(),
        "state_scope": "trainable_grounding_parameters_only",
        "base_checkpoint_sha256": model.checkpoint_status["sha256"],
        "manifest": model.manifest,
        "training_args": vars(args),
        "selected_epoch": best_epoch,
        "selected_image_step": best_image_step,
        "object_delta_scaling": {
            "weight": args.object_weight_delta_scale,
            "bias": args.object_bias_delta_scale,
            "selection": "fixed_before_multi_seed_experiment",
        },
        "parameter_update_audit": parameter_update_audit,
    }, checkpoint_path)
    with open(output_dir / "mitigation_results.json", "w", encoding="utf-8") as handle:
        json.dump({
            "experiment": "V_grounding_dependency_mitigation",
            "dataset": args.dataset,
            "model": model.name,
            "architecture_family": model.architecture_family,
            "training_mode": args.training_mode,
            "seed": args.seed,
            "split_protocol": split_protocol,
            "before_validation": before_validation,
            "after_validation": after_validation,
            "before_test": before_test,
            "after_test": after_test,
            "acceptance": acceptance,
            "parameter_update_audit": parameter_update_audit,
            "history": history,
            "selection_history": selection_history,
            "selected_epoch": best_epoch,
            "selected_image_step": best_image_step,
            "object_delta_scaling": {
                "weight": args.object_weight_delta_scale,
                "bias": args.object_bias_delta_scale,
                "selection": "fixed_before_multi_seed_experiment",
            },
            "epochs_completed": len(history),
            "early_stopping": {
                "enabled": True,
                "maximum_epochs": args.epochs,
                "minimum_epochs": args.minimum_epochs,
                "patience": args.early_stopping_patience,
                "stopped_early": stopped_early,
                "reason": stopping_reason,
            },
            "checkpoint": str(checkpoint_path),
            "claim_scope": (
                "A cache-free GT-box live SGCls holdout selects the mitigation; "
                + (
                    "VG split-2 is an untouched within-dataset test. "
                    if test_loader is not None
                    else "the gate does not inspect VG split-2. "
                )
                + "OOD requires a separate ontology-compatible run."
            ),
        }, handle, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
