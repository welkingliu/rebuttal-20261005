"""Experiment IV: standard SGG benchmark of verified official checkpoints.

This is the main paper experiment. It accepts only official adapter manifests,
never local architecture imitations or randomly initialized substitutes. All
outputs are machine-readable JSON; plotting belongs in the paper repository,
not in the experimental evidence package. Expensive pair and motif
interventions belong to Experiments II and III and are intentionally not part
of the default Experiment-IV steps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from sgg_core.audits.error_decomposition import GroundingErrorDecompositionAudit
from sgg_core.audits.evidence_chain import EvidenceChainAnalyzer
from sgg_core.audits.feature_audit import FeatureLevelAudit
from sgg_core.audits.graph_audit import GraphLevelAudit
from sgg_core.audits.pair_audit import PairLevelAudit
from sgg_core.audits.perturbation_sweep import PerturbationSweepAudit
from sgg_core.audits.physical_consistency import PhysicalConsistencyAudit
from sgg_core.audits.standard_sgg_eval import StandardSGGAudit
from sgg_core.models.official_adapter import OfficialSGGAdapter
from sgg_core.models.panel import load_model_panel, panel_summary
from sgg_core.protocol import PAIR_HIT_KS, RECALL_KS, build_loaders, model_provenance, seed_everything, write_json
from sgg_core.submission_protocol import (
    ALL_DATASETS,
    EXPERIMENT_4_STEPS,
    GLOBAL_MODEL_FAMILY_TARGET,
    STANDARD_BENCHMARK_DATASETS,
)


DATASETS = ALL_DATASETS
DEFAULT_STEPS = ("standard", "feature", "pair", "graph", "grounding")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS,
        default=list(STANDARD_BENCHMARK_DATASETS),
    )
    parser.add_argument("--vg_root")
    parser.add_argument("--oi_root")
    parser.add_argument("--gqa_train_ann")
    parser.add_argument("--gqa_eval_ann")
    parser.add_argument("--gqa_image_root")
    parser.add_argument("--psg_train_ann")
    parser.add_argument("--psg_eval_ann")
    parser.add_argument("--psg_image_root")
    parser.add_argument("--psg_panoptic_root")
    parser.add_argument("--vrd_root")
    parser.add_argument("--official_manifest", action="append", default=[])
    parser.add_argument("--manifest_dir")
    parser.add_argument("--model_panel", default=str(Path(__file__).parents[1] / "models" / "model_panel.json"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--seen_triplets_manifest",
        help="Full-training-split triplets produced by sgg_core.tools.build_seen_triplets.",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=DEFAULT_STEPS,
        default=list(EXPERIMENT_4_STEPS),
    )
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument(
        "--minimum_model_families", type=int,
        default=GLOBAL_MODEL_FAMILY_TARGET,
    )
    parser.add_argument("--minimum_models_per_dataset", type=int, default=2)
    parser.add_argument("--allow_unlisted_family", action="store_true")
    parser.add_argument("--allow_incomplete_audits", action="store_true")
    parser.add_argument("--recall_ks", nargs="+", type=int, default=list(RECALL_KS))
    parser.add_argument("--sgg_tasks", nargs="+", choices=("predcls", "sgcls", "sgdet"),
                        default=["predcls", "sgcls", "sgdet"])
    parser.add_argument("--sgg_iou", type=float, default=0.5)
    parser.add_argument("--primary_k", type=int, default=5)
    parser.add_argument("--minimum_relations", type=int, default=200)
    parser.add_argument("--minimum_clean_recall", type=float, default=0.01)
    parser.add_argument("--minimum_pvr_checked", type=int, default=100)
    parser.add_argument("--perturbation_levels", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--perturbation_seeds", nargs="+", type=int,
                        default=[17, 29, 43])
    parser.add_argument("--top_k_motifs", type=int, default=20)
    parser.add_argument("--minimum_motif_train_support", type=int, default=20)
    parser.add_argument("--minimum_motif_eval_support", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _manifest_paths(args) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in args.official_manifest]
    if args.manifest_dir:
        paths.extend(sorted(Path(args.manifest_dir).expanduser().resolve().glob("*.json")))
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise ValueError("Provide --official_manifest or --manifest_dir")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official manifests not found: {missing}")
    return unique


def _load_models(args, panel: dict) -> dict:
    listed = {item["family"] for item in panel["models"]}
    models = {}
    for path in _manifest_paths(args):
        model = OfficialSGGAdapter(str(path), device=args.device)
        if model.name in models:
            raise ValueError(f"Duplicate model run name: {model.name}")
        if model.architecture_family not in listed and not args.allow_unlisted_family:
            raise ValueError(
                f"Manifest family {model.architecture_family!r} is absent from the survey panel"
            )
        models[model.name] = model
    families = {model.architecture_family for model in models.values()}
    if len(families) < args.minimum_model_families:
        raise RuntimeError(
            "Formal survey run requires distinct official model families, not seed variants: "
            f"required={args.minimum_model_families}, loaded={len(families)}, "
            f"families={sorted(families)}"
        )
    return models


def _loader_args(args, dataset: str, *, require_images: bool = True) -> dict:
    if dataset == "vg":
        values = {"data_root": args.vg_root}
    elif dataset == "oi":
        values = {"data_root": args.oi_root}
    elif dataset == "gqa":
        values = {
            "data_root": str(Path(args.gqa_train_ann or ".").parent),
            "train_ann": args.gqa_train_ann,
            "eval_ann": args.gqa_eval_ann,
            "image_root": args.gqa_image_root,
        }
    elif dataset == "psg":
        values = {
            "data_root": str(Path(args.psg_train_ann or ".").parent),
            "train_ann": args.psg_train_ann,
            "eval_ann": args.psg_eval_ann,
            "image_root": args.psg_image_root,
            "panoptic_root": args.psg_panoptic_root,
        }
    else:
        values = {"data_root": args.vrd_root}
    optional = {"panoptic_root"} if dataset == "psg" else set()
    if not require_images and dataset in {"gqa", "psg"}:
        optional.add("image_root")
    missing = [
        key for key, value in values.items()
        if value is None and key not in optional
    ]
    if missing:
        raise ValueError(f"Dataset {dataset} is missing path arguments: {missing}")
    return values


def _seen_triplets(loader) -> set[tuple[int, int, int]]:
    seen = set()
    for batch in loader:
        labels = batch.get("entity_labels")
        pairs = batch.get("rel_pairs")
        predicates = batch.get("rel_labels")
        if labels is None or pairs is None or predicates is None:
            continue
        for pair, predicate in zip(pairs.tolist(), predicates.tolist()):
            subject, obj = map(int, pair)
            predicate = int(predicate)
            if predicate > 0 and 0 <= subject < labels.numel() and 0 <= obj < labels.numel():
                seen.add((int(labels[subject]), predicate, int(labels[obj])))
    return seen


def _formal_seen_triplets(args, dataset: str, train_loader) -> tuple[set, dict]:
    if not args.seen_triplets_manifest:
        if args.allow_incomplete_audits:
            values = _seen_triplets(train_loader)
            return values, {
                "status": "diagnostic_subset_only",
                "num_unique_triplets": len(values),
            }
        raise ValueError(
            "Formal zero-shot evaluation requires --seen_triplets_manifest built "
            "from the complete training annotation split"
        )
    path = Path(args.seen_triplets_manifest).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if dataset not in payload:
        raise KeyError(f"Seen-triplet manifest has no dataset={dataset}: {path}")
    metadata = payload.get("_metadata", {}).get(dataset, {})
    ontology_id = getattr(train_loader.dataset, "ontology_id", None)
    if metadata.get("ontology_id") != ontology_id:
        raise RuntimeError(
            f"Seen-triplet ontology mismatch for {dataset}: "
            f"{metadata.get('ontology_id')} != {ontology_id}"
        )
    values = {tuple(map(int, row)) for row in payload[dataset]}
    return values, {
        **metadata,
        "status": "full_training_split",
        "path": str(path),
    }


def _compatible_models(models: dict, dataset: str, loader, minimum: int) -> dict:
    ontology_id = getattr(loader.dataset, "ontology_id", None)
    active = {
        name: model for name, model in models.items()
        if model.supports_dataset(dataset, ontology_id)
    }
    active_families = {model.architecture_family for model in active.values()}
    if len(active_families) < minimum:
        rejected = sorted(set(models) - set(active))
        raise RuntimeError(
            f"Dataset {dataset} requires {minimum} ontology-compatible official model "
            f"families; families={sorted(active_families)}, runs={list(active)}, "
            f"rejected={rejected}, ontology_id={ontology_id}"
        )
    return active


def _assert_complete(dataset: str, results: dict, model_names) -> None:
    failures = []
    requirements = {
        "standard_sgg": {"ok"},
        "feature_audit": {"ok"},
        "pair_audit": {"ok"},
        "perturbation_sweep": {"ok"},
        "graph_audit": {"ok"},
        "grounding_error_decomposition": {
            "ok", "partial", "sgdet_not_supported",
        },
    }
    for audit_name, accepted in requirements.items():
        if audit_name not in results:
            continue
        for name in model_names:
            status = results[audit_name].get(name, {}).get("status")
            if status not in accepted:
                failures.append(f"{name}/{audit_name}={status}")
    if failures:
        raise RuntimeError(f"Incomplete formal audit on {dataset}: {', '.join(failures[:20])}")


def _run_dataset(args, dataset: str, models: dict, output_dir: Path) -> dict:
    declared_models = [
        model for model in models.values()
        if dataset.lower() in {
            str(name).lower()
            for name in model.manifest["supported_datasets"]
        }
    ]
    cache_only = bool(declared_models) and all(
        model.execution_mode == "prediction_cache"
        for model in declared_models
    )
    paths = _loader_args(args, dataset, require_images=not cache_only)
    train_loader, eval_loader = build_loaders(
        dataset=dataset,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        include_proxy_features=bool(
            {"feature", "pair", "graph"}.intersection(args.steps)
        ),
        include_raw_images=not cache_only,
        **paths,
    )
    active = _compatible_models(
        models, dataset, eval_loader, args.minimum_models_per_dataset
    )
    results = {}
    seen_triplets, seen_metadata = _formal_seen_triplets(
        args, dataset, train_loader
    )
    if "standard" in args.steps:
        results["standard_sgg"] = StandardSGGAudit(
            ks=args.recall_ks,
            tasks=args.sgg_tasks,
            iou_threshold=args.sgg_iou,
            seen_triplets=seen_triplets,
            device=args.device,
        ).run(active, eval_loader)
        results["reproduction_validation"] = {
            name: model.validate_reproduction(results["standard_sgg"][name], dataset)
            for name, model in active.items()
        }
        failed = [
            name for name, value in results["reproduction_validation"].items()
            if value.get("status") not in {
                "pass", "pass_with_protocol_qualification", "not_applicable",
            }
        ]
        if failed and not args.allow_incomplete_audits:
            # Preserve the expensive full-split metrics before enforcing the
            # reference gate so a failed reproduction remains auditable.
            results["model_provenance"] = model_provenance(active)
            results["dataset_metadata"] = {
                "ontology_id": getattr(eval_loader.dataset, "ontology_id", None),
                "train_images": len(train_loader.dataset),
                "eval_images": len(eval_loader.dataset),
                "seen_triplets": len(seen_triplets),
                "seen_triplet_provenance": seen_metadata,
                "active_models": list(active),
                "active_families": sorted({
                    model.architecture_family for model in active.values()
                }),
            }
            results["formal_audit_status"] = {
                "status": "failed_reference_reproduction",
                "failed_models": failed,
            }
            write_json(output_dir / dataset / "results.json", results)
            raise RuntimeError(f"Reference-metric reproduction failed on {dataset}: {failed}")
    if "feature" in args.steps:
        results["feature_audit"] = FeatureLevelAudit(device=args.device).run(active, eval_loader)
    if "pair" in args.steps:
        results["pair_audit"] = PairLevelAudit(
            recall_ks=PAIR_HIT_KS,
            primary_k=args.primary_k,
            min_relations=args.minimum_relations,
            min_clean_recall=args.minimum_clean_recall,
            device=args.device,
        ).run(active, eval_loader)
        results["perturbation_sweep"] = PerturbationSweepAudit(
            recall_ks=PAIR_HIT_KS,
            levels=args.perturbation_levels,
            seeds=args.perturbation_seeds,
            device=args.device,
        ).run(active, eval_loader)
        results["physical_consistency"] = PhysicalConsistencyAudit(
            min_checked=args.minimum_pvr_checked,
            device=args.device,
        ).run(active, eval_loader)
    if "graph" in args.steps:
        results["graph_audit"] = GraphLevelAudit(
            top_k_motifs=args.top_k_motifs,
            mine_batches=args.train_samples,
            min_motif_support=args.minimum_motif_train_support,
            min_eval_support=args.minimum_motif_eval_support,
            device=args.device,
        ).run(active, eval_loader, motif_loader=train_loader)
    if "grounding" in args.steps:
        results["grounding_error_decomposition"] = GroundingErrorDecompositionAudit(
            iou_threshold=args.sgg_iou,
            device=args.device,
            class_frequency=seen_metadata.get("object_class_support", {}),
        ).run(active, eval_loader)

    if not args.allow_incomplete_audits:
        _assert_complete(dataset, results, active)
    results["model_provenance"] = model_provenance(active)
    results["dataset_metadata"] = {
        "ontology_id": getattr(eval_loader.dataset, "ontology_id", None),
        "train_images": len(train_loader.dataset),
        "eval_images": len(eval_loader.dataset),
        "seen_triplets": len(seen_triplets),
        "seen_triplet_provenance": seen_metadata,
        "active_models": list(active),
        "active_families": sorted({model.architecture_family for model in active.values()}),
    }
    write_json(output_dir / dataset / "results.json", results)
    return results


def _cross_dataset_rows(all_results: dict) -> list[dict]:
    rows = []
    for dataset, results in all_results.items():
        provenance = results.get("model_provenance", {})
        for name, model_info in provenance.items():
            standard = results.get("standard_sgg", {}).get(name, {})
            sgdet = standard.get("tasks", {}).get("sgdet", {}).get("metrics", {})
            pair = results.get("pair_audit", {}).get(name, {})
            graph = results.get("graph_audit", {}).get(name, {})
            physical = results.get("physical_consistency", {}).get(name, {})
            feature = results.get("feature_audit", {}).get(name, {})
            rows.append({
                "dataset": dataset,
                "model": name,
                "architecture_family": model_info.get("architecture_family"),
                "checkpoint_sha256": model_info.get("checkpoint_status", {}).get("sha256"),
                "R@50": sgdet.get("R@50"),
                "mR@50": sgdet.get("mR@50"),
                "zR@50": sgdet.get("zR@50"),
                "effective_rank": feature.get("effective_rank"),
                "dirichlet_energy": feature.get("dirichlet_energy"),
                "BRR@5": pair.get("brr_at_k", {}).get("5", pair.get("BRR")),
                "MAR": graph.get("MAR"),
                "PVR": physical.get("PVR"),
                "pvr_checked": physical.get("pvr_checked"),
                "PVR_coverage": physical.get("coverage"),
            })
    return rows


def main():
    args = parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets contains duplicates")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_model_panel(args.model_panel)
    models = _load_models(args, panel)

    all_results = {}
    for dataset in args.datasets:
        print(f"\n{'=' * 72}\nDATASET: {dataset.upper()}\n{'=' * 72}")
        all_results[dataset] = _run_dataset(args, dataset, models, output_dir)

    load_report = {
        name: {
            **model_provenance({name: model})[name],
            "parameter_count": model.checkpoint_status["parameter_count"],
            "paradigm": model.paradigm,
        }
        for name, model in models.items()
    }
    evidence = EvidenceChainAnalyzer(
        performance_k=50, diagnostic_k=args.primary_k
    ).analyze(all_results, load_report)
    summary = {
        "experiment": "IV_standard_official_model_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(args.datasets),
        "diagnostic_reuse_contract": {
            "pair_and_perturbation": "Experiment II",
            "motif_intervention": "not_in_converged_submission",
            "rerun_inside_experiment_4": False,
        },
        "candidate_panel": panel_summary(args.model_panel),
        "loaded_model_runs": len(models),
        "loaded_model_families": sorted({model.architecture_family for model in models.values()}),
        "formal_model_family_count": len({model.architecture_family for model in models.values()}),
        "cross_dataset_rows": _cross_dataset_rows(all_results),
        "evidence_chain": evidence,
        "model_provenance": load_report,
        "config": vars(args),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"\nCompleted Experiment IV. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
