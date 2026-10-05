"""Experiment II: real-checkpoint pair grounding and prior-dependence audit.

This paper entry point refuses surrogate or randomly initialized models. The
former GT-injected simulator is intentionally absent from this core package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgg_core.protocol import (
    PAIR_HIT_KS, build_loaders, load_official_models, model_provenance,
    seed_everything, write_json,
)
from sgg_core.audits.pair_audit import PairLevelAudit
from sgg_core.audits.object_propagation import ObjectErrorPropagationAudit
from sgg_core.audits.perturbation_sweep import PerturbationSweepAudit
from sgg_core.audits.physical_consistency import PhysicalConsistencyAudit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vg", "oi", "gqa", "psg", "vrd"), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_ann")
    parser.add_argument("--eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--panoptic_root")
    parser.add_argument("--official_manifest", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--analysis_scope", choices=("observational", "interventional", "both"),
        default="both",
        help=(
            "observational reuses SGCls/SGDet predictions for endpoint-error "
            "stratification; interventional requires a GT-pair live adapter"
        ),
    )
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument(
        "--pair_hit_ks", nargs="+", type=int, default=list(PAIR_HIT_KS),
        help="Predicate Hit@K on GT pairs; this is not image-level SGG Recall@K.",
    )
    parser.add_argument("--primary_k", type=int, default=5)
    parser.add_argument("--perturbation_levels", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--perturbation_strategies", nargs="+",
        default=[
            "key_node_mask", "random_node_mask", "unrelated_node_mask",
            "on_manifold_replacement", "color_jitter",
        ],
    )
    parser.add_argument("--perturbation_seeds", nargs="+", type=int,
                        default=[17, 29, 43])
    parser.add_argument("--minimum_relations", type=int, default=200)
    parser.add_argument("--minimum_clean_recall", type=float, default=0.01)
    parser.add_argument("--minimum_pvr_checked", type=int, default=100)
    parser.add_argument(
        "--skip_pair_audit", action="store_true",
        help="Skip the duplicate terminal BRR audit when the dose sweep is primary.",
    )
    parser.add_argument(
        "--skip_physical_consistency", action="store_true",
        help="Skip PVR in the converged object-error propagation experiment.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    cache_only = bool(args.official_manifest) and all(
        json.loads(Path(path).expanduser().read_text(encoding="utf-8")).get(
            "execution_mode"
        ) == "prediction_cache"
        for path in args.official_manifest
    )
    image_root = None if cache_only else args.image_root
    panoptic_root = None if cache_only else args.panoptic_root
    train_loader, eval_loader = build_loaders(
        args.dataset, args.data_root, args.train_samples, args.eval_samples,
        args.train_ann, args.eval_ann, image_root, panoptic_root,
        include_proxy_features=not cache_only,
        include_raw_images=not cache_only,
    )
    models = load_official_models(
        args.official_manifest, args.device, args.dataset, eval_loader
    )
    pair_results = sweep_results = pvr_results = None
    if args.analysis_scope in {"interventional", "both"}:
        if not args.skip_pair_audit:
            pair_results = PairLevelAudit(
                recall_ks=args.pair_hit_ks, primary_k=args.primary_k,
                min_relations=args.minimum_relations,
                min_clean_recall=args.minimum_clean_recall,
                perturbation_seeds=args.perturbation_seeds,
                device=args.device,
            ).run(models, eval_loader)
        sweep_results = PerturbationSweepAudit(
            recall_ks=args.pair_hit_ks, levels=args.perturbation_levels,
            seeds=args.perturbation_seeds,
            strategies=args.perturbation_strategies, device=args.device,
        ).run(models, eval_loader)
        if not args.skip_physical_consistency:
            pvr_results = PhysicalConsistencyAudit(
                min_checked=args.minimum_pvr_checked, device=args.device
            ).run(models, eval_loader)
    propagation_results = None
    if args.analysis_scope in {"observational", "both"}:
        propagation_results = ObjectErrorPropagationAudit(
            ks=(1, 5), device=args.device,
        ).run(models, eval_loader)
    output = {
        "experiment": "II_real_checkpoint_pair_grounding",
        "dataset": args.dataset,
        "analysis_scope": args.analysis_scope,
        "scan_profile": (
            "full_vg" if args.dataset == "vg" and len(args.perturbation_levels) > 3
            else "three_strength_external"
        ),
        "metric_scope": "GT-pair diagnostic; standard image-level SGG metrics are Experiment IV",
        "pair_metric": "predicate Hit@K with K restricted below the predicate vocabulary size",
        "causal_claims": (
            "controlled_feature_intervention_not_total_causal_effect"
            if args.analysis_scope in {"interventional", "both"}
            else "association_only"
        ),
        "randomness_protocol": {
            "base_perturbation_seeds": list(args.perturbation_seeds),
            "per_image_seed_derivation": "base_seed + 1000003 * (image_index + 1)",
            "bootstrap_unit": "image after averaging perturbation seeds",
        },
        "pair_audit": pair_results,
        "dose_response_and_controls": sweep_results,
        "physical_consistency": pvr_results,
        "object_error_propagation": propagation_results,
        "model_provenance": model_provenance(models),
        "config": vars(args),
    }
    write_json(Path(args.output_dir) / "experiment_2.json", output)


if __name__ == "__main__":
    main()
