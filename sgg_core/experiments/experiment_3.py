"""Experiment III: real-checkpoint motif intervention audit.

Motifs are selected from the training split only and evaluated on a disjoint
test split.  MAR excludes predictions that were already wrong before ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sgg_core.protocol import (
    build_loaders, load_official_models, model_provenance,
    seed_everything, write_json,
)
from sgg_core.audits.graph_audit import GraphLevelAudit


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
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--top_k_motifs", type=int, default=20)
    parser.add_argument("--minimum_train_support", type=int, default=20)
    parser.add_argument("--minimum_eval_support", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    train_loader, eval_loader = build_loaders(
        args.dataset, args.data_root, args.train_samples, args.eval_samples,
        args.train_ann, args.eval_ann, args.image_root, args.panoptic_root,
    )
    models = load_official_models(
        args.official_manifest, args.device, args.dataset, eval_loader
    )
    unsupported = [
        name for name, model in models.items()
        if not bool(
            model.manifest.get("diagnostic_contract", {}).get(
                "graph_intervention", False
            )
        )
    ]
    if unsupported:
        raise RuntimeError(
            "Experiment III requires live, model-consumed graph interventions; "
            f"unsupported models={unsupported}"
        )
    results = GraphLevelAudit(
        top_k_motifs=args.top_k_motifs,
        mine_batches=args.train_samples,
        min_motif_support=args.minimum_train_support,
        min_eval_support=args.minimum_eval_support,
        device=args.device,
    ).run(models, eval_loader, motif_loader=train_loader)
    output = {
        "experiment": "III_real_checkpoint_motif_intervention",
        "dataset": args.dataset,
        "paper_role": "appendix_supporting_diagnostic",
        "submission_scope": "matched_motifs_and_transformer_live_depth_panel",
        "motif_source": "training_split_only",
        "causal_claims": "controlled_intervention_not_full_causal_identification",
        "intervention_protocol": {
            "name": "gt_box_conditioned_endpoint_evidence_removal",
            "target": "object_terminal_of_frequent_training_motif",
            "fixed": [
                "gt_boxes", "entity_labels", "relation_pairs",
                "relation_labels", "graph_topology",
            ],
            "matched_control": "closest_area_non_endpoint_node",
            "segmentation_claim": "none",
        },
        "randomness_protocol": {
            "intervention": "deterministic_for_a_fixed_checkpoint",
            "checkpoint_replication": (
                "use independently trained checkpoint manifests; do not count "
                "repeated inference as training seeds"
            ),
        },
        "graph_audit": results,
        "model_provenance": model_provenance(models),
        "config": vars(args),
    }
    write_json(Path(args.output_dir) / "experiment_3.json", output)


if __name__ == "__main__":
    main()
