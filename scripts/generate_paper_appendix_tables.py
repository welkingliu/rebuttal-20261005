#!/usr/bin/env python3
"""Validate the curated appendix or export the legacy I--IV table block.

The submission appendix contains hand-reviewed prose plus Experiment III and V
tables that were added after the original generator.  The default command is
therefore intentionally read-only: it verifies that the curated appendix still
contains every required result table.  ``--legacy-output`` is available only
for auditing the older generated I--IV block and never rewrites ``main.tex``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


BEGIN = "% BEGIN AUTO-GENERATED FORMAL RESULT TABLES"
END = "% END AUTO-GENERATED FORMAL RESULT TABLES"
KS = (1, 5, 10, 20, 50, 100)

REQUIRED_CURATED_LABELS = (
    "tab:app-inventory",
    "tab:app-exp1a-complete",
    "tab:app-exp1a-endpoints",
    "tab:app-exp1b-r",
    "tab:app-exp1b-diagnostics",
    "tab:app-exp2-coverage",
    "tab:app-exp2-groups",
    "tab:app-exp2b-dose",
    "tab:app-exp2b-endpoints",
    "tab:app-exp3-motif",
    "tab:app-exp4-r",
    "tab:app-exp4-grounding",
    "tab:app-exp4-depth-metrics",
    "tab:app-exp4-depth-reference",
    "tab:app-kern-sgcls",
    "tab:app-exp5-vg",
    "tab:app-exp5-test",
    "tab:app-exp5-external",
)

BACKBONES = (
    ("cradio_v4_so400m", "C-RADIOv4"),
    ("dinov2_b", "DINOv2-B"),
    ("radio_v25_b", "RADIOv2.5-B"),
    ("siglip2_b", "SigLIP2-B"),
    ("sam_vit_b", "SAM ViT-B"),
    ("resnet50", "ResNet-50"),
)

EXP4_GROUNDING_MODELS = (
    ("egtr_vg_official", "VG", "EGTR"),
    ("reltr_vg_official", "VG", "RelTR"),
    ("sgtr_vg_official", "VG", "SGTR"),
    ("egtr_oi_official", "OI", "EGTR"),
    ("sgtr_oi_official", "OI", "SGTR"),
    ("openpsg_motifs_psg_official", "PSG", "Motifs"),
    ("openpsg_vctree_psg_official", "PSG", "VCTree"),
    ("openpsg_psgformer_psg_official", "PSG", "PSGFormer"),
    ("openpsg_psgtr_psg_official", "PSG", "PSGTR"),
)

EXP4_STANDARD_MODELS = (
    ("egtr_vg_official", "VG", "EGTR"),
    ("reltr_vg_official", "VG", "RelTR"),
    ("sgtr_vg_official", "VG", "SGTR"),
    ("kern_official", "VG", "KERN"),
    ("pysgg_bgnn_vg_sgdet_official", "VG", "BGNN"),
    ("egtr_oi_official", "OI", "EGTR"),
    ("sgtr_oi_official", "OI", "SGTR"),
    ("openpsg_motifs_psg_official", "PSG", "Motifs"),
    ("openpsg_vctree_psg_official", "PSG", "VCTree"),
    ("openpsg_psgformer_psg_official", "PSG", "PSGFormer"),
    ("openpsg_psgtr_psg_official", "PSG", "PSGTR"),
)

EXP2_RUNS = (
    ("vg", "egtr_vg_official", "sgdet", "VG", "EGTR"),
    ("vg", "kern_official", "sgdet", "VG", "KERN"),
    ("vg", "kern_official", "sgcls", "VG", "KERN"),
    ("vg", "sgtr_vg_official", "sgdet", "VG", "SGTR"),
    ("oi", "egtr_oi_official", "sgdet", "OI", "EGTR"),
    ("oi", "sgtr_oi_official", "sgdet", "OI", "SGTR"),
    ("psg", "openpsg_motifs_psg_official", "sgdet", "PSG", "Motifs"),
    ("psg", "openpsg_vctree_psg_official", "sgdet", "PSG", "VCTree"),
)

EXP2_LIVE_RUNS = (
    ("pysgg_motifs_vg_live", "PySGG Motifs"),
    ("pysgg_transformer_vg_live", "PySGG Transformer"),
)

EXP2_INTERVENTIONS = (
    ("key_node_mask", "Key-node mask"),
    ("random_node_mask", "Random-node mask"),
    ("unrelated_node_mask", "Unrelated-node mask"),
    ("on_manifold_replacement", "On-manifold replacement"),
    ("color_jitter", "Color jitter"),
)

EXP4_DEPTH_RUNS = (
    ("pysgg_motifs_vg_tritask", "PySGG Motifs"),
    ("pysgg_transformer_vg_tritask", "PySGG Transformer"),
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing formal result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return math.nan, math.nan
    return statistics.mean(clean), statistics.pstdev(clean)


def pct(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.{digits}f}"


def pct_pm(values: Iterable[float], digits: int = 2) -> str:
    mean, std = mean_std(values)
    if not math.isfinite(mean):
        return "--"
    return f"${100.0 * mean:.{digits}f}\\pm{100.0 * std:.{digits}f}$"


def num_pm(values: Iterable[float], digits: int = 3) -> str:
    mean, std = mean_std(values)
    if not math.isfinite(mean):
        return "--"
    return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"


def ci_pct(value: float, interval: list[float]) -> str:
    return f"{pct(value)} [{pct(interval[0])}, {pct(interval[1])}]"


def table(lines: list[str], caption: str, label: str, columns: str,
          header: str, rows: list[str], *,
          font: str = "footnotesize", wide: bool = True,
          tabcolsep: float = 2.5, arraystretch: float = 1.0) -> None:
    environment = "table*" if wide else "table"
    lines.extend([
        f"\\begin{{{environment}}}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        f"\\{font}",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep:g}pt}}",
        f"\\renewcommand{{\\arraystretch}}{{{arraystretch:g}}}",
    ])
    lines.extend([
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ])
    lines.extend([f"\\end{{{environment}}}", ""])


def exp1a_tables(root: Path, lines: list[str]) -> None:
    base = root / "artifacts/experiment_1a/exp1a_refit_zscore_20260722_144126"
    summaries = {key: load(base / key / "summary.json") for key, _ in BACKBONES}
    rows = []
    view_names = (("box", "Box"), ("pred_mask", "SAM mask"), ("gt_mask", "GT mask"))
    for key, name in BACKBONES:
        for view_key, view_name in view_names:
            agg = summaries[key]["views"][view_key]["aggregate"]
            rows.append(
                f"{name} & {view_name} & "
                f"{pct_pm([r['metrics']['top1_accuracy'] for r in summaries[key]['views'][view_key]['runs']])} & "
                f"{pct_pm([r['metrics']['top5_accuracy'] for r in summaries[key]['views'][view_key]['runs']])} & "
                f"{pct_pm([r['metrics']['macro_accuracy'] for r in summaries[key]['views'][view_key]['runs']])} & "
                f"{pct_pm([r['metrics']['ece_15'] for r in summaries[key]['views'][view_key]['runs']])} & "
                f"{num_pm([r['metrics']['nll'] for r in summaries[key]['views'][view_key]['runs']])} & "
                f"{num_pm([r['metrics']['brier'] for r in summaries[key]['views'][view_key]['runs']])} \\\\"
            )
    table(
        lines,
        "Complete Experiment I-A object-probe results on PSG (mean $\\pm$ standard deviation over seeds 17, 23, and 31). Accuracy and validation-temperature-scaled ECE are percentages. SAM masks are predicted from ground-truth box prompts.",
        "tab:app-exp1a-complete",
        "llrrrrrr",
        "Backbone & Support & Top-1 & Top-5 & Macro & ECE & NLL & Brier",
        rows,
    )

    rows = []
    for key, name in BACKBONES:
        runs = summaries[key]["views"]["pred_mask"]["runs"]
        hbt = {
            group: pct_pm(r["metrics"]["head_body_tail_accuracy"][group] for r in runs)
            for group in ("head", "body", "tail")
        }
        area = {
            group: pct_pm(r["metrics"]["area_accuracy"][group]["top1_accuracy"] for r in runs)
            for group in ("small", "medium", "large")
        }
        rows.append(
            f"{name} & {hbt['head']} & {hbt['body']} & {hbt['tail']} & "
            f"{area['small']} & {area['medium']} & {area['large']} \\\\"
        )
    table(
        lines,
        "Experiment I-A SAM-mask Top-1 accuracy by training-frequency and object-area strata (\\%). Frequency strata are fixed from the probe training partition.",
        "tab:app-exp1a-strata",
        "lrrrrrr",
        "Backbone & Head & Body & Tail & Small & Medium & Large",
        rows,
    )

    bins = ("[0.00,0.50)", "[0.50,0.70)", "[0.70,0.85)", "[0.85,0.95)", "[0.95,1.00]")
    rows = []
    for key, name in BACKBONES:
        runs = summaries[key]["views"]["pred_mask"]["runs"]
        values = [pct_pm(r["metrics"]["mask_iou_bins"][b]["top1_accuracy"] for r in runs) for b in bins]
        rows.append(f"{name} & " + " & ".join(values) + " \\\\")
    table(
        lines,
        "Experiment I-A SAM-mask object Top-1 accuracy (\\%) by predicted-mask IoU bin. The same 11,667 evaluation objects are partitioned by mask quality.",
        "tab:app-exp1a-iou",
        "lrrrrr",
        "Backbone & $[0,.50)$ & $[.50,.70)$ & $[.70,.85)$ & $[.85,.95)$ & $[.95,1]$",
        rows,
    )

    thresholds = ("both_iou>=0.75", "both_iou>=0.85", "both_iou>=0.90", "both_iou>=0.95")
    rows = []
    for key, name in BACKBONES:
        runs = summaries[key]["views"]["pred_mask"]["runs"]
        vals = []
        for threshold in thresholds:
            vals.append(pct_pm(
                r["relationship_endpoints"]["conditioned_on_both_endpoint_mask_iou"][threshold]["endpoint_failure_rate"]
                for r in runs
            ))
        rows.append(f"{name} & " + " & ".join(vals) + " \\\\")
    table(
        lines,
        "Experiment I-A endpoint identity disagreement rate (\\%) conditioned on both relation endpoints meeting the stated box-prompted mask-IoU threshold. Supports are 3,012, 1,511, 621, and 60 relations at thresholds 0.75, 0.85, 0.90, and 0.95, respectively.",
        "tab:app-exp1a-endpoints",
        "lrrrr",
        "Backbone & IoU$\\geq.75$ & IoU$\\geq.85$ & IoU$\\geq.90$ & IoU$\\geq.95$",
        rows,
        font="scriptsize",
        tabcolsep=1.2,
        arraystretch=0.94,
    )


def exp1b_tables(root: Path, lines: list[str]) -> None:
    base = root / "artifacts/experiment_1b/exp1b_20260716_085021"
    summaries = {key: load(base / key / "summary.json") for key, _ in BACKBONES}
    for metric, label in (("R", "Recall"), ("mR", "mean Recall"), ("zR", "zero-shot Recall")):
        rows = []
        for key, name in BACKBONES:
            runs = summaries[key]["runs"]
            for depth in (0, 2, 4, 8):
                selected = [r for r in runs if int(r["depth"]) == depth]
                values = [pct_pm(r["evaluation"]["metrics"][f"{metric}@{k}"] for r in selected) for k in KS]
                rows.append(f"{name} & {depth} & " + " & ".join(values) + " \\\\")
        table(
            lines,
            f"Complete Experiment I-B VG-150 PredCls {label} (\\%, mean $\\pm$ standard deviation over three seeds). Ranking is image-level and uses one top-$K$ budget per image.",
            f"tab:app-exp1b-{metric.lower()}",
            "lrrrrrrr",
            "Backbone & Depth & @1 & @5 & @10 & @20 & @50 & @100",
            rows,
        )

    rows = []
    for key, name in BACKBONES:
        runs = summaries[key]["runs"]
        for depth in (0, 2, 4, 8):
            selected = [r for r in runs if int(r["depth"]) == depth]
            ranks, energies, pvrs, checked, coverage = [], [], [], [], []
            for run in selected:
                diag = run["evaluation"]["layer_diagnostics"]
                final_layer = diag[str(max(int(k) for k in diag))]
                ranks.append(final_layer["effective_rank"])
                energies.append(final_layer["dirichlet_energy"])
                pc = run["evaluation"]["physical_consistency"]
                if pc["PVR"] is not None:
                    pvrs.append(pc["PVR"])
                checked.append(pc["pvr_checked"])
                coverage.append(pc["coverage"])
            pvr_text = pct_pm(pvrs) if pvrs else "--"
            rows.append(
                f"{name} & {depth} & {num_pm(ranks)} & {num_pm(energies)} & "
                f"{pvr_text} & {len(pvrs)}/3 & {statistics.mean(checked):.0f} & {pct(statistics.mean(coverage), 2)} \\\\"
            )
    table(
        lines,
        "Experiment I-B representation and physical-consistency diagnostics. Effective rank and Dirichlet energy are taken at the final reasoning layer. PVR is shown only for seeds meeting the registered minimum of 100 checked pairs; valid gives the number of eligible seeds.",
        "tab:app-exp1b-diagnostics",
        "lrrrrrrr",
        "Backbone & Depth & Eff. rank & Dirichlet & PVR (\\%) & Valid & Checked & Coverage (\\%)",
        rows,
    )


def exp2_paths(root: Path, dataset: str, model: str) -> Path:
    if dataset == "vg":
        base = root / "artifacts/experiment_2/mac_cpu_queue_20260721_144107_vg_observational_full/vg"
    elif dataset == "psg":
        base = root / "artifacts/experiment_2/mac_cpu_queue_20260721_144107_psg_observational_full/psg"
    else:
        base = root / "artifacts/experiment_2/exp2_observational_scheduled_20260718_210647/oi/oi"
    return base / model / "experiment_2.json"


def exp2_tables(root: Path, lines: list[str]) -> None:
    payloads = {
        (dataset, model): load(exp2_paths(root, dataset, model))
        for dataset, model, _, _, _ in EXP2_RUNS
    }
    coverage_rows, group_rows = [], []
    group_names = (("both_correct", "Both match"), ("one_wrong", "One differs"), ("both_wrong", "Both differ"))
    for dataset, model, task_name, dataset_name, model_name in EXP2_RUNS:
        task = payloads[(dataset, model)]["object_error_propagation"][model]["tasks"][task_name]
        counts = task["counts"]
        coverage_rows.append(
            f"{dataset_name} & {model_name} & {task['task']} & {task['num_images']:,} & "
            f"{counts['gt_relations']:,} & {counts['localized_endpoint_relations']:,} & "
            f"{counts['evaluable_relations']:,} & {pct(task['localized_endpoint_coverage'])} & "
            f"{pct(task['endpoint_and_pair_coverage'])} \\\\"
        )
        for group_key, group_name in group_names:
            group = task["groups"][group_key]
            group_rows.append(
                f"{dataset_name} & {model_name} & {task['task']} & {group_name} & {group['support']:,} & "
                f"{ci_pct(group['relation_Hit@1'], group['bootstrap_95ci_Hit@1'])} & "
                f"{ci_pct(group['relation_Hit@5'], group['bootstrap_95ci_Hit@5'])} \\\\"
            )
    table(
        lines,
        "Experiment II-A observational-audit coverage. Pair coverage is the fraction of all ground-truth relations that are localized and have evaluable endpoint and predicate predictions.",
        "tab:app-exp2-coverage",
        "lllrrrrrr",
        "Data & Model & Task & Images & GT rel. & Localized & Evaluable & Loc. (\\%) & Pair (\\%)",
        coverage_rows,
    )
    table(
        lines,
        "Complete Experiment II-A endpoint-agreement strata. Predicate Hit@1 and Hit@5 are percentages followed by image-bootstrap 95\\% confidence intervals. These are observational associations, not intervention effects.",
        "tab:app-exp2-groups",
        "llllrrr",
        "Data & Model & Task & Endpoint stratum & Support & Hit@1 [95\\% CI] & Hit@5 [95\\% CI]",
        group_rows,
    )


def exp2_live_path(root: Path, model: str) -> Path:
    return (
        root
        / "artifacts/experiment_2/postcache_submission_20260723_104627_exp2/vg"
        / model
        / "experiment_2.json"
    )


def exp2_live_tables(root: Path, lines: list[str]) -> None:
    payloads = {model: load(exp2_live_path(root, model)) for model, _ in EXP2_LIVE_RUNS}
    dose_rows = []
    for model, model_name in EXP2_LIVE_RUNS:
        dose = payloads[model]["dose_response_and_controls"][model]
        for strategy, strategy_name in EXP2_INTERVENTIONS:
            curve = dose["strategies"][strategy]["curve"]
            values = [pct(curve[level]["1"]["mean"]) for level in ("0.000", "0.100", "0.250", "0.500", "1.000")]
            terminal = curve["1.000"]["1"]
            terminal_ci = terminal["paired_drop_bootstrap_95ci"]
            dose_rows.append(
                f"{model_name} & {strategy_name} & {terminal['n_paired']:,} & "
                + " & ".join(values)
                + f" & {pct(terminal['absolute_drop'])} [{pct(terminal_ci[0])}, {pct(terminal_ci[1])}] \\\\"
            )
    table(
        lines,
        "Experiment II-B primary dose--response results on the VG-150 evaluation split. Entries are macro-image predicate Hit@1 (\\%); the final column is the paired clean-minus-perturbed drop at $\\alpha=1$ with an image-clustered 95\\% bootstrap interval. Intervention seeds 17, 29, and 43 are averaged within image before resampling. One malformed image is excluded from the 2,000-image run; unrelated-node and on-manifold rows additionally report their exact eligible paired-image counts.",
        "tab:app-exp2b-dose",
        "llrrrrrrr",
        "Model & Intervention & Paired $n$ & $\\alpha=0$ & $.1$ & $.25$ & $.5$ & $1$ & Drop [95\\% CI]",
        dose_rows,
        font="scriptsize",
    )

    endpoint_rows = []
    group_names = (("both_correct", "Both match"), ("one_wrong", "One differs"), ("both_wrong", "Both differ"))
    for model, model_name in EXP2_LIVE_RUNS:
        tasks = payloads[model]["object_error_propagation"][model]["tasks"]
        for task_name in ("sgcls", "sgdet"):
            task = tasks[task_name]
            for group_key, group_name in group_names:
                group = task["groups"][group_key]
                endpoint_rows.append(
                    f"{model_name} & {task_name.upper()} & {group_name} & {group['support']:,} & "
                    f"{ci_pct(group['relation_Hit@1'], group['bootstrap_95ci_Hit@1'])} & "
                    f"{ci_pct(group['relation_Hit@5'], group['bootstrap_95ci_Hit@5'])} \\\\"
                )
    table(
        lines,
        "Experiment II-B endpoint-agreement decomposition from the same live adapters. Hit rates are percentages followed by image-clustered bootstrap 95\\% intervals. SGCls evaluates 16,963 relations (99.94\\% pair coverage); SGDet evaluates 9,933 relations (58.52\\% pair coverage). This decomposition is associative; the controlled dose--response in Table~\\ref{tab:app-exp2b-dose} provides the intervention evidence.",
        "tab:app-exp2b-endpoints",
        "lllrrr",
        "Model & Task & Endpoint stratum & Support & Hit@1 [95\\% CI] & Hit@5 [95\\% CI]",
        endpoint_rows,
    )


def exp4_results_path(root: Path, model: str, dataset: str) -> Path:
    if model in {"kern_official", "pysgg_bgnn_vg_sgdet_official"}:
        return root / "artifacts/experiment_4/mac_exp4_vg_full_20260721_142949/vg/results.json"
    base = root / "artifacts/experiment_4/exp4_sgdet_submission_fixed_20260718_192834/shards"
    ds = dataset.lower()
    return base / model / ds / ds / "results.json"


def exp4_tables(root: Path, lines: list[str]) -> None:
    payloads = {
        model: load(exp4_results_path(root, model, dataset))
        for model, dataset, _ in EXP4_STANDARD_MODELS
    }
    for metric, label in (("R", "Recall"), ("mR", "mean Recall"), ("zR", "zero-shot Recall")):
        rows = []
        for model, dataset, model_name in EXP4_STANDARD_MODELS:
            task = payloads[model]["standard_sgg"][model]["tasks"]["sgdet"]
            values = [pct(task["metrics"][f"{metric}@{k}"]) for k in KS]
            rows.append(f"{dataset} & {model_name} & " + " & ".join(values) + " \\\\")
        table(
            lines,
            (f"Full-split Experiment IV SGDet {label} (\\%). All 11 rows pass provenance, ontology, and cache-coverage checks. Results preserve each dataset's native ontology and are comparable only within dataset blocks. "
             + ("Zero-shot supports are 7,601 relations for VG, 27 for OI, and 427 for PSG. " if metric == "zR" else "")
             + "PSG uses its native panoptic-mask matching protocol."),
            f"tab:app-exp4-{metric.lower()}",
            "llrrrrrr",
            "Data & Model & @1 & @5 & @10 & @20 & @50 & @100",
            rows,
            font="footnotesize",
        )

    rows = []
    for model, dataset, model_name in EXP4_GROUNDING_MODELS:
        payload = payloads[model]
        task = payload["standard_sgg"][model]["tasks"]["sgdet"]
        grounding = payload["grounding_error_decomposition"][model]
        gm = grounding["metrics"]
        oi = grounding["object_identity"]
        repro = payload["reproduction_validation"][model]["status"].replace("_", " ")
        rows.append(
            f"{dataset} & {model_name} & {task['num_images']:,} & {task['num_ground_truth_relations']:,} & "
            f"{pct(gm['localization_recall'])} & {pct(gm['recognition_accuracy_given_localized'])} & "
            f"{pct(gm['grounded_object_recall'])} & {pct(oi['macro_accuracy_given_localized'])} & "
            f"{pct(oi['ece_15'])} & {repro} \\\\"
        )
    table(
        lines,
        "Experiment IV grounding decomposition for the nine runs with exported decomposition records. Recognition is conditioned on localized objects at IoU 0.5; macro accuracy and ECE use the localized object-class logits. Reproduction status is evaluated only against declared checkpoint-compatible references.",
        "tab:app-exp4-grounding",
        "llrrrrrrrl",
        "Data & Model & Images & GT rel. & Loc. & Rec.$|$loc. & Grounded & Macro & ECE & Reproduction",
        rows,
        font="footnotesize",
    )

def exp4_depth_path(root: Path, model: str) -> Path:
    return (
        root
        / "artifacts/experiment_4/postcache_submission_20260723_104627_exp4/shards"
        / model
        / "vg/vg/results.json"
    )


def exp4_depth_tables(root: Path, lines: list[str]) -> None:
    payloads = {model: load(exp4_depth_path(root, model)) for model, _ in EXP4_DEPTH_RUNS}
    metric_rows = []
    task_labels = {"predcls": "PredCls", "sgcls": "SGCls", "sgdet": "SGDet"}
    for model, model_name in EXP4_DEPTH_RUNS:
        tasks = payloads[model]["standard_sgg"][model]["tasks"]
        for task_name in ("predcls", "sgcls", "sgdet"):
            for metric in ("R", "mR", "zR"):
                values = [pct(tasks[task_name]["metrics"][f"{metric}@{k}"]) for k in KS]
                metric_rows.append(
                    f"{model_name} & {task_labels[task_name]} & {metric} & "
                    + " & ".join(values)
                    + " \\\\"
                )
    table(
        lines,
        "Experiment IV matched-depth VG-150 results over all 26,446 test images and 183,642 ground-truth relations. Both locally integrated PySGG checkpoints use the same ontology, image-level triplet ranking, and evaluator. Zero-shot metrics use 7,601 relations whose triplets are absent from the registered 5,000-image training manifest.",
        "tab:app-exp4-depth-metrics",
        "lllrrrrrr",
        "Model & Task & Metric & @1 & @5 & @10 & @20 & @50 & @100",
        metric_rows,
        font="scriptsize",
    )


def exp4_compact_tables(root: Path, lines: list[str]) -> None:
    kern_path = root / "artifacts/experiment_4/mac_cpu_queue_20260721_144107_kern_sgcls_full/vg/results.json"
    kern = load(kern_path)["standard_sgg"]["kern_official"]["tasks"]["sgcls"]["metrics"]
    kern_rows = [
        f"@{k} & {pct(kern[f'R@{k}'])} & {pct(kern[f'mR@{k}'])} & {pct(kern[f'zR@{k}'])} \\\\"
        for k in KS
    ]
    payloads = {model: load(exp4_depth_path(root, model)) for model, _ in EXP4_DEPTH_RUNS}
    reference_rows = []
    for model, model_name in EXP4_DEPTH_RUNS:
        short_name = model_name.removeprefix("PySGG ")
        validation = payloads[model]["reproduction_validation"][model]
        for task_name in ("PredCls", "SGCls", "SGDet"):
            cells = []
            for metric in ("R@50", "mR@50"):
                comparison = validation["comparisons"][f"{task_name}/{metric}"]
                signed_delta = 100.0 * (comparison["observed"] - comparison["reference"])
                status = "P" if comparison["within_tolerance"] else "X"
                cells.append(
                    f"{pct(comparison['reference'])}/{pct(comparison['observed'])}/"
                    f"{signed_delta:+.2f}/{status}"
                )
            reference_rows.append(
                f"{short_name} & {task_name} & {cells[0]} & {cells[1]} \\\\"
            )
    lines.extend([
        "\\begin{table*}[t]",
        "\\caption{Compact Experiment IV auxiliary results. (a) Full-split VG-150 KERN SGCls. (b) Matched-depth audit at $K=50$; entries are reference/observed/signed delta/status, where P is within the two-point tolerance and X is outside. Neither tri-task run is an exact reproduction.}",
        "\\label{tab:app-kern-sgcls}",
        "\\label{tab:app-exp4-depth-reference}",
        "\\centering",
        "\\begin{minipage}[t]{0.29\\textwidth}",
        "\\centering",
        "\\scriptsize",
        "\\textbf{(a) KERN SGCls (\\%)}\\par\\smallskip",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{0.94}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "$K$ & R@$K$ & mR@$K$ & zR@$K$ \\\\",
        "\\midrule",
        *kern_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "\\hfill",
        "\\begin{minipage}[t]{0.68\\textwidth}",
        "\\centering",
        "\\scriptsize",
        "\\textbf{(b) Matched-depth reference audit}\\par\\smallskip",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Model & Task & R@50 & mR@50 \\\\",
        "\\midrule",
        *reference_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "\\end{table*}",
        "",
    ])


def build(root: Path) -> str:
    lines = [
        BEGIN,
        "\\section{Complete Formal Result Tables}",
        "\\label{app:formal-results}",
        "",
        "This appendix contains only paper-eligible full, formal, or converged runs. Smoke tests, 100/1,000-image debugging subsets, failed attempts, superseded Experiment~I runs, and duplicate reruns are excluded. Percentages are reported on the 0--100 scale. Native dataset ontologies and task contracts are retained; values from VG, OI, and PSG are not treated as a shared leaderboard.",
        "",
        "\\begin{table*}[t]",
        "\\caption{Formal-result inventory included in this appendix. Pending registered experiments remain explicit to-dos and receive no numerical placeholders. Completed runs without a checkpoint-compatible reference are reported with reproduction status not applicable.}",
        "\\label{tab:app-inventory}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Stage & Dataset/task & Formal scale & Included systems & Appendix tables \\\\",
        "\\midrule",
        "I-A & PSG object probe & 65,102 train / 11,667 eval objects & 6 backbones, 3 views, 3 seeds & \\ref{tab:app-exp1a-complete}--\\ref{tab:app-exp1a-endpoints} \\\\",
        "I-B & VG-150 PredCls & 2,000 test images & 6 backbones, 4 depths, 3 seeds & \\ref{tab:app-exp1b-r}--\\ref{tab:app-exp1b-diagnostics} \\\\",
        "II-A & VG/OI/PSG observational & Full registered splits & 8 task runs / 7 model--dataset runs & \\ref{tab:app-exp2-coverage}--\\ref{tab:app-exp2-groups} \\\\",
        "II-B & VG controlled interventions & 1,999/2,000 images, 3 seeds & Motifs; SGG Transformer & \\ref{tab:app-exp2b-dose}--\\ref{tab:app-exp2b-endpoints} \\\\",
        "IV & Native SGDet & 26,446/1,813/1,000 images & 11 runs, 9 families; 9 decomposed & \\ref{tab:app-exp4-r}--\\ref{tab:app-exp4-grounding} \\\\",
        "IV-depth & VG PredCls/SGCls/SGDet & 26,446 images & Motifs; SGG Transformer & \\ref{tab:app-exp4-depth-metrics}--\\ref{tab:app-exp4-depth-reference} \\\\",
        "IV auxiliary & VG-150 SGCls & 26,446 images & KERN & \\ref{tab:app-kern-sgcls} \\\\",
        "V & Grounding mitigation & 2 families, 2 modes, 3 seeds & Motifs; SGG Transformer & \\experimenttodo{V} \\\\",
        "V external & Frozen shared-VG transfer & GQA/VRD exact-overlap subsets & Same two selected families & \\experimenttodo{V-ext} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "% Experiment I-A tables",
    ]
    exp1a_tables(root, lines)
    lines.extend(["% Experiment I-B tables"])
    exp1b_tables(root, lines)
    lines.extend(["% Experiment II-A tables"])
    exp2_tables(root, lines)
    lines.extend(["% Experiment II-B tables"])
    exp2_live_tables(root, lines)
    lines.extend(["% Experiment IV tables"])
    exp4_tables(root, lines)
    lines.extend(["% Experiment IV matched-depth tables"])
    exp4_depth_tables(root, lines)
    lines.extend(["% Compact Experiment IV auxiliary tables"])
    exp4_compact_tables(root, lines)
    lines.extend([END, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--main_tex", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Deprecated alias for the default read-only curated-appendix check.",
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        help="Write the legacy generated I--IV block to a separate file for auditing.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    main_tex = (args.main_tex or root / "tex/kdd2027_submission/main.tex").resolve()
    source = main_tex.read_text(encoding="utf-8")

    if args.legacy_output:
        output = args.legacy_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build(root), encoding="utf-8")
        print(f"[legacy export] {output}")
        return

    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise SystemExit("Curated appendix markers are missing or duplicated")
    missing = [
        label
        for label in REQUIRED_CURATED_LABELS
        if f"\\label{{{label}}}" not in source
    ]
    if missing:
        raise SystemExit("Curated appendix is incomplete: " + ", ".join(missing))
    print(
        f"[ok] curated appendix: {len(REQUIRED_CURATED_LABELS)} required tables "
        f"present in {main_tex}"
    )


if __name__ == "__main__":
    main()
