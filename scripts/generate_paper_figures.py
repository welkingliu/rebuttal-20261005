#!/usr/bin/env python3
"""Generate manuscript figures from provenance-validated experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "tex" / "kdd2027_submission" / "figures"
EXP1A_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment_1a"
    / "exp1a_refit_zscore_20260722_144126"
)
EXP1B_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment_1b"
    / "exp1b_20260716_085021"
)
EXP2_VG_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment_2"
    / "mac_cpu_queue_20260721_144107_vg_observational_full"
    / "vg"
)
EXP2_OI_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment_2"
    / "exp2_observational_scheduled_20260718_210647"
    / "oi"
    / "oi"
)
EXP2_PSG_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment_2"
    / "mac_cpu_queue_20260721_144107_psg_observational_full"
    / "psg"
)

MODEL_ORDER = [
    "cradio_v4_so400m",
    "dinov2_b",
    "radio_v25_b",
    "siglip2_b",
    "sam_vit_b",
    "resnet50",
]
MODEL_LABEL = {
    "cradio_v4_so400m": "C-RADIOv4",
    "dinov2_b": "DINOv2-B",
    "radio_v25_b": "RADIOv2.5-B",
    "siglip2_b": "SigLIP2-B",
    "sam_vit_b": "SAM ViT-B",
    "resnet50": "ResNet-50",
}

COLORS = {
    "navy": "#24507A",
    "teal": "#16847A",
    "orange": "#D07A2D",
    "red": "#A33A3A",
    "gray": "#68737D",
    "light_gray": "#D5DADF",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#E8EBED", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)
    plt.close(figure)


def experiment_1a_figure() -> None:
    records = []
    for model in MODEL_ORDER:
        summary = load_json(EXP1A_ROOT / model / "summary.json")
        views = summary["views"]
        endpoint_runs = [
            run["relationship_endpoints"]["conditioned_on_both_endpoint_mask_iou"]
            ["both_iou>=0.85"]
            for run in views["pred_mask"]["runs"]
        ]
        view_intervals = {}
        for view_name in ("box", "pred_mask", "gt_mask"):
            run_intervals = [
                run["metrics"]["bootstrap_95ci"]["top1_accuracy"]
                for run in views[view_name]["runs"]
            ]
            view_intervals[view_name] = (
                min(interval[0] for interval in run_intervals),
                max(interval[1] for interval in run_intervals),
            )
        records.append(
            {
                "model": model,
                "box": views["box"]["aggregate"]["top1_accuracy"]["mean"],
                "box_lo": view_intervals["box"][0],
                "box_hi": view_intervals["box"][1],
                "gt_mask": views["gt_mask"]["aggregate"]["top1_accuracy"]["mean"],
                "gt_mask_lo": view_intervals["gt_mask"][0],
                "gt_mask_hi": view_intervals["gt_mask"][1],
                "pred_mask": views["pred_mask"]["aggregate"]["top1_accuracy"]["mean"],
                "pred_mask_lo": view_intervals["pred_mask"][0],
                "pred_mask_hi": view_intervals["pred_mask"][1],
                "failure": np.mean([item["endpoint_failure_rate"] for item in endpoint_runs]),
                "failure_lo": min(item["bootstrap_95ci"][0] for item in endpoint_runs),
                "failure_hi": max(item["bootstrap_95ci"][1] for item in endpoint_runs),
            }
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 3.75),
        gridspec_kw={"height_ratios": [1.08, 0.92], "hspace": 0.62},
    )
    y = np.arange(len(records))
    view_specs = [
        ("box", "Box", COLORS["gray"], "o"),
        ("pred_mask", "Box-prompted SAM", COLORS["teal"], "s"),
        ("gt_mask", "GT mask", COLORS["navy"], "D"),
    ]
    offsets = [-0.16, 0.0, 0.16]
    for (key, label, color, marker), offset in zip(view_specs, offsets):
        values = np.array([row[key] for row in records]) * 100
        lower = np.array([row[f"{key}_lo"] for row in records]) * 100
        upper = np.array([row[f"{key}_hi"] for row in records]) * 100
        axes[0].errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lower, upper - values]),
            fmt=marker,
            markersize=2.6,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    axes[0].set_yticks(y, [MODEL_LABEL[row["model"]] for row in records])
    axes[0].invert_yaxis()
    axes[0].set_xlim(35, 92)
    axes[0].set_xlabel("Object Top-1 accuracy (%)")
    clean_axis(axes[0])

    failures = np.array([row["failure"] for row in records]) * 100
    failure_lo = np.array([row["failure_lo"] for row in records]) * 100
    failure_hi = np.array([row["failure_hi"] for row in records]) * 100
    axes[1].errorbar(
        failures,
        y,
        xerr=np.vstack([failures - failure_lo, failure_hi - failures]),
        fmt="o",
        color=COLORS["red"],
        ecolor=COLORS["red"],
        elinewidth=1.0,
        capsize=2.3,
        markersize=3.1,
        label="Identity disagreement",
        zorder=3,
    )
    axes[1].set_yticks(y, [MODEL_LABEL[row["model"]] for row in records])
    axes[1].invert_yaxis()
    axes[1].set_xlim(5, 65)
    axes[1].set_xlabel("Endpoint identity disagreement (%)")
    clean_axis(axes[1])

    axes[0].legend(
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor=COLORS["light_gray"],
        ncol=1,
        fontsize=5.8,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.25,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        borderaxespad=0.2,
    )
    figure.subplots_adjust(top=0.91, bottom=0.11, left=0.34, right=0.97)
    for axis, title in zip(
        axes,
        (
            "a  Identity by spatial support",
            r"b  Both endpoint masks: IoU $\geq$ 0.85",
        ),
    ):
        figure.text(
            0.5,
            axis.get_position().y1 + 0.025,
            title,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )

    save_figure(figure, "experiment1a_identity_grounding")


def load_experiment_2_records() -> list[tuple[str, dict]]:
    specifications = [
        (EXP2_VG_ROOT / "egtr_vg_official" / "experiment_2.json", "VG / EGTR", "sgdet"),
        (EXP2_VG_ROOT / "kern_official" / "experiment_2.json", "VG / KERN", "sgdet"),
        (EXP2_VG_ROOT / "sgtr_vg_official" / "experiment_2.json", "VG / SGTR", "sgdet"),
        (EXP2_OI_ROOT / "egtr_oi_official" / "experiment_2.json", "OI / EGTR", "sgdet"),
        (EXP2_OI_ROOT / "sgtr_oi_official" / "experiment_2.json", "OI / SGTR", "sgdet"),
        (EXP2_PSG_ROOT / "openpsg_motifs_psg_official" / "experiment_2.json", "PSG / Motifs", "sgdet"),
        (EXP2_PSG_ROOT / "openpsg_vctree_psg_official" / "experiment_2.json", "PSG / VCTree", "sgdet"),
    ]
    records = []
    for path, label, task_name in specifications:
        payload = load_json(path)
        model = next(iter(payload["object_error_propagation"]))
        task = payload["object_error_propagation"][model]["tasks"][task_name]
        records.append((label, task["groups"]))
    return records


def experiment_2_figure() -> None:
    records = load_experiment_2_records()

    figure, axis = plt.subplots(figsize=(3.35, 3.15))
    y = np.arange(len(records))
    groups = [
        ("both_correct", "Both match", COLORS["teal"], "o", -0.18),
        ("one_wrong", "One differs", COLORS["orange"], "s", 0.0),
        ("both_wrong", "Both differ", COLORS["red"], "D", 0.18),
    ]
    for key, legend, color, marker, offset in groups:
        values = np.array([row[1][key]["relation_Hit@1"] for row in records]) * 100
        lower = np.array([row[1][key]["bootstrap_95ci_Hit@1"][0] for row in records]) * 100
        upper = np.array([row[1][key]["bootstrap_95ci_Hit@1"][1] for row in records]) * 100
        axis.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lower, upper - values]),
            fmt=marker,
            markersize=3.0,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            label=legend,
            zorder=3,
        )
    axis.set_yticks(y, [row[0] for row in records])
    axis.invert_yaxis()
    axis.set_ylim(len(records) - 0.5, -1.25)
    # Leave enough room for the lowest bootstrap bound (VG/SGTR, both differ).
    # Clipping that interval would visually understate its uncertainty.
    axis.set_xlim(25, 95)
    axis.set_xlabel("Predicate Hit@1 (%)")
    for boundary in (2.5, 4.5):
        axis.axhline(boundary, color=COLORS["light_gray"], linewidth=0.8, zorder=1)
    clean_axis(axis)
    figure.suptitle(
        "Predicate accuracy by endpoint agreement",
        x=0.56,
        y=0.94,
        ha="center",
        fontweight="bold",
        fontsize=8.2,
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        borderaxespad=0.0,
        fontsize=5.5,
        columnspacing=0.55,
        handletextpad=0.3,
    )
    figure.subplots_adjust(top=0.86, bottom=0.14, left=0.32, right=0.98)
    save_figure(figure, "experiment2_error_propagation")


def experiment_2_small_multiples_figure() -> None:
    records = load_experiment_2_records()
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for label, groups in records:
        dataset, model = label.split(" / ")
        grouped.setdefault(dataset, []).append((model, groups))

    group_keys = ["both_correct", "one_wrong", "both_wrong"]
    x = np.arange(len(group_keys), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharey=True)
    model_styles = {
        "EGTR": (COLORS["navy"], "o"),
        "KERN": (COLORS["orange"], "^"),
        "SGTR": (COLORS["teal"], "s"),
        "Motifs": (COLORS["navy"], "o"),
        "VCTree": (COLORS["teal"], "s"),
    }
    for panel_index, (axis, dataset) in enumerate(zip(axes, ["VG", "OI", "PSG"])):
        dataset_records = grouped[dataset]
        offsets = np.linspace(-0.07, 0.07, len(dataset_records))
        for (model, groups), offset in zip(dataset_records, offsets):
            color, marker = model_styles[model]
            values = np.array([groups[key]["relation_Hit@1"] for key in group_keys]) * 100
            lower = np.array([groups[key]["bootstrap_95ci_Hit@1"][0] for key in group_keys]) * 100
            upper = np.array([groups[key]["bootstrap_95ci_Hit@1"][1] for key in group_keys]) * 100
            axis.errorbar(
                x + offset,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                marker=marker,
                markersize=4.2,
                linewidth=1.2,
                elinewidth=0.8,
                capsize=1.8,
                color=color,
                label=model,
                zorder=3,
            )
        axis.set_title(dataset, fontweight="bold", pad=22)
        axis.set_xticks(x, ["Both\nmatch", "One\ndiffers", "Both\ndiffer"])
        axis.set_xlim(-0.25, 2.25)
        axis.set_ylim(25, 95)
        axis.grid(axis="y", color="#E8EBED", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if panel_index:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False)
        axis.legend(
            frameon=False,
            ncol=len(dataset_records),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            borderaxespad=0.0,
            fontsize=5.8 if len(dataset_records) == 3 else 6.2,
            columnspacing=0.55,
            handletextpad=0.35,
        )
    axes[0].set_ylabel("Predicate Hit@1 (%)")
    figure.suptitle(
        "Predicate accuracy by endpoint agreement",
        x=0.08,
        y=0.995,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.72, bottom=0.22, left=0.09, right=0.99, wspace=0.08)
    save_figure(figure, "experiment2_error_propagation_small_multiples")


def experiment_1b_figure() -> None:
    palette = ["#24507A", "#16847A", "#D07A2D", "#8E5EA2", "#A33A3A", "#68737D"]
    markers = ["o", "s", "^", "D", "P", "X"]
    depths = [0, 2, 4, 8]
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 4.25), gridspec_kw={"hspace": 0.58})
    for model, color, marker in zip(MODEL_ORDER, palette, markers):
        summary = load_json(EXP1B_ROOT / model / "summary.json")
        mr_means = []
        mr_stds = []
        rank_means = []
        rank_stds = []
        for depth in depths:
            runs = [run for run in summary["runs"] if run["depth"] == depth]
            mr = np.array([run["evaluation"]["metrics"]["mR@50"] for run in runs]) * 100
            ranks = []
            for run in runs:
                diagnostics = run["evaluation"]["layer_diagnostics"]
                final_layer = diagnostics[max(diagnostics, key=lambda item: int(item))]
                ranks.append(final_layer["effective_rank"])
            mr_means.append(mr.mean())
            mr_stds.append(mr.std(ddof=1))
            rank_means.append(np.mean(ranks))
            rank_stds.append(np.std(ranks, ddof=1))
        label = MODEL_LABEL[model]
        axes[0].errorbar(
            depths,
            mr_means,
            yerr=mr_stds,
            marker=marker,
            markersize=3.2,
            linewidth=1.1,
            capsize=1.5,
            color=color,
            label=label,
        )
        axes[1].errorbar(
            depths,
            rank_means,
            yerr=rank_stds,
            marker=marker,
            markersize=3.2,
            linewidth=1.1,
            capsize=1.5,
            color=color,
            label=label,
        )
    axes[0].set_title("a  PredCls mean recall", loc="left", fontweight="bold")
    axes[0].set_ylabel("mR@50 (%)")
    axes[1].set_title("b  Final relation-feature rank", loc="left", fontweight="bold")
    axes[1].set_ylabel("Effective rank")
    for axis in axes:
        axis.set_xlabel("GCN relation depth")
        axis.set_xticks(depths)
        clean_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        columnspacing=0.8,
        handlelength=1.6,
    )
    figure.subplots_adjust(top=0.77, bottom=0.11, left=0.19, right=0.97)
    save_figure(figure, "experiment1b_depth_diagnostics")


def main() -> None:
    configure_style()
    experiment_1a_figure()
    experiment_2_figure()
    experiment_2_small_multiples_figure()
    experiment_1b_figure()
    print(f"figures={FIGURE_DIR}")


if __name__ == "__main__":
    main()
