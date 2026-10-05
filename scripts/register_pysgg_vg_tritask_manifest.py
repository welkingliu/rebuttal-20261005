#!/usr/bin/env python3
"""Register one complete PySGG tri-task cache with published references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SPECS = {
    "motifs": {
        "name": "pysgg_motifs_vg_tritask",
        "family": "Neural Motifs",
        "paradigm": "sequential_context",
        "references": {
            "PredCls/R@50": 0.6518, "PredCls/mR@50": 0.1479,
            "SGCls/R@50": 0.3892, "SGCls/mR@50": 0.0828,
            "SGDet/R@50": 0.3278, "SGDet/mR@50": 0.0675,
        },
    },
    "vctree": {
        "name": "pysgg_vctree_vg_tritask",
        "family": "VCTree",
        "paradigm": "tree_context",
        "references": {
            "PredCls/R@50": 0.6542, "PredCls/mR@50": 0.1674,
            "SGCls/R@50": 0.4667, "SGCls/mR@50": 0.1181,
            "SGDet/R@50": 0.3193, "SGDet/mR@50": 0.0744,
        },
    },
    "transformer": {
        "name": "pysgg_transformer_vg_tritask",
        "family": "SGG Transformer",
        "paradigm": "transformer_context",
        "references": {
            "PredCls/R@50": 0.6555, "PredCls/mR@50": 0.1630,
            "SGCls/R@50": 0.4018, "SGCls/mR@50": 0.1009,
            "SGDet/R@50": 0.3304, "SGDet/mR@50": 0.0813,
        },
    },
    "bgnn": {
        "name": "pysgg_bgnn_vg_tritask",
        "family": "BGNN",
        "paradigm": "bipartite_graph_message_passing",
        "references": {
            "SGDet/R@50": 0.2980, "SGDet/mR@50": 0.1090,
        },
    },
    "tde_motifs": {
        "name": "pysgg_tde_motifs_vg_tritask",
        "family": "TDE-Motifs",
        "paradigm": "causal_debiased_context",
        "references": {
            "SGDet/R@50": 0.1940, "SGDet/mR@50": 0.0930,
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model", required=True, choices=sorted(SPECS))
    parser.add_argument("--cache_root")
    parser.add_argument("--tasks", nargs="+", choices=("predcls", "sgcls", "sgdet"),
                        default=["predcls", "sgcls", "sgdet"])
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    spec = SPECS[args.model]
    cache = Path(
        args.cache_root
        or root / "artifacts/prediction_cache" / ("pysgg_" + args.model + "_vg_tritask")
    ).resolve()
    tasks = tuple(dict.fromkeys(args.tasks))
    states = {
        task: json.loads((cache / ("state_" + task + ".json")).read_text())
        for task in tasks
    }
    source = root / "external/official_repos/PySGG"
    marker = json.loads((source / ".official_source.json").read_text())
    output = Path(
        args.output
        or root / "checkpoints/sgg/manifests" / ("pysgg_" + args.model + "_vg_tritask.json")
    ).resolve()
    command = [
        sys.executable, str(root / "scripts/create_prediction_cache_manifest.py"),
        "--cache_root", str(cache),
        "--name", (
            spec["name"] if set(tasks) == {"predcls", "sgcls", "sgdet"}
            else "pysgg_" + args.model + "_vg_" + "_".join(tasks) + "_official"
        ),
        "--family", spec["family"],
        "--paradigm", spec["paradigm"],
        "--source_url", marker["repository_url"],
        "--source_root", str(source),
        "--source_commit", marker["commit"],
        "--training_dataset", "VG-150",
        "--reference_dataset", "vg",
        "--metric_scale", "fraction",
        "--reproduction_tolerance", "0.02",
        "--training_seed", "666",
        "--baseline_mr", str(spec["references"].get("SGDet/mR@50", 0.0)),
        "--output", str(output),
    ]
    for task, state in states.items():
        command.extend(["--checkpoint", task + "=" + state["checkpoint"]])
    task_prefixes = {task.lower() for task in tasks}
    for name, value in spec["references"].items():
        if name.split("/", 1)[0].lower() in task_prefixes:
            command.extend(["--reference_metric", name + "=" + str(value)])
    subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
