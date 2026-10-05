#!/usr/bin/env python3
"""Register a finalized Causal Motifs-SUM or KERN prediction cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CAUSAL_REFERENCE = {
    "none": {
        "predcls": {"R@50": 66.11, "mR@50": 14.60, "zR@50": 11.02},
        "sgcls": {"R@50": 39.25, "mR@50": 8.02, "zR@50": 2.18},
        "sgdet": {"R@50": 32.45, "mR@50": 5.83, "zR@50": 0.08},
    },
    "tde": {
        "predcls": {"R@50": 45.88, "mR@50": 24.75, "zR@50": 14.31},
        "sgcls": {"R@50": 26.31, "mR@50": 13.21, "zR@50": 2.95},
        "sgdet": {"R@50": 16.56, "mR@50": 8.94, "zR@50": 2.33},
    },
}
KERN_REFERENCE = {
    "predcls": {"mR@50": 17.7, "mR@100": 19.2},
    "sgcls": {"mR@50": 9.4, "mR@100": 10.0},
    "sgdet": {"mR@50": 6.4, "mR@100": 7.3},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--model", choices=("causal_motifs_sum", "kern"), required=True)
    parser.add_argument("--effect_type", choices=("none", "tde"), default="none")
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    cache = Path(args.cache_root).expanduser().resolve()
    metadata = json.loads((cache / "metadata.json").read_text())
    tasks = metadata["tasks"]
    states = {
        task: json.loads((cache / f"state_{task}.json").read_text())
        for task in tasks
    }

    if args.model == "causal_motifs_sum":
        name = f"causal_motifs_sum_{args.effect_type}_official"
        family = "causal_motifs_sum"
        paradigm = "two_stage_causal_context"
        source_url = "https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch"
        source_root = root / "external/official_repos/Scene-Graph-Benchmark.pytorch"
        references = CAUSAL_REFERENCE[args.effect_type]
        default_output = root / "checkpoints/sgg/manifests" / f"{name}_vg.json"
    else:
        name = "kern_official"
        family = "kern"
        paradigm = "knowledge_embedded_routing"
        source_url = "https://github.com/yuweihao/KERN"
        source_root = root / "external/official_repos/KERN"
        references = KERN_REFERENCE
        default_output = root / "checkpoints/sgg/manifests/kern_official_vg.json"
    if metadata["model_name"] != name:
        raise RuntimeError(
            f"Cache model identity mismatch: {metadata['model_name']} != {name}"
        )

    control_task = "sgdet" if "sgdet" in tasks else tasks[-1]
    baseline_mr = float(references[control_task]["mR@50"]) / 100.0

    command = [
        sys.executable, str(root / "scripts/create_prediction_cache_manifest.py"),
        "--cache_root", str(cache),
        "--name", name,
        "--family", family,
        "--paradigm", paradigm,
        "--source_url", source_url,
        "--source_root", str(source_root),
        "--source_commit", metadata["source_commit"],
        "--training_dataset", "VG-150",
        "--reference_dataset", "vg",
        "--metric_scale", "percent",
        "--baseline_mr", str(baseline_mr),
        "--reproduction_tolerance", "2.0",
        "--reference_eval_images", "26446",
        "--output", str(Path(args.output).expanduser().resolve() if args.output else default_output),
    ]
    for task in tasks:
        command.extend(["--checkpoint", f"{task}={states[task]['checkpoint']}"])
        if task not in references:
            raise RuntimeError(f"No published reference metrics for {task}")
        for metric, value in references[task].items():
            command.extend(["--reference_metric", f"{task}/{metric}={value}"])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
