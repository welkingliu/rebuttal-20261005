"""Compute-aware protocol for the two-RTX-3090 submission run."""

from __future__ import annotations


ALL_DATASETS = ("vg", "oi", "gqa", "psg", "vrd")
STANDARD_BENCHMARK_DATASETS = ("vg", "oi", "psg")
EXTERNAL_DIAGNOSTIC_DATASETS = ("gqa", "vrd")

SURVEY_LITERATURE_FAMILY_TARGET = 10
# SGDet breadth and tri-task depth are separate contracts. The broad panel
# covers at least five families, while the expensive matched PredCls/SGCls/
# SGDet comparison is restricted to one classic and one transformer family.
CORE_VG_MODEL_FAMILIES = (
    "Neural Motifs",
    "VCTree",
    "TDE-Motifs",
    "BGNN",
    "SGG Transformer",
)
CORE_VG_TRITASK_FAMILIES = (
    "Neural Motifs",
    "SGG Transformer",
)
GLOBAL_MODEL_FAMILY_TARGET = len(CORE_VG_MODEL_FAMILIES)
RECOMMENDED_MODEL_FAMILY_TARGET = GLOBAL_MODEL_FAMILY_TARGET
STANDARD_DATASET_FAMILY_TARGETS = {
    "vg": 5,
    "oi": 2,
    "psg": 2,
}
STANDARD_TASK_FAMILY_TARGETS = {
    "vg": {"predcls": 2, "sgcls": 2, "sgdet": 5},
    "oi": {"sgdet": 2},
    "psg": {"sgdet": 2},
}
EXTERNAL_DATASET_MODEL_TARGETS = {
    # Native-manifest targets remain zero: VG checkpoints must not claim
    # native GQA/VRD support. External result coverage has a separate contract.
    "gqa": 0,
    "vrd": 0,
}
EXTERNAL_INFERENCE_FAMILY_TARGETS = {"gqa": 2, "vrd": 2}
EXTERNAL_INFERENCE_MODES = ("supervised_control", "grounding")

STANDARD_MODEL_RANGE = (5, 9)
DIAGNOSTIC_MODEL_FAMILIES = (
    "Neural Motifs",
    "SGG Transformer",
)
DIAGNOSTIC_MODEL_RANGE = (2, 2)
EXPERIMENT_2_FULL_DATASET = "vg"
EXPERIMENT_2_FULL_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)
EXPERIMENT_2_LIGHT_LEVELS = (0.0, 0.5, 1.0)
EXPERIMENT_2_DATASETS = ("vg",)
# Key-node masking and on-manifold replacement are identity-evidence
# interventions. Random/unrelated-node masking and mild color jitter are
# matched structural and label-preserving appearance controls.
EXPERIMENT_2_SWEEP_STRATEGIES = (
    "key_node_mask",
    "random_node_mask",
    "unrelated_node_mask",
    "on_manifold_replacement",
    "color_jitter",
)
EXPERIMENT_3_DATASETS = ("vg",)

EXPERIMENT_4_STEPS = ("standard", "grounding")
MITIGATION_MODEL_FAMILIES = ("Neural Motifs", "SGG Transformer")
MITIGATION_SEEDS = (17, 23, 31)


def parse_dataset_targets(values, defaults: dict[str, int]) -> dict[str, int]:
    """Parse ``dataset=count`` values while rejecting unknown or duplicate keys."""
    if not values:
        return dict(defaults)
    parsed: dict[str, int] = {}
    for value in values:
        dataset, separator, count = str(value).partition("=")
        dataset = dataset.strip().lower()
        if not separator or dataset not in ALL_DATASETS:
            raise ValueError(f"Invalid dataset target: {value!r}")
        if dataset in parsed:
            raise ValueError(f"Duplicate dataset target: {dataset}")
        number = int(count)
        if number < 0:
            raise ValueError(f"Dataset target must be non-negative: {value!r}")
        parsed[dataset] = number
    return parsed


def format_dataset_targets(targets: dict[str, int]) -> list[str]:
    return [f"{dataset}={targets[dataset]}" for dataset in targets]
