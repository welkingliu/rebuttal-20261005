"""Survey model-panel metadata and formal counting rules."""

from __future__ import annotations

import json
from pathlib import Path


PANEL_PATH = Path(__file__).with_name("model_panel.json")


def load_model_panel(path: str | Path = PANEL_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        panel = json.load(handle)
    models = panel.get("models", [])
    families = [str(item.get("family", "")).strip() for item in models]
    if not models or any(not family for family in families):
        raise ValueError("Every model-panel entry requires a non-empty family")
    if len(families) != len(set(families)):
        raise ValueError("Model-panel families must be unique")
    if int(panel.get("candidate_families", -1)) != len(models):
        raise ValueError("candidate_families does not match the model list")
    return panel


def panel_summary(path: str | Path = PANEL_PATH) -> dict:
    panel = load_model_panel(path)
    by_dataset = {name: [] for name in ("vg", "oi", "gqa", "psg", "vrd")}
    for model in panel["models"]:
        for dataset in model["native_datasets"]:
            if dataset in by_dataset:
                by_dataset[dataset].append(model["family"])
    return {
        "candidate_families": len(panel["models"]),
        "formal_target_families": int(panel["formal_target_families"]),
        "recommended_target_families": int(
            panel.get("recommended_target_families", panel["formal_target_families"])
        ),
        "survey_literature_target_families": int(
            panel.get("survey_literature_target_families", 10)
        ),
        "native_family_coverage": by_dataset,
    }


if __name__ == "__main__":
    print(json.dumps(panel_summary(), indent=2))
