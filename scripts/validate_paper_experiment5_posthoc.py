#!/usr/bin/env python3
"""Validate the selected VG test and six-state GQA/VRD transfer outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posthoc_root", type=Path, required=True)
    args = parser.parse_args()
    root = args.posthoc_root.expanduser().resolve()
    checks = {}
    for key in ("baseline", "selected"):
        payload = _json(root / "test" / f"{key}.json")
        checks[f"test_{key}"] = (
            payload.get("schema") == "experiment5_checkpoint_test_v1"
            and payload.get("test_samples") == 26446
            and set(payload.get("tasks", [])) == {"predcls", "sgcls", "sgdet"}
        )
    for dataset in ("gqa", "vrd"):
        base = _json(root / "external" / dataset / "base" / "summary.json")
        checks[f"{dataset}_base"] = (
            base.get("schema_version") == "experiment_5_external_shared_vg_v1"
            and base.get("mitigation_state") is None
        )
        for mode in ("supervised_control", "grounding"):
            for seed in (17, 23, 31):
                payload = _json(
                    root
                    / "external"
                    / dataset
                    / f"{mode}_seed_{seed}"
                    / "summary.json"
                )
                state = payload.get("mitigation_state") or {}
                checks[f"{dataset}_{mode}_{seed}"] = (
                    payload.get("schema_version")
                    == "experiment_5_external_shared_vg_v1"
                    and state.get("training_mode") == mode
                    and state.get("seed") == seed
                )
    status = "pass" if all(checks.values()) else "fail"
    report = {"status": status, "checks": checks}
    destination = root / "validation_report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{status.upper()}] Experiment V posthoc contract: {destination}")
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
