"""Isolated release-source execution and validation-exposure inventory."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import ROOT, OLD, atomic_json, ensure_storage, output_path, sha256
from evidence_completion import read


OUT = ROOT / "results/R21_release_execution"


def exposure():
    universe_file = ROOT / "results/R9_reproduction/sgcls/eval_corrected/val/protocol.json"
    universe = set(map(str, read(universe_file)["image_ids"]))
    candidates = set()
    for pattern in ["cache/V*/*/protocol.json", "results/V*/splits.json", "results/V*/*/splits.json",
                    "results/V*/protocol.json", "results/V*/*/protocol.json", "manifests/V*_protocol.json"]:
        candidates.update(ROOT.glob(pattern))
    explicit = {"gate", "development", "validation", "dev", "selection", "screening", "image_ids",
                "gate_ids", "development_ids", "validation_ids", "dev_ids", "train_ids", "train"}
    records = []; used = set()
    def gather(value, trail=""):
        found = []
        if isinstance(value, dict):
            for k, v in value.items():
                location = trail + "/" + k
                if k in explicit and isinstance(v, list) and all(isinstance(x, (str, int)) for x in v):
                    overlap = set(map(str, v)) & universe
                    if overlap: found.append((location, overlap))
                else: found.extend(gather(v, location))
        return found
    for path in sorted(candidates):
        for key, overlap in gather(read(path)):
            used |= overlap
            records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), key=key,
                                validation_overlap=len(overlap), image_ids=sorted(overlap)))
    return dict(validation_images=len(universe), known_repair_exposed=len(used),
        outside_indexed_repair_protocols=sorted(universe-used), records=records,
        exhaustive_history_audit=False, new_independent_gate_certified=False,
        warning="Outside indexed protocols is not proof of freshness. All native validation images were also used in baseline selection. No new confirmation set is authorized by this audit.")


def main():
    ensure_storage(); start = time.monotonic()
    imported = ROOT / "imported/anonymous_release_20260929"
    sandbox = ROOT / "releases/isolated_execution_20261003"
    sandbox.mkdir(parents=True, exist_ok=True)
    records = []
    for line in (imported / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts: raise RuntimeError("Unsafe release path")
        is_source = name.startswith(("scripts/", "sgg_core/", "configs/", "tests/"))
        is_metadata = len(relative.parts) == 1 and relative.suffix in (".md", ".txt", ".toml", ".yml", ".yaml")
        if not is_source and not is_metadata: continue
        verified = next((p for p in (imported / name, OLD / name) if p.is_file() and sha256(p) == digest), None)
        row = dict(path=name, expected_sha256=digest, status="verified" if verified else "missing_or_mismatched")
        if verified:
            destination = output_path(sandbox / name)
            if not destination.exists(): shutil.copy2(verified, destination)
            if sha256(destination) != digest: raise RuntimeError("Isolated source drift: " + name)
            row["source"] = str(verified)
        records.append(row)
    atomic_json(OUT / "source_inventory.json", dict(files=records, immutable_submitted_release=True))
    env = dict(os.environ, PYTHONPATH=str(sandbox), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               SGG_PROJECT_ROOT=str(sandbox), SGG_ARTIFACT_DIR=str(sandbox / "artifacts"))
    # Existing package environment is reused; source isolation is not a fresh install.
    commands = [
        ("runtime_inventory", [sys.executable, "-c", "import sys,json,torch,numpy; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__)))"]),
        ("unit_suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]),
    ]
    for name in ["experiment_1a", "experiment_1b", "experiment_2", "experiment_3", "experiment_4", "experiment_5"]:
        commands.append((name+"_entrypoint", [sys.executable, "-m", "sgg_core.experiments."+name, "--help"]))
    outcomes = []
    for n, (name, command) in enumerate(commands, 1):
        log = output_path(OUT / "logs" / (name + ".log"))
        with log.open("w") as handle:
            try:
                result = subprocess.run(command, cwd=str(sandbox), env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=3600)
                code = result.returncode
            except subprocess.TimeoutExpired: code = 124
        outcomes.append(dict(name=name, command=command, returncode=code, log=str(log)))
        atomic_json(OUT / "progress.json", dict(stage=name, images=n, total=len(commands), seconds=time.monotonic()-start))
        print(json.dumps(outcomes[-1]), flush=True)
    atomic_json(OUT / "validation_exposure.json", exposure())
    atomic_json(OUT / "summary.json", dict(status="complete", commands=outcomes,
        all_commands_passed=all(x["returncode"] == 0 for x in outcomes),
        missing_metadata=[r for r in records if r["status"] != "verified"], source_root=str(sandbox),
        existing_dependency_environment=True, clean_install_claim=False, full_workflow_claim=False,
        interpretation="Source-isolated offline unit and CLI tests, not a replay of all native models or full datasets",
        elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__": main()
