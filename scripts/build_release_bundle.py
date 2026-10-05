#!/usr/bin/env python3
"""Build and verify the reviewer-facing GroundedSGG-Bench update archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import torch


RELEASE_NAME = "GroundedSGG-Bench_update_20260727"
ROOT_FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "REPRODUCIBILITY.md",
    "RUNNING.md",
    "THIRD_PARTY_ASSETS.md",
    "UPLOAD_CHECKLIST.md",
    "ASSET_SETUP.md",
    "DATASETS.md",
    "FOUNDATION_MODELS.md",
    "INSTALLATION.md",
    "LEGACY_MODEL_SETUP.md",
    "MODELS.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-foundation.txt",
)
SOURCE_DIRS = ("sgg_core", "scripts", "tests")
PAPER_FILES = (
    "tex/kdd2027_submission/main.tex",
    "tex/kdd2027_submission/reference.bib",
    "tex/kdd2027_submission/figures/benchmark_overview.pdf",
    "tex/kdd2027_submission/figures/experiment1a_identity_grounding.pdf",
    "tex/kdd2027_submission/figures/experiment1a_identity_grounding.png",
    "tex/kdd2027_submission/figures/experiment1b_depth_diagnostics.pdf",
    "tex/kdd2027_submission/figures/experiment1b_depth_diagnostics.png",
    "tex/kdd2027_submission/figures/experiment2_error_propagation.pdf",
    "tex/kdd2027_submission/figures/experiment2_error_propagation.png",
    "tex/kdd2027_submission/figures/experiment3_motif_intervention.pdf",
    "tex/kdd2027_submission/figures/experiment3_motif_intervention.png",
    "tex/kdd2027_submission/figures/generate_experiment3_motif_intervention.py",
    "tex/kdd2027_submission/figures/appendix_result_summary.pdf",
    "tex/kdd2027_submission/figures/appendix_result_summary.png",
    "tex/kdd2027_submission/figures/generate_appendix_result_summary.py",
)
RESULT_DIRS = {
    "experiment_1a/psg": (
        "artifacts/experiment_1a/exp1a_refit_zscore_20260722_144126"
    ),
    "experiment_1a/external_gqa": (
        "artifacts/experiment_1a/exp1a_external_gqa_20260722_192417"
    ),
    "experiment_1a/external_vrd": (
        "artifacts/experiment_1a/exp1a_external_vrd_20260722_192008"
    ),
    "experiment_1b": "artifacts/experiment_1b/exp1b_20260716_085021",
    "experiment_2/observational_oi": (
        "artifacts/experiment_2/exp2_observational_scheduled_20260718_210647/oi/oi"
    ),
    "experiment_2/observational_psg": (
        "artifacts/experiment_2/"
        "mac_cpu_queue_20260721_144107_psg_observational_full/psg"
    ),
    "experiment_2/observational_vg": (
        "artifacts/experiment_2/"
        "mac_cpu_queue_20260721_144107_vg_observational_full/vg"
    ),
    "experiment_2/controlled": (
        "artifacts/experiment_2/postcache_submission_20260723_104627_exp2/vg"
    ),
    "experiment_3": "artifacts/experiment_3/formal_gtpairs_20260727",
    "experiment_4/native": (
        "artifacts/experiment_4/exp4_sgdet_submission_fixed_20260718_192834"
    ),
    "experiment_4/matched_depth": (
        "artifacts/experiment_4/postcache_submission_20260723_104627_exp4"
    ),
    "experiment_4/kern_sgcls": (
        "artifacts/experiment_4/mac_cpu_queue_20260721_144107_kern_sgcls_full"
    ),
    "experiment_5": "artifacts/paper_release/formal_20260727",
}
RESULT_NAMES = {
    "summary.json",
    "results.json",
    "experiment_2.json",
    "experiment_3.json",
    "mitigation_results.json",
    "mitigated_state_dict.pth",
    "test_full_split2_entity_aligned.json",
    "formal_before_test_entity_aligned.json",
    "validation_report.json",
}
FORBIDDEN_PATTERNS = {
    "password": re.compile(
        r"(?i)(?:password|passwd|pwd|--password)"
        r"(?:[\"'\s:=]+)"
        + "123"
        + "456"
    ),
    "private_ip": re.compile(r"192\.168\." + r"1\.242"),
    "server_home": re.compile("/home/" + "ccda"),
    "local_home": re.compile("/Users/" + "welkinliu"),
    "external_drive": re.compile("/Volumes/" + "One Touch"),
}
DISALLOWED_SUFFIXES = {
    ".tar",
    ".gz",
    ".tgz",
    ".npz",
    ".npy",
    ".h5",
    ".hdf5",
    ".ckpt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(root: Path, stage: Path, relative: str) -> None:
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(f"Required release file is missing: {source}")
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _ignore_source(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if (
            name in {"__pycache__", ".DS_Store"}
            or name.endswith((".pyc", ".pyo"))
        ):
            ignored.add(name)
    return ignored


def _copy_sources(root: Path, stage: Path) -> None:
    for relative in ROOT_FILES:
        _copy_file(root, stage, relative)
    _copy_file(root, stage, "configs/README.md")
    for relative in SOURCE_DIRS:
        shutil.copytree(
            root / relative,
            stage / relative,
            ignore=_ignore_source,
        )
    for relative in PAPER_FILES:
        _copy_file(root, stage, relative)
    compiled = root / "tex/kdd2027_submission/build_revision/main.pdf"
    if compiled.is_file():
        destination = stage / "paper/manuscript_current.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled, destination)


def _copy_results(root: Path, stage: Path) -> dict[str, list[str]]:
    copied: dict[str, list[str]] = {}
    for release_relative, source_relative in RESULT_DIRS.items():
        source = root / source_relative
        if not source.is_dir():
            raise FileNotFoundError(
                f"Required reported-result directory is missing: {source}"
            )
        destination_root = stage / "results/reported" / release_relative
        rows = []
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.name not in RESULT_NAMES:
                continue
            relative = path.relative_to(source)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            rows.append(str(Path("results/reported") / release_relative / relative))
        if not rows:
            raise RuntimeError(f"No release-eligible results found under {source}")
        copied[release_relative] = rows
    return copied


def _portable_string(value: str) -> str:
    normalized = value.replace("\\", "/")
    normalized = re.sub(
        r"/(?:[^/;\s=]+/)*kdd_sgg_core_experiments/",
        "${PROJECT_ROOT}/",
        normalized,
    )
    formal = "tde_motifs_causal_full_exp5_20260725/"
    if formal in normalized:
        normalized = (
            "${EXPERIMENT5_RUN_ROOT}/"
            + normalized.split(formal, 1)[1]
        )
    normalized = re.sub(
        r"/(?:home|Users|root)/[^;,\s\"']+",
        "${EXTERNAL_RUNTIME}/ASSET",
        normalized,
    )
    return normalized


def _sanitize(value):
    if isinstance(value, str):
        return _portable_string(value)
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    return value


def _sanitize_results(stage: Path) -> None:
    result_root = stage / "results/reported"
    for path in result_root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(
                _sanitize(payload),
                indent=2,
                sort_keys=True,
                allow_nan=True,
            )
            + "\n",
            encoding="utf-8",
        )
    for path in result_root.rglob("mitigated_state_dict.pth"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        torch.save(_sanitize(payload), path)


def _is_text(path: Path) -> bool:
    if path.suffix.lower() in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".pth",
        ".zip",
    }:
        return False
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def _scan(stage: Path) -> None:
    failures = []
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(stage)
        suffix = path.suffix.lower()
        if suffix in DISALLOWED_SUFFIXES:
            failures.append(f"disallowed extension: {relative}")
        if suffix == ".pth" and path.name != "mitigated_state_dict.pth":
            failures.append(f"third-party checkpoint-like file: {relative}")
        if path.stat().st_size > 25 * 1024 * 1024:
            failures.append(f"oversized file: {relative}")
        if _is_text(path):
            text = path.read_text(encoding="utf-8")
            for name, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{name} in {relative}")
    if failures:
        raise RuntimeError("Release scan failed:\n" + "\n".join(failures))


def _write_manifest(stage: Path) -> None:
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha256(path)}  {path.relative_to(stage)}")
    (stage / "MANIFEST.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _verify_manifest(stage: Path) -> None:
    manifest = stage / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("Release has no MANIFEST.sha256")
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = stage / relative
        if not path.is_file() or _sha256(path) != digest:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Checksum mismatch: {failures}")


def _run_smoke(stage: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "sgg_core", "scripts"],
        cwd=stage,
        check=True,
    )
    shell_files = [
        str(path.relative_to(stage))
        for path in (stage / "scripts").glob("*.sh")
    ]
    subprocess.run(["bash", "-n", *shell_files], cwd=stage, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-q",
        ],
        cwd=stage,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(stage),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def _verify_stage(stage: Path, run_smoke: bool = True) -> None:
    _scan(stage)
    _verify_manifest(stage)
    if run_smoke:
        _run_smoke(stage)


def _verify_zip(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        with tempfile.TemporaryDirectory(prefix="grounded_sgg_verify_") as temp:
            archive.extractall(temp)
            roots = [item for item in Path(temp).iterdir() if item.is_dir()]
            if len(roots) != 1:
                raise RuntimeError("ZIP must contain exactly one top-level directory")
            _verify_stage(roots[0])
    print(f"[PASS] Verified release ZIP: {path}")


def _build(root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / RELEASE_NAME
    archive_path = output_dir / f"{RELEASE_NAME}.zip"
    if stage.exists():
        shutil.rmtree(stage)
    if archive_path.exists():
        archive_path.unlink()
    stage.mkdir(parents=True)
    _copy_sources(root, stage)
    result_inventory = _copy_results(root, stage)
    _sanitize_results(stage)
    (stage / "results/RESULT_INVENTORY.json").write_text(
        json.dumps(result_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _scan(stage)
    _write_manifest(stage)
    _verify_stage(stage)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(Path(RELEASE_NAME) / path.relative_to(stage)),
                )
    _verify_zip(archive_path)
    print(f"[PASS] Built release directory: {stage}")
    print(f"[PASS] Built release ZIP: {archive_path}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        _verify_zip(args.verify_only.expanduser().resolve())
        return
    root = args.project_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "dist"
    )
    _build(root, output)


if __name__ == "__main__":
    main()
