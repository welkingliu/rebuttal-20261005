#!/usr/bin/env python3
"""Report external checkpoint readiness without loading GPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from official_model_catalog import CORE_MODELS, MODELS, REPOSITORIES, SURVEY_REPOSITORIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgg_core.submission_protocol import GLOBAL_MODEL_FAMILY_TARGET


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--models", nargs="+", default=list(CORE_MODELS), choices=sorted(MODELS))
    parser.add_argument("--require_manifests", action="store_true")
    parser.add_argument("--survey_repositories", action="store_true")
    parser.add_argument(
        "--minimum_model_families", type=int,
        default=GLOBAL_MODEL_FAMILY_TARGET,
    )
    parser.add_argument(
        "--manifests_only", action="store_true",
        help="Check the integrated manifest matrix without requiring the six auto-download assets.",
    )
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    failures = []

    panel_path = root / "sgg_core" / "models" / "model_panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    asset_families = sorted({MODELS[name]["architecture"] for name in args.models})
    print("Model-count contract")
    print("=" * 72)
    print(f"  survey_candidates={len(panel['models'])}")
    print(f"  formal_target_families={args.minimum_model_families}")
    print(f"  auto_download_assets={len(args.models)} families={len(asset_families)}")
    print("  integrated_count is determined from valid manifests below")

    if not args.manifests_only:
        print("\nOfficial repository check")
        print("=" * 72)
        repository_keys = (
            SURVEY_REPOSITORIES if args.survey_repositories
            else sorted({MODELS[name]["repository"] for name in args.models})
        )
        for key in repository_keys:
            spec = REPOSITORIES[key]
            path = root / "external" / "official_repos" / spec["directory"]
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            marker_path = path / ".official_source.json"
            marker = None
            if marker_path.is_file():
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker = None
            archive_path = Path(marker.get("archive_path", "")) if marker else None
            archive_ok = bool(
                marker
                and marker.get("commit") == spec["commit"]
                and marker.get("source_url") == spec.get("archive_url")
                and marker.get("archive_sha256")
                and archive_path.is_file()
                and _sha256(archive_path) == marker.get("archive_sha256")
            )
            ok = commit == spec["commit"] or archive_ok
            print(f"  [{'ok' if ok else 'miss'}] {key}: {path}")
            if commit:
                print(f"         commit={commit}")
            elif archive_ok:
                print(f"         commit={marker['commit']} source=official_commit_archive")
                print(f"         archive_sha256={marker['archive_sha256']}")
            if not ok:
                failures.append(f"repository:{key}")

        print("\nCheckpoint check")
        print("=" * 72)
        for name in args.models:
            spec = MODELS[name]
            path = root / "checkpoints" / "sgg" / "weights" / spec["relative_path"]
            required = [
                root / "checkpoints" / "sgg" / "weights" / relative
                for relative in spec.get("required_paths", [])
            ]
            ok = (
                path.is_file()
                and path.stat().st_size >= 1024 * 1024
                and all(companion.is_file() for companion in required)
            )
            print(f"  [{'ok' if ok else 'miss'}] {name}: {path}")
            if ok:
                print(f"         bytes={path.stat().st_size:,} sha256={_sha256(path)}")
                for companion in required:
                    print(f"         companion={companion}")
            else:
                for companion in required:
                    if not companion.is_file():
                        print(f"         missing companion={companion}")
                failures.append(f"checkpoint:{name}")

    manifest_dir = root / "checkpoints" / "sgg" / "manifests"
    manifests = sorted(manifest_dir.glob("*.json")) if manifest_dir.is_dir() else []
    print("\nPaper manifest check")
    print("=" * 72)
    print(f"  manifests={len(manifests)} directory={manifest_dir}")
    valid_json = 0
    manifest_families = set()
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            family = str(payload.get("architecture_family", "")).strip()
            if not family:
                raise ValueError("missing architecture_family")
            source_root = Path(str(payload.get("source_root", ""))).expanduser()
            if not source_root.is_dir():
                raise ValueError(f"missing source_root: {source_root}")
            environment_python = Path(
                str(payload.get("environment_python") or sys.executable)
            ).expanduser()
            if not environment_python.is_file():
                raise ValueError(f"missing environment_python: {environment_python}")
            checkpoints = payload.get("checkpoints", {})
            if not checkpoints:
                raise ValueError("missing task-specific checkpoints")
            missing_weights = [
                str(Path(spec.get("path", "")).expanduser())
                for spec in checkpoints.values()
                if not Path(spec.get("path", "")).expanduser().is_file()
            ]
            if missing_weights:
                raise ValueError(f"missing checkpoints: {missing_weights}")
            valid_json += 1
            manifest_families.add(family)
            print(f"  [json] {path.name} family={family}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"  [bad]  {path.name}: {exc}")
    print(f"  declared_model_runs={valid_json} distinct_families={len(manifest_families)}")
    if args.require_manifests and len(manifest_families) < args.minimum_model_families:
        failures.append(
            f"manifest_families:{len(manifest_families)}/{args.minimum_model_families}"
        )

    if failures:
        print("\n[NOT READY] " + ", ".join(failures))
        raise SystemExit(1)
    print("\n[ASSETS READY]")
    if len(manifest_families) < args.minimum_model_families:
        print(
            "[NOT PAPER READY] Too few distinct official model families: "
            f"{len(manifest_families)}/{args.minimum_model_families}."
        )


if __name__ == "__main__":
    main()
