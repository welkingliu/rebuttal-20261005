#!/usr/bin/env python3
"""Clone pinned official repositories and download Experiment-4 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone

from official_model_catalog import CORE_MODELS, MODELS, REPOSITORIES, SURVEY_REPOSITORIES


def _run(command, timeout=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, timeout=timeout)


def _git_output(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def _normalise_git_url(value):
    normalised = str(value).lower()
    if normalised.endswith(".git"):
        normalised = normalised[:-4]
    return normalised.rstrip("/")


def _archive_marker(destination: Path):
    return destination / ".official_source.json"


def _read_archive_marker(destination: Path):
    marker = _archive_marker(destination)
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _valid_archive_checkout(destination: Path, spec: dict):
    marker = _read_archive_marker(destination)
    if not (
        marker
        and marker.get("commit", "").lower() == spec["commit"].lower()
        and marker.get("source_url") == spec.get("archive_url")
        and marker.get("archive_sha256")
    ):
        return False
    archive_path = Path(marker.get("archive_path", ""))
    return bool(
        archive_path.is_file()
        and sha256_file(archive_path) == marker["archive_sha256"]
    )


def _move_incomplete(destination: Path):
    if not destination.exists():
        return
    backup = destination.with_name(
        f"{destination.name}.incomplete_{int(time.time())}"
    )
    print(f"[repository] moving incomplete checkout to {backup}", flush=True)
    destination.replace(backup)


def _prepare_git_checkout(destination: Path, spec: dict, timeout: int):
    destination.mkdir(parents=True, exist_ok=False)
    try:
        _run(["git", "init", destination], timeout=timeout)
        _run(["git", "-C", destination, "remote", "add", "origin", spec["url"]], timeout=timeout)
        _run([
            "git", "-c", "http.version=HTTP/1.1", "-C", destination,
            "fetch", "--depth", "1", "--no-tags", "origin", spec["commit"],
        ], timeout=timeout)
        _run(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"], timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        _move_incomplete(destination)
        raise


def _safe_extract_archive(archive: Path, destination: Path, spec: dict):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="official_repo_", dir=destination.parent) as tmp:
        tmp_root = Path(tmp).resolve()
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                target = (tmp_root / member.name).resolve()
                if tmp_root not in target.parents and target != tmp_root:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
                if member.issym() or member.islnk():
                    link = Path(member.linkname)
                    if link.is_absolute():
                        raise RuntimeError(
                            f"unsafe absolute archive link: {member.name} -> {member.linkname}"
                        )
                    link_target = (
                        target.parent / link if member.issym()
                        else tmp_root / link
                    ).resolve()
                    if tmp_root not in link_target.parents and link_target != tmp_root:
                        raise RuntimeError(
                            f"unsafe archive link: {member.name} -> {member.linkname}"
                        )
            handle.extractall(tmp_root)
        roots = [path for path in tmp_root.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"expected one repository root in {archive}, found {len(roots)}")
        shutil.move(str(roots[0]), str(destination))
    marker = {
        "schema_version": 1,
        "source_type": "official_commit_archive",
        "source_url": spec["archive_url"],
        "repository_url": spec["url"],
        "commit": spec["commit"],
        "archive_sha256": sha256_file(archive),
        "archive_path": str(archive.resolve()),
    }
    _archive_marker(destination).write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )


def _local_archive_candidates(archive_root: Path, key: str, spec: dict):
    return [
        archive_root / f"{key}-{spec['commit']}.tar.gz",
        archive_root / f"{spec['directory']}-{spec['commit']}.tar.gz",
        archive_root / f"{key}.tar.gz",
    ]


def _prepare_archive_checkout(destination: Path, archive_root: Path, key: str, spec: dict, allow_network: bool):
    archive_root.mkdir(parents=True, exist_ok=True)
    candidates = _local_archive_candidates(archive_root, key, spec)
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None and allow_network:
        archive = candidates[0]
        print(f"[repository] downloading pinned archive: {spec['archive_url']}", flush=True)
        _download_url(spec["archive_url"], archive)
    if archive is None:
        expected = " or ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No local archive for {key}. Upload the official commit archive as {expected}"
        )
    if destination.exists():
        _move_incomplete(destination)
    _safe_extract_archive(archive, destination, spec)


def prepare_repository(
    external_root: Path,
    archive_root: Path,
    key: str,
    verify_only: bool,
    transport: str,
    git_timeout: int,
):
    spec = REPOSITORIES[key]
    destination = external_root / spec["directory"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (destination / ".git").is_dir():
        origin = _git_output("-C", destination, "remote", "get-url", "origin")
        commit = _git_output("-C", destination, "rev-parse", "HEAD")
        if _normalise_git_url(origin) != _normalise_git_url(spec["url"]):
            raise RuntimeError(f"origin mismatch for {destination}: {origin}")
        if commit.lower() != spec["commit"].lower():
            if verify_only:
                raise RuntimeError(
                    f"commit mismatch for {destination}: expected {spec['commit']}, got {commit}"
                )
            _run([
                "git", "-c", "http.version=HTTP/1.1", "-C", destination,
                "fetch", "--depth", "1", "--no-tags", "origin", spec["commit"],
            ], timeout=git_timeout)
            _run(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"])
            commit = _git_output("-C", destination, "rev-parse", "HEAD")
        return {
            "path": str(destination.resolve()), "url": spec["url"],
            "commit": commit, "source_type": "git",
        }

    if _valid_archive_checkout(destination, spec):
        marker = _read_archive_marker(destination)
        return {
            "path": str(destination.resolve()), "url": spec["url"],
            "commit": spec["commit"], "source_type": marker["source_type"],
            "archive_sha256": marker["archive_sha256"],
        }
    if verify_only:
        raise FileNotFoundError(f"missing or invalid repository source: {destination}")
    if destination.exists():
        _move_incomplete(destination)

    local_available = any(
        path.is_file() for path in _local_archive_candidates(archive_root, key, spec)
    )
    if transport == "local" or (transport == "auto" and local_available):
        _prepare_archive_checkout(destination, archive_root, key, spec, allow_network=False)
    elif transport in {"auto", "git"}:
        try:
            _prepare_git_checkout(destination, spec, git_timeout)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if transport == "git":
                raise
            print("[repository] shallow git fetch failed; trying official commit archive", flush=True)
            _prepare_archive_checkout(destination, archive_root, key, spec, allow_network=True)
    elif transport == "archive":
        _prepare_archive_checkout(destination, archive_root, key, spec, allow_network=True)
    else:
        raise ValueError(f"unsupported repository transport: {transport}")
    return prepare_repository(
        external_root, archive_root, key, True, transport, git_timeout
    )


def _download_gdrive(file_id: str, destination: Path):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive download requires gdown; run: python -m pip install 'gdown>=5.2,<6'"
        ) from exc
    temporary = destination.with_suffix(destination.suffix + ".part")
    result = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
    if not result or not temporary.is_file():
        raise RuntimeError(f"gdown did not create {temporary}")
    temporary.replace(destination)


def _download_url(url: str, destination: Path):
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint and url.startswith("https://huggingface.co/"):
        url = endpoint + url[len("https://huggingface.co"):]
        print(f"[huggingface] endpoint={endpoint}", flush=True)
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable checkpoint downloads")
    temporary = destination.with_suffix(destination.suffix + ".part")
    _run([
        curl, "--fail", "--location", "--retry", "8", "--retry-all-errors",
        "--connect-timeout", "20", "--max-time", "1800",
        "--continue-at", "-", "--output", temporary, url,
    ])
    temporary.replace(destination)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_weight(weights_root: Path, name: str, verify_only: bool):
    spec = MODELS[name]
    destination = weights_root / spec["relative_path"]
    if not destination.is_file():
        if spec.get("download_status", "available") != "available":
            raise FileNotFoundError(
                f"checkpoint requires manual verified source: {destination}. "
                f"{spec.get('download_note', '')}"
            )
        if verify_only:
            raise FileNotFoundError(f"missing checkpoint: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        download = spec["download"]
        print(f"\n[download] {name} -> {destination}", flush=True)
        if download["kind"] == "gdrive":
            _download_gdrive(download["id"], destination)
        elif download["kind"] == "url":
            _download_url(download["url"], destination)
        else:
            raise ValueError(f"unsupported download kind: {download['kind']}")
    size = destination.stat().st_size
    if size < 1024 * 1024:
        raise RuntimeError(f"checkpoint is suspiciously small ({size} bytes): {destination}")
    digest = sha256_file(destination)
    expected = spec.get("sha256")
    if expected and digest.lower() != expected.lower():
        raise RuntimeError(
            f"checkpoint SHA256 mismatch for {name}: expected {expected}, got {digest}"
        )
    required_files = []
    for relative_path in spec.get("required_paths", []):
        required = weights_root / relative_path
        if not required.is_file():
            raise FileNotFoundError(f"missing companion asset for {name}: {required}")
        required_files.append(str(required.resolve()))
    return {
        "path": str(destination.resolve()),
        "bytes": size,
        "sha256": digest,
        "dataset": spec["dataset"],
        "architecture": spec["architecture"],
        "supported_tasks": spec["supported_tasks"],
        "repository": spec["repository"],
        "config": spec.get("config"),
        "reference_metrics": spec.get("reference_metrics", {}),
        "required_files": required_files,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--models", nargs="+", default=list(CORE_MODELS), choices=sorted(MODELS))
    parser.add_argument(
        "--repositories", nargs="+", choices=sorted(REPOSITORIES),
        help="Pinned repositories to fetch; defaults to repositories required by --models.",
    )
    parser.add_argument(
        "--survey_repositories", action="store_true",
        help="Fetch every official repository represented in the 19-family survey panel.",
    )
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--repos_only", action="store_true")
    parser.add_argument("--weights_only", action="store_true")
    parser.add_argument(
        "--repository_transport", choices=["auto", "git", "archive", "local"],
        default="auto",
        help="Repository acquisition method. local reads external/official_archives only.",
    )
    parser.add_argument("--git_timeout", type=int, default=180)
    parser.add_argument("--repo_archive_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repos_only and args.weights_only:
        raise SystemExit("--repos_only and --weights_only are mutually exclusive")
    project_root = Path(args.project_root).expanduser().resolve()
    external_root = project_root / "external" / "official_repos"
    archive_root = Path(args.repo_archive_dir).expanduser().resolve() if args.repo_archive_dir else project_root / "external" / "official_archives"
    weights_root = project_root / "checkpoints" / "sgg" / "weights"
    inventory_path = project_root / "checkpoints" / "sgg" / "download_inventory.json"

    if args.survey_repositories:
        selected_repositories = list(SURVEY_REPOSITORIES)
    elif args.repositories:
        selected_repositories = list(dict.fromkeys(args.repositories))
    else:
        selected_repositories = sorted({MODELS[name]["repository"] for name in args.models})
    repositories = {}
    weights = {}
    if not args.weights_only:
        for key in selected_repositories:
            print(f"\n[repository] {key}", flush=True)
            repositories[key] = prepare_repository(
                external_root, archive_root, key, args.verify_only,
                args.repository_transport, args.git_timeout,
            )
    if not args.repos_only:
        for name in args.models:
            weights[name] = prepare_weight(weights_root, name, args.verify_only)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "repositories": repositories,
        "weights": weights,
        "paper_manifest_ready": False,
        "note": "Downloads are assets only; official adapters and manifests must still pass validation.",
    }
    if not args.verify_only:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ASSETS READY] repositories={len(repositories)} weights={len(weights)}")
    if not args.verify_only:
        print(f"inventory={inventory_path}")
    print("[NOT PAPER READY] Build and validate official adapters/manifests before Experiment 4.")


if __name__ == "__main__":
    try:
        main()
    except (
        OSError, RuntimeError, tarfile.TarError,
        subprocess.CalledProcessError, subprocess.TimeoutExpired,
    ) as exc:
        print(f"\n[FAILED] {exc}", file=sys.stderr)
        raise SystemExit(1)
