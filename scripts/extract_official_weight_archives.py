#!/usr/bin/env python3
"""Safely materialize runtime checkpoints from official model archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone

from official_model_catalog import MODELS


ARCHIVE_MODELS = ("egtr_vg", "egtr_oi")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_regular_member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [
        member for member in archive.getmembers()
        if member.isfile() and member.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one regular archive member ending in {suffix!r}; "
            f"found {[member.name for member in matches]}"
        )
    return matches[0]


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo,
                    destination: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Cannot read archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            shutil.copyfileobj(source, temporary, length=8 * 1024 * 1024)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def prepare_one(weights_root: Path, name: str, verify_only: bool) -> dict:
    spec = MODELS[name]
    archive_path = weights_root / spec["relative_path"]
    checkpoint_path = weights_root / spec["runtime_checkpoint"]
    config_path = weights_root / spec["runtime_config"]
    manifest_path = checkpoint_path.parent / "extraction_manifest.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing official archive: {archive_path}")

    archive_digest = sha256_file(archive_path)
    if not verify_only:
        with tarfile.open(archive_path, "r:gz") as archive:
            checkpoint_member = _unique_regular_member(
                archive, spec["checkpoint_member"]
            )
            config_member = _unique_regular_member(archive, "/config.json")
            _extract_member(archive, checkpoint_member, checkpoint_path)
            _extract_member(archive, config_member, config_path)
        json.loads(config_path.read_text(encoding="utf-8"))
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": name,
            "source_archive": str(archive_path.resolve()),
            "source_archive_sha256": archive_digest,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Missing or suspicious runtime checkpoint: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid EGTR extraction metadata for {name}") from exc
    expected = {
        "source_archive_sha256": archive_digest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
    }
    mismatches = {
        key: {"recorded": payload.get(key), "actual": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Extraction digest mismatch for {name}: {mismatches}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--models", nargs="+", choices=ARCHIVE_MODELS,
                        default=list(ARCHIVE_MODELS))
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    weights_root = root / "checkpoints" / "sgg" / "weights"
    for name in args.models:
        payload = prepare_one(weights_root, name, args.verify_only)
        print(
            f"[ok] {name}: {payload['checkpoint']} "
            f"sha256={payload['checkpoint_sha256']}"
        )
    print(f"[RUNTIME ASSETS READY] models={len(args.models)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"[FAILED] {exc}")
        raise SystemExit(1)
