#!/usr/bin/env python3
"""Check whether downloaded official weights are integrated for experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from official_model_catalog import MODELS, REPOSITORIES


SUBMISSION_ASSETS = (
    "reltr_vg", "reltr_oi", "egtr_vg", "egtr_oi", "openpsg_psgtr_psg",
    "pgsg_vg", "pgsg_psg", "ovsgtr_vg_closed", "ovsgtr_vg_open",
    "bgnn_vg", "bgnn_oi", "sgtr_vg", "sgtr_oi",
    "openpsg_motifs_psg", "openpsg_vctree_psg", "openpsg_psgformer_psg",
)


def _factory_importable(python: Path, project_root: Path, manifest: dict) -> tuple[bool, str]:
    module, separator, function = str(manifest.get("factory", "")).partition(":")
    if not separator:
        return False, "factory must use module:function"
    code = (
        "import importlib,sys,sgg_core; "
        "assert sys.version_info >= (3,10), sys.version; "
        f"m=importlib.import_module({module!r}); "
        f"assert callable(getattr(m,{function!r},None))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [str(python), "-c", code], cwd=project_root,
        env=environment, capture_output=True, text=True,
    )
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--require_foundation", action="store_true")
    parser.add_argument("--check_factory_imports", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    weights_root = root / "checkpoints" / "sgg" / "weights"
    manifest_dir = root / "checkpoints" / "sgg" / "manifests"
    failures, assets, repositories, manifests = [], [], [], []

    for name in SUBMISSION_ASSETS:
        spec = MODELS[name]
        asset = weights_root / spec["relative_path"]
        runtime = (
            weights_root / spec["runtime_checkpoint"]
            if spec.get("runtime_checkpoint") else asset
        )
        companions = [
            weights_root / value for value in spec.get("required_paths", [])
        ]
        if spec.get("runtime_config"):
            companions.append(weights_root / spec["runtime_config"])
        ok = (
            asset.is_file() and asset.stat().st_size >= 1024 * 1024
            and runtime.is_file() and runtime.stat().st_size >= 1024 * 1024
            and all(path.is_file() for path in companions)
        )
        assets.append({"name": name, "ok": ok, "asset": str(asset),
                       "runtime": str(runtime),
                       "companions": [str(path) for path in companions]})
        print(f"[{'ok' if ok else 'miss'}] asset {name}: {runtime}")
        if not ok:
            failures.append(f"asset:{name}")

    needed_repositories = sorted({MODELS[name]["repository"] for name in SUBMISSION_ASSETS})
    for key in needed_repositories:
        spec = REPOSITORIES[key]
        path = root / "external" / "official_repos" / spec["directory"]
        marker = path / ".official_source.json"
        git_dir = path / ".git"
        ok = path.is_dir() and (marker.is_file() or git_dir.is_dir())
        repositories.append({"name": key, "ok": ok, "path": str(path)})
        print(f"[{'ok' if ok else 'miss'}] source {key}: {path}")
        if not ok:
            failures.append(f"source:{key}")

    paths = sorted(manifest_dir.glob("*.json")) if manifest_dir.is_dir() else []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            python = Path(str(payload.get("environment_python", ""))).expanduser()
            source = Path(str(payload.get("source_root", ""))).expanduser()
            checkpoints = payload.get("checkpoints", {})
            checkpoint_ok = bool(checkpoints) and all(
                Path(str(item.get("path", ""))).expanduser().is_file()
                for item in checkpoints.values()
            )
            ok = python.is_file() and source.is_dir() and checkpoint_ok
            factory_detail = "not checked"
            if ok and args.check_factory_imports:
                ok, factory_detail = _factory_importable(python, root, payload)
            manifests.append({"path": str(path), "name": payload.get("name"),
                              "ok": ok, "factory": factory_detail})
            print(f"[{'ok' if ok else 'bad'}] manifest {path.name}")
            if not ok:
                failures.append(f"manifest:{path.name}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            manifests.append({"path": str(path), "ok": False, "error": str(exc)})
            failures.append(f"manifest:{path.name}")
            print(f"[bad] manifest {path.name}: {exc}")
    if not paths:
        failures.append("manifests:0")
        print(f"[miss] manifests: {manifest_dir}")

    foundation = root / "checkpoints" / "foundation"
    foundation_ok = foundation.is_dir()
    print(f"[{'ok' if foundation_ok else 'remote'}] foundation: {foundation}")
    if args.require_foundation and not foundation_ok:
        failures.append("foundation")

    report = {
        "status": "ready" if not failures else "not_ready",
        "assets": assets,
        "repositories": repositories,
        "manifests": manifests,
        "foundation_local": foundation_ok,
        "failures": failures,
    }
    report_path = (
        Path(args.report).expanduser().resolve() if args.report else
        root / "artifacts" / "manifests" / "official_integration.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_path}")
    if failures:
        print("[NOT READY] " + "; ".join(failures))
        raise SystemExit(1)
    print("[READY] Official model integration is complete.")


if __name__ == "__main__":
    main()
