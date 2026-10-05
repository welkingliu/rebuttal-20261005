#!/usr/bin/env python3
"""Download, verify, smoke-test, and package the main foundation backbones."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


MAIN_MODELS = (
    "resnet50", "dinov2_b", "siglip2_b", "radio_v25_b",
    "cradio_v4_so400m", "sam_vit_b",
)
SUPPORTED_MODELS = MAIN_MODELS + ("dinov3_b",)
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_WEIGHT_SHA256 = (
    "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
)
RADIO_COMMIT = "c0f37017930e9dda53f93424cf4bf39fc51f287e"
RADIO_HF_REVISION = "10f0448935988a74dd59b4969ac520dbcd7db293"
RADIO_WEIGHT_SHA256 = (
    "6bff4bd732d815136652d454598e8fe6c6f4e658716d5e6a697f0e6b60bd8a98"
)
HF_REVISIONS = {
    "dinov3_b": (
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "5931719e67bbdb9737e363e781fb0c67687896bc",
    ),
    "siglip2_b": (
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ),
    "cradio_v4_so400m": (
        "nvidia/C-RADIOv4-SO400M",
        "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
    ),
    "sam_vit_b": (
        "facebook/sam-vit-base",
        "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
    ),
}
CRADIO_CODE_FILES = (
    "adaptor_attn.py", "adaptor_base.py", "adaptor_generic.py",
    "adaptor_mlp.py", "adaptor_module_factory.py", "adaptor_registry.py",
    "cls_token.py", "common.py", "dinov2_arch.py", "dual_hybrid_vit.py",
    "enable_cpe_support.py", "enable_damp.py", "enable_spectral_reparam.py",
    "eradio_model.py", "extra_models.py", "extra_timm_models.py",
    "feature_normalizer.py", "forward_intermediates.py", "hf_model.py",
    "input_conditioner.py", "open_clip_adaptor.py", "radio_model.py",
    "siglip2_adaptor.py", "utils.py", "vit_patch_generator.py", "vitdet.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, expected_sha256: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if expected_sha256 is None or sha256(destination) == expected_sha256:
            print(f"[cached] {destination}", flush=True)
            return
        raise RuntimeError(f"Checksum mismatch for existing file: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            downloaded = temporary.stat().st_size if temporary.is_file() else 0
            headers = {"User-Agent": "kdd-sgg-foundation-preparer/1.0"}
            if downloaded:
                headers["Range"] = f"bytes={downloaded}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = downloaded > 0 and getattr(response, "status", 200) == 206
                mode = "ab" if append else "wb"
                if not append:
                    downloaded = 0
                with open(temporary, mode) as output:
                    last_report = downloaded
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 64 * 1024 * 1024:
                            print(
                                f"[download] {destination.name}: "
                                f"{downloaded / 1024**2:.0f} MiB",
                                flush=True,
                            )
                            last_report = downloaded
            if expected_sha256 and sha256(temporary) != expected_sha256:
                raise RuntimeError(f"Checksum mismatch after download: {url}")
            temporary.replace(destination)
            print(f"[downloaded] {destination}", flush=True)
            return
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == 5:
                raise
            print(f"[retry {attempt}/5] {url}: {exc}", flush=True)
            time.sleep(attempt * 3)


def _download_repo(owner_repo: str, commit: str, destination: Path) -> None:
    marker = destination / ".sgg_source.json"
    if marker.is_file() and (destination / "hubconf.py").is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("commit") == commit:
            print(f"[cached] {destination} commit={commit}", flush=True)
            return
    if destination.exists():
        raise RuntimeError(
            f"Repository directory exists without the requested pin: {destination}. "
            "Move it aside before downloading."
        )

    url = f"https://github.com/{owner_repo}/archive/{commit}.zip"
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_dir:
        temporary_root = Path(temporary_dir)
        archive = temporary_root / "repo.zip"
        _download_file(url, archive, expected_sha256=None)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temporary_root / "extract")
        extracted = [
            path for path in (temporary_root / "extract").iterdir()
            if path.is_dir()
        ]
        if len(extracted) != 1 or not (extracted[0] / "hubconf.py").is_file():
            raise RuntimeError(f"Unexpected repository archive layout: {url}")
        shutil.move(str(extracted[0]), str(destination))
    marker.write_text(
        json.dumps(
            {"source": f"https://github.com/{owner_repo}", "commit": commit},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[downloaded] {destination} commit={commit}", flush=True)


def paths_for(root: Path) -> dict[str, dict[str, Path]]:
    foundation = root / "checkpoints" / "foundation"
    return {
        "resnet50": {
            "root": foundation / "torch_hub" / "hub" / "checkpoints",
        },
        "dinov2_b": {
            "repo": root / "external" / "foundation_repos" / "dinov2",
            "weights": foundation / "dinov2" / "dinov2_vitb14_pretrain.pth",
        },
        "dinov3_b": {
            "model": foundation / "hf_models" / "dinov3_b",
        },
        "siglip2_b": {
            "model": foundation / "hf_models" / "siglip2_b",
        },
        "cradio_v4_so400m": {
            "model": foundation / "hf_models" / "cradio_v4_so400m",
        },
        "sam_vit_b": {
            "model": foundation / "hf_models" / "sam_vit_b",
        },
        "radio_v25_b": {
            "repo": root / "external" / "foundation_repos" / "radio",
            "weights": foundation / "radio" / "radio-v2.5-b_half.pth.tar",
        },
    }


def download_model(
    model: str,
    root: Path,
    token: str | None,
    hf_endpoint: str = DEFAULT_HF_ENDPOINT,
) -> None:
    hf_endpoint = hf_endpoint.rstrip("/")
    paths = paths_for(root)[model]
    if model == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        print("[download] torchvision ResNet-50", flush=True)
        instance = resnet50(weights=ResNet50_Weights.DEFAULT)
        del instance
        return
    if model == "dinov2_b":
        paths["repo"].parent.mkdir(parents=True, exist_ok=True)
        _download_repo("facebookresearch/dinov2", DINOV2_COMMIT, paths["repo"])
        _download_file(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
            "dinov2_vitb14_pretrain.pth",
            paths["weights"], DINOV2_WEIGHT_SHA256,
        )
        return
    if model in HF_REVISIONS:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install huggingface-hub before downloading HF models") from exc
        repo_id, revision = HF_REVISIONS[model]
        paths["model"].mkdir(parents=True, exist_ok=True)
        print(f"[download] {repo_id}@{revision}", flush=True)
        try:
            allow_patterns = ["*.json", "*.safetensors", "*.model"]
            if model == "cradio_v4_so400m":
                allow_patterns.append("*.py")
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=paths["model"],
                token=token,
                allow_patterns=allow_patterns,
                endpoint=hf_endpoint,
            )
        except Exception as exc:
            if model == "dinov3_b":
                raise RuntimeError(
                    "DINOv3 is gated. Accept its Hugging Face license and set "
                    "HF_TOKEN on the connected download machine."
                ) from exc
            raise
        (paths["model"] / ".sgg_source.json").write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": revision,
                    "download_endpoint": hf_endpoint,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    if model == "radio_v25_b":
        paths["repo"].parent.mkdir(parents=True, exist_ok=True)
        _download_repo("NVlabs/RADIO", RADIO_COMMIT, paths["repo"])
        _download_file(
            f"{hf_endpoint}/nvidia/RADIO/resolve/"
            f"{RADIO_HF_REVISION}/radio-v2.5-b_half.pth.tar?download=true",
            paths["weights"], RADIO_WEIGHT_SHA256,
        )
        return
    raise ValueError(f"Unsupported model: {model}")


def status_for(model: str, root: Path) -> dict:
    paths = paths_for(root)[model]
    missing: list[str] = []
    mismatches: list[str] = []
    resolved: list[str] = []
    if model == "resnet50":
        candidates = sorted(paths["root"].glob("resnet50-*.pth"))
        if not candidates:
            missing.append(str(paths["root"] / "resnet50-*.pth"))
        else:
            resolved.extend(str(path) for path in candidates)
    elif model in {"dinov2_b", "radio_v25_b"}:
        repo = paths["repo"]
        weights = paths["weights"]
        if not (repo / "hubconf.py").is_file():
            missing.append(str(repo / "hubconf.py"))
        else:
            resolved.append(str(repo))
            marker = repo / ".sgg_source.json"
            expected_commit = (
                DINOV2_COMMIT if model == "dinov2_b" else RADIO_COMMIT
            )
            if not marker.is_file():
                mismatches.append(f"Missing source marker: {marker}")
            else:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload.get("commit") != expected_commit:
                    mismatches.append(
                        f"Unexpected source commit in {marker}: "
                        f"{payload.get('commit')} != {expected_commit}"
                    )
        if not weights.is_file():
            missing.append(str(weights))
        else:
            resolved.append(str(weights))
            expected = (
                DINOV2_WEIGHT_SHA256
                if model == "dinov2_b" else RADIO_WEIGHT_SHA256
            )
            actual = sha256(weights)
            if actual != expected:
                mismatches.append(f"{weights}: expected={expected} actual={actual}")
    else:
        model_dir = paths["model"]
        configs = list(model_dir.glob("config.json"))
        weights = list(model_dir.glob("*.safetensors"))
        if not configs:
            missing.append(str(model_dir / "config.json"))
        if not weights:
            missing.append(str(model_dir / "*.safetensors"))
        if configs and weights:
            resolved.append(str(model_dir))
        marker = model_dir / ".sgg_source.json"
        if model == "cradio_v4_so400m":
            for filename in CRADIO_CODE_FILES:
                if not (model_dir / filename).is_file():
                    missing.append(str(model_dir / filename))
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            expected_repo, expected_revision = HF_REVISIONS[model]
            if (
                payload.get("repo_id") != expected_repo
                or payload.get("revision") != expected_revision
            ):
                mismatches.append(f"Unexpected source marker: {marker}")
        elif model_dir.exists():
            mismatches.append(f"Missing source marker: {marker}")
    return {
        "model": model,
        "ok": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
        "resolved": resolved,
    }


def smoke_load(model: str, root: Path, device: str) -> dict:
    os.environ["SGG_PROJECT_ROOT"] = str(root)
    os.environ["SGG_OFFLINE"] = "1"
    from sgg_core.backbones.perception import PerceptionModule
    import torch

    input_size = PerceptionModule.BACKBONE_REGISTRY[model][2]
    print(f"[smoke] model={model} device={device} input={input_size}", flush=True)
    instance = PerceptionModule(
        model, cache_dir=str(root / "data" / "derived" / "cache" / "model_smoke"),
        output_dim=PerceptionModule.get_output_dim_for(model),
    ).to(device).eval()
    with torch.inference_mode():
        output = instance._extract_feature_map(
            torch.zeros(1, 3, input_size, input_size, device=device)
        )
    shape = list(output.shape)
    del output, instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(shape) != 4 or shape[0] != 1 or min(shape[2:]) < 1:
        raise RuntimeError(f"Invalid spatial output for {model}: {shape}")
    print(f"[smoke-ok] {model} shape={shape}", flush=True)
    return {"model": model, "shape": shape, "device": device}


def package_assets(root: Path, models: list[str], output: Path, report: Path) -> None:
    members: list[Path] = [report]
    all_paths = paths_for(root)
    for model in models:
        if model == "resnet50":
            members.extend(sorted(all_paths[model]["root"].glob("resnet50-*.pth")))
        else:
            members.extend(all_paths[model].values())
    unique: list[Path] = []
    for path in members:
        if path.exists() and path not in unique:
            unique.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(output, mode, dereference=True) as archive:
        for path in unique:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=True)
    print(f"[package] {output} size={output.stat().st_size:,}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_root", default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS,
        default=list(MAIN_MODELS),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--smoke_load", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--package", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
        help=(
            "Hugging Face download endpoint. Defaults to HF_ENDPOINT or "
            f"{DEFAULT_HF_ENDPOINT}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    models = list(dict.fromkeys(args.models))
    hf_endpoint = args.hf_endpoint.rstrip("/")
    if not hf_endpoint.startswith(("http://", "https://")):
        raise ValueError("--hf_endpoint must start with http:// or https://")
    os.environ["SGG_PROJECT_ROOT"] = str(root)
    os.environ.setdefault("TORCH_HOME", str(root / "checkpoints" / "foundation" / "torch_hub"))
    os.environ.setdefault("HF_HOME", str(root / "checkpoints" / "foundation" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[huggingface] endpoint={hf_endpoint}", flush=True)

    download_errors = {}
    if args.download and not args.check_only:
        for model in models:
            try:
                download_model(model, root, token, hf_endpoint)
            except Exception as exc:
                download_errors[model] = f"{type(exc).__name__}: {exc}"
                print(f"[download-failed] {model}: {download_errors[model]}", flush=True)

    statuses = [status_for(model, root) for model in models]
    smoke_results, smoke_errors = [], {}
    if args.smoke_load:
        for status in statuses:
            if not status["ok"]:
                continue
            try:
                smoke_results.append(smoke_load(status["model"], root, args.device))
            except Exception as exc:
                smoke_errors[status["model"]] = f"{type(exc).__name__}: {exc}"
                print(f"[smoke-failed] {status['model']}: {smoke_errors[status['model']]}")

    report = Path(args.report).expanduser().resolve() if args.report else (
        root / "artifacts" / "manifests" / "foundation_assets.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "project_root": str(root),
        "huggingface_endpoint": hf_endpoint,
        "models": models,
        "status": statuses,
        "download_errors": download_errors,
        "smoke_results": smoke_results,
        "smoke_errors": smoke_errors,
        "complete": (
            all(item["ok"] for item in statuses)
            and not download_errors and not smoke_errors
            and (not args.smoke_load or len(smoke_results) == len(models))
        ),
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in statuses:
        print(f"[{'ok' if item['ok'] else 'miss'}] {item['model']}")
        for value in item["missing"] + item["mismatches"]:
            print(f"       {value}")
    print(f"report={report}")

    if args.package:
        if not payload["complete"]:
            print("[NOT PACKAGED] Assets must pass all requested checks first.")
        else:
            package_assets(root, models, Path(args.package).expanduser().resolve(), report)
    if payload["complete"]:
        print("[READY] Requested foundation models are available.")
        return 0
    print("[NOT READY] Fix the missing or failed foundation models.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
