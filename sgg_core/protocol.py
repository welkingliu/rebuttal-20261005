"""Dataset, model-provenance, and result helpers shared by Experiments II-IV."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch


from sgg_core.models.official_adapter import OfficialSGGAdapter
from sgg_core.data.data_utils import build_vg_test_loader
from sgg_core.data.gqa_psg_data_utils import build_gqa_loader, build_psg_loader
from sgg_core.data.oi_data_utils import build_oi_loader
from sgg_core.data.vrd_data_utils import build_vrd_loader


RECALL_KS = (1, 5, 10, 20, 50, 100)
PAIR_HIT_KS = (1, 5, 10)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(dataset: str, data_root: str, train_samples: int,
                  eval_samples: int, train_ann: str | None = None,
                  eval_ann: str | None = None,
                  image_root: str | None = None,
                  panoptic_root: str | None = None,
                  include_proxy_features: bool = True,
                  include_raw_images: bool = True):
    """Return disjoint train/eval loaders with one stable ontology."""
    dataset = dataset.lower()
    if dataset == "vg":
        return (
            build_vg_test_loader(
                data_root, train_samples, split=0,
                include_proxy_features=include_proxy_features,
                include_raw_images=include_raw_images,
            ),
            build_vg_test_loader(
                data_root, eval_samples, split=2,
                include_proxy_features=include_proxy_features,
                include_raw_images=include_raw_images,
            ),
        )
    if dataset == "oi":
        return (
            build_oi_loader(data_root, "train", train_samples),
            build_oi_loader(data_root, "validation", eval_samples),
        )
    if dataset == "gqa":
        if not train_ann or not eval_ann:
            raise ValueError("GQA requires --train_ann and --eval_ann")
        return (
            build_gqa_loader(
                train_ann, train_samples, vocabulary_path=train_ann,
                image_root=image_root,
            ),
            build_gqa_loader(
                eval_ann, eval_samples, vocabulary_path=train_ann,
                image_root=image_root,
            ),
        )
    if dataset == "psg":
        if not train_ann or not eval_ann:
            raise ValueError("PSG requires --train_ann and --eval_ann")
        return (
            build_psg_loader(
                train_ann, train_samples, exclude_annotation_path=eval_ann,
                image_root=image_root, panoptic_root=panoptic_root,
            ),
            build_psg_loader(
                eval_ann, eval_samples, image_root=image_root,
                panoptic_root=panoptic_root,
            ),
        )
    if dataset == "vrd":
        return (
            build_vrd_loader(data_root, "train", train_samples),
            build_vrd_loader(data_root, "test", eval_samples),
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_official_models(manifest_paths, device: str, dataset: str,
                         loader, minimum_models: int = 1) -> dict:
    """Load only provenance-verified official adapters compatible with a loader."""
    if not manifest_paths:
        raise ValueError("At least one --official_manifest is required")
    ontology_id = getattr(getattr(loader, "dataset", None), "ontology_id", None)
    models = {}
    rejected = {}
    for manifest_path in manifest_paths:
        model = OfficialSGGAdapter(manifest_path, device=device)
        if not model.supports_dataset(dataset, ontology_id):
            rejected[model.name] = {
                "reason": "dataset_or_ontology_mismatch",
                "ontology_id": ontology_id,
            }
            continue
        if model.name in models:
            raise ValueError(f"Duplicate official model name: {model.name}")
        models[model.name] = model
    if len(models) < int(minimum_models):
        raise RuntimeError(
            f"Paper run requires {minimum_models} compatible official models; "
            f"loaded={len(models)}, rejected={rejected}"
        )
    return models


def model_provenance(models: dict) -> dict:
    report = {}
    for name, model in models.items():
        status = dict(model.checkpoint_status)
        report[name] = {
            "architecture_family": model.architecture_family,
            "implementation_kind": model.implementation_kind,
            "manifest_path": model.manifest_path,
            "checkpoint_status": status,
            "supported_tasks": list(model.supported_tasks),
            "perturbation_contract": model.manifest["perturbation_contract"],
        }
    return report


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
