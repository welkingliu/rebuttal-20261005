"""Experiment I-A external box-only zero-shot object-identity audit.

GQA and VRD have box annotations but do not provide the panoptic-mask protocol
used by the primary PSG experiment. This entry point therefore evaluates one
frozen vision-language encoder on ground-truth box crops without fitting a
classifier. Dataset training annotations are read only to define fixed
head/body/tail frequency strata.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F

from sgg_core.audits.object_grounding import (
    evaluate_object_logits,
    relationship_endpoint_summary,
)
from sgg_core.data.gqa_psg_data_utils import build_gqa_loader
from sgg_core.data.vrd_data_utils import build_vrd_loader


SCHEMA_VERSION = "experiment_1a_external_box_zero_shot_v1"
DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"


def _normalise_label(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _gqa_frequency(path: Path) -> Counter:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter()
    for graph in payload.values():
        for obj in graph.get("objects", {}).values():
            counts[_normalise_label(obj.get("name", "object"))] += 1
    return counts


def _vrd_frequency(root: Path) -> Counter:
    objects = json.loads(
        (root / "json_dataset" / "objects.json").read_text(encoding="utf-8")
    )
    annotations = json.loads(
        (root / "json_dataset" / "annotations_train.json").read_text(encoding="utf-8")
    )
    counts = Counter()
    records = annotations.values() if isinstance(annotations, dict) else annotations
    for relations in records:
        for relation in relations:
            for role in ("subject", "object"):
                category = int(relation.get(role, {}).get("category", -1))
                if 0 <= category < len(objects):
                    counts[_normalise_label(objects[category])] += 1
    return counts


def _frequency_groups_from_names(class_names: list[str], counts: Counter) -> dict:
    supported = [
        index for index, name in enumerate(class_names)
        if counts[_normalise_label(name)] > 0
    ]
    supported.sort(
        key=lambda index: (-counts[_normalise_label(class_names[index])], index)
    )
    partitions = np.array_split(np.asarray(supported, dtype=np.int64), 3)
    return {
        name: [int(value) for value in partition.tolist()]
        for name, partition in zip(("head", "body", "tail"), partitions)
    }


def _model_assets(model_ref: str) -> dict:
    path = Path(model_ref).expanduser()
    if not path.is_dir():
        return {"source": model_ref, "local": False}
    files = []
    for pattern in ("*.safetensors", "config.json", "preprocessor_config.json"):
        for item in sorted(path.glob(pattern)):
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files.append({
                "path": str(item.resolve()),
                "size": item.stat().st_size,
                "sha256": digest.hexdigest(),
            })
    return {"source": str(path.resolve()), "local": True, "files": files}


def _load_model(model_ref: str, allow_download: bool, device: torch.device):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("External I-A requires transformers") from exc
    path = Path(model_ref).expanduser()
    local = path.is_dir()
    if not local and not allow_download:
        raise FileNotFoundError(
            f"Model directory does not exist: {path}. Pass --allow_download only "
            "when downloading the registered Hugging Face revision is intended."
        )
    source = str(path.resolve()) if local else model_ref
    processor = AutoProcessor.from_pretrained(source, local_files_only=local)
    model = AutoModel.from_pretrained(source, local_files_only=local).to(device)
    model.eval()
    if not callable(getattr(model, "get_text_features", None)):
        raise TypeError("The selected model does not expose get_text_features")
    if not callable(getattr(model, "get_image_features", None)):
        raise TypeError("The selected model does not expose get_image_features")
    return model, processor


def _feature_tensor(value, kind: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    preferred = (
        ("text_embeds", "pooler_output", "last_hidden_state")
        if kind == "text"
        else ("image_embeds", "pooler_output", "last_hidden_state")
    )
    for name in preferred:
        tensor = getattr(value, name, None)
        if isinstance(tensor, torch.Tensor):
            if name == "last_hidden_state" and tensor.ndim == 3:
                return tensor.mean(dim=1)
            return tensor
    raise TypeError(f"Could not extract {kind} features from {type(value).__name__}")


def _text_features(model, processor, prompts: list[str], device: torch.device,
                   batch_size: int) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            inputs = processor(
                text=prompts[start:start + batch_size],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            features = _feature_tensor(
                model.get_text_features(**inputs), "text"
            )
            values.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(values)


def _crop_objects(image_path: Path, boxes: torch.Tensor,
                  context_fraction: float) -> tuple[list[Image.Image], torch.Tensor]:
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    crops, valid = [], []
    for index, raw_box in enumerate(boxes.tolist()):
        x1, y1, x2, y2 = [float(value) for value in raw_box]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        pad_x = box_width * context_fraction
        pad_y = box_height * context_fraction
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right <= left or bottom <= top:
            continue
        crops.append(image.crop((left, top, right, bottom)))
        valid.append(index)
    return crops, torch.tensor(valid, dtype=torch.long)


def _image_logits(model, processor, crops: list[Image.Image],
                  text_features: torch.Tensor, device: torch.device,
                  batch_size: int) -> torch.Tensor:
    values = []
    scale_parameter = getattr(model, "logit_scale", None)
    bias_parameter = getattr(model, "logit_bias", None)
    scale = (
        float(scale_parameter.detach().float().exp().cpu())
        if isinstance(scale_parameter, torch.Tensor) else 1.0
    )
    bias = (
        float(bias_parameter.detach().float().cpu())
        if isinstance(bias_parameter, torch.Tensor) else 0.0
    )
    text = text_features.to(device)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            inputs = processor(
                images=crops[start:start + batch_size], return_tensors="pt"
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            features = F.normalize(_feature_tensor(
                model.get_image_features(**inputs), "image"
            ).float(), dim=-1)
            values.append((features @ text.T * scale + bias).cpu())
    return torch.cat(values)


def _dataset(args):
    if args.dataset == "gqa":
        if not args.eval_ann or not args.image_root or not args.train_ann:
            raise ValueError("GQA requires --train_ann, --eval_ann, and --image_root")
        loader = build_gqa_loader(
            args.eval_ann,
            num_samples=args.eval_samples,
            vocabulary_path=args.eval_ann,
            image_root=args.image_root,
            include_proxy_features=False,
            include_raw_images=False,
        )
        frequencies = _gqa_frequency(Path(args.train_ann))
    else:
        if not args.data_root:
            raise ValueError("VRD requires --data_root")
        loader = build_vrd_loader(
            args.data_root, "test", args.eval_samples,
            include_proxy_features=False, include_raw_images=False,
        )
        frequencies = _vrd_frequency(Path(args.data_root))
    labels_by_id = loader.dataset.sgg_dict["idx_to_label"]
    class_names = [
        labels_by_id[str(index)]
        for index in range(1, loader.dataset.num_entity_classes)
    ]
    return loader, class_names, frequencies


def _safe_image_path(batch: dict) -> Path:
    value = batch.get("image_path")
    if not value:
        raise FileNotFoundError(f"Raw image missing for image_id={batch.get('image_id')}")
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("gqa", "vrd"))
    parser.add_argument("--data_root")
    parser.add_argument("--train_ann")
    parser.add_argument("--eval_ann")
    parser.add_argument("--image_root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--model_path")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--prompt_template", default="This is a photo of a {}.")
    parser.add_argument("--context_fraction", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        default=(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, class_names, frequencies = _dataset(args)
    if "{}" not in args.prompt_template:
        raise ValueError("--prompt_template must contain one '{}' placeholder")
    groups = _frequency_groups_from_names(class_names, frequencies)
    image_paths = []
    missing = []
    for batch in loader:
        try:
            image_paths.append(str(_safe_image_path(batch)))
        except FileNotFoundError as exc:
            missing.append(str(exc))
    preflight = {
        "dataset": args.dataset,
        "graphs": len(loader.dataset),
        "classes": len(class_names),
        "images_ready": len(image_paths),
        "images_missing": len(missing),
        "missing_examples": missing[:20],
        "model": args.model_path or args.model_id,
        "device": args.device,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        raise RuntimeError(
            f"{args.dataset} external I-A is not ready: {len(missing)} images missing"
        )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    model_ref = args.model_path or args.model_id
    device = torch.device(args.device)
    model, processor = _load_model(model_ref, args.allow_download, device)
    prompts = [args.prompt_template.format(name) for name in class_names]
    text_features = _text_features(
        model, processor, prompts, device, args.text_batch_size
    )

    logits_chunks = []
    labels = []
    image_ids = []
    areas = []
    graph_records = []
    skipped_images = []
    object_offset = 0
    for batch in loader:
        image_id = str(batch["image_id"])
        try:
            crops, valid = _crop_objects(
                _safe_image_path(batch), batch["boxes"], args.context_fraction
            )
        except (FileNotFoundError, OSError) as exc:
            skipped_images.append({"image_id": image_id, "reason": str(exc)})
            continue
        if not crops:
            skipped_images.append({"image_id": image_id, "reason": "no_valid_crops"})
            continue
        image_logits = _image_logits(
            model, processor, crops, text_features, device, args.batch_size
        )
        local_labels = batch["entity_labels"].long()[valid] - 1
        if image_logits.size(0) != local_labels.numel():
            raise RuntimeError("Crop/logit alignment mismatch")
        local_index = {int(old): new for new, old in enumerate(valid.tolist())}
        pairs = []
        relation_labels = []
        for pair, predicate in zip(
            batch["rel_pairs"].tolist(), batch["rel_labels"].tolist()
        ):
            subject, obj = map(int, pair)
            if subject in local_index and obj in local_index:
                pairs.append([local_index[subject], local_index[obj]])
                relation_labels.append(int(predicate))
        start = object_offset
        stop = start + local_labels.numel()
        graph_records.append({
            "image_id": image_id,
            "object_start": start,
            "object_stop": stop,
            "rel_pairs": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
            "rel_labels": torch.tensor(relation_labels, dtype=torch.long),
        })
        object_offset = stop
        logits_chunks.append(image_logits)
        labels.append(local_labels)
        image_ids.extend([image_id] * local_labels.numel())
        boxes = batch["boxes"][valid].float()
        areas.append(
            ((boxes[:, 2] - boxes[:, 0]).clamp_min(0)
             * (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
        )
        print(json.dumps({
            "dataset": args.dataset,
            "image_id": image_id,
            "objects": int(local_labels.numel()),
            "processed_images": len(graph_records),
            "total_images": len(loader.dataset),
        }), flush=True)

    if not labels:
        raise RuntimeError("No external I-A object crops were evaluated")
    logits = torch.cat(logits_chunks)
    target = torch.cat(labels)
    area = torch.cat(areas)
    prediction = logits.argmax(dim=1)
    metrics = evaluate_object_logits(
        logits, target, image_ids, groups, areas=area, temperature=1.0
    )
    endpoint = relationship_endpoint_summary(
        prediction, target, graph_records, bootstrap_seed=args.seed + 9000
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "I-A_external_box_only_zero_shot_identity",
        "dataset": args.dataset,
        "paper_evidence_tier": "box_only_zero_shot_external",
        "claim_scope": (
            "Inference-only object identity under GT boxes. This protocol has "
            "no mask-IoU condition and is not directly ranked against the PSG "
            "supervised linear-probe panel."
        ),
        "parameter_updates": 0,
        "classifier_training": False,
        "calibration": "uncalibrated_zero_shot_softmax",
        "model": model_ref,
        "model_assets": _model_assets(model_ref),
        "prompt_template": args.prompt_template,
        "context_fraction": args.context_fraction,
        "ontology_id": loader.dataset.ontology_id,
        "num_classes": len(class_names),
        "class_names": class_names,
        "frequency_groups_from_training_annotations": groups,
        "metrics": metrics,
        "relationship_endpoints": endpoint,
        "processed_images": len(graph_records),
        "processed_objects": int(target.numel()),
        "skipped_images": skipped_images,
        "preflight": preflight,
        "config": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=target.numpy(),
        predictions=prediction.numpy(),
        confidence=logits.softmax(dim=1).max(dim=1).values.numpy(),
        image_ids=np.asarray(image_ids, dtype=str),
    )
    print(f"Experiment I-A external complete: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
