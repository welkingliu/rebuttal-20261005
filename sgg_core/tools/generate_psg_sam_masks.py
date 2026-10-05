"""Generate class-agnostic SAM masks for PSG objects using GT-box prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sgg_core.data.data_utils import load_rgb_image
from sgg_core.data.gqa_psg_data_utils import build_psg_loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_digest(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _safe(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_")


def _mask_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape[-2:] != target.shape[-2:]:
        predicted = F.interpolate(
            predicted[:, None].float(), size=target.shape[-2:], mode="nearest"
        )[:, 0].bool()
    target = target.bool()
    intersection = (predicted & target).flatten(1).sum(dim=1).float()
    union = (predicted | target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--panoptic_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=0,
                        help="0 means every graph in the annotation split")
    parser.add_argument("--prompt_chunk_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import SamModel, SamProcessor
    except ImportError as exc:
        raise ImportError("Install transformers>=4.57 to generate SAM masks") from exc
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"SAM model directory is incomplete: {model_dir}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    processor = SamProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = SamModel.from_pretrained(str(model_dir), local_files_only=True).to(args.device)
    model.eval()
    loader = build_psg_loader(
        args.annotation, args.samples,
        image_root=args.image_root, panoptic_root=args.panoptic_root,
        include_proxy_features=False, include_raw_images=False,
    )
    records = []
    for batch in loader:
        image_id = batch["image_id"]
        destination = mask_dir / f"{_safe(image_id)}.npz"
        if destination.is_file() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as cached:
                cached_masks = np.asarray(cached["masks"])
                cached_iou = np.asarray(
                    cached["gt_mask_iou"]
                    if "gt_mask_iou" in cached
                    else np.asarray([], dtype=np.float32)
                )
            records.append({
                "image_id": str(image_id),
                "status": "cached",
                "image_decode_status": "not_redecoded_cached_mask",
                "objects": int(cached_masks.shape[0]),
                "mean_gt_mask_iou": (
                    float(cached_iou.mean()) if cached_iou.size else None
                ),
                "path": str(destination),
            })
            continue
        image_path = Path(batch.get("image_path", ""))
        if not image_path.is_file() or "masks" not in batch:
            records.append({"image_id": str(image_id), "status": "missing_image_or_gt_mask"})
            continue
        image, decode_status = load_rgb_image(image_path)
        if image is None:
            record = {
                "image_id": str(image_id),
                "status": "image_decode_failed",
                "image_decode_status": decode_status,
                "image_path": str(image_path),
                "image_sha256": _sha256(image_path),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            continue
        if decode_status != "strict":
            print(json.dumps({
                "image_id": str(image_id),
                "image_path": str(image_path),
                "warning": decode_status,
            }, sort_keys=True), flush=True)
        width, height = image.size
        boxes = batch["boxes"].float().clone()
        boxes[:, (0, 2)] *= width
        boxes[:, (1, 3)] *= height
        selected_masks, selected_scores = [], []
        for start in range(0, boxes.size(0), args.prompt_chunk_size):
            chunk = boxes[start:start + args.prompt_chunk_size]
            inputs = processor(
                images=image,
                input_boxes=[chunk.tolist()],
                return_tensors="pt",
            )
            device_inputs = {
                key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = model(**device_inputs, multimask_output=True)
            post = processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"], inputs["reshaped_input_sizes"],
            )[0]
            scores = outputs.iou_scores.detach().cpu()[0]
            if post.ndim != 4:
                raise RuntimeError(f"Unexpected SAM post-processed shape: {tuple(post.shape)}")
            best = scores.argmax(dim=-1)
            row = torch.arange(post.size(0))
            selected_masks.append(post[row, best].bool())
            selected_scores.append(scores[row, best])
        masks = torch.cat(selected_masks, dim=0)
        scores = torch.cat(selected_scores, dim=0)
        gt_masks = batch["masks"].bool()
        iou = _mask_iou(masks, gt_masks)
        np.savez_compressed(
            destination,
            masks=masks.numpy().astype(np.uint8),
            predicted_iou=scores.numpy().astype(np.float32),
            gt_mask_iou=iou.numpy().astype(np.float32),
            segment_ids=(
                batch["segment_ids"].numpy().astype(np.int64)
                if "segment_ids" in batch else np.arange(masks.size(0), dtype=np.int64)
            ),
        )
        record = {
            "image_id": str(image_id),
            "status": "ok",
            "image_decode_status": decode_status,
            "image_sha256": _sha256(image_path),
            "objects": int(masks.size(0)),
            "mean_gt_mask_iou": float(iou.mean()),
            "path": str(destination),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    valid = [row for row in records if row["status"] in {"ok", "cached"}]
    model_files = sorted(
        path for pattern in ("config.json", "*.safetensors", ".sgg_source.json")
        for path in model_dir.glob(pattern) if path.is_file()
    )
    mask_files = sorted({Path(row["path"]) for row in valid})
    manifest = {
        "schema": "psg_sam_gt_box_prompt_v1",
        "annotation": str(Path(args.annotation).expanduser().resolve()),
        "annotation_sha256": _sha256(
            Path(args.annotation).expanduser().resolve()
        ),
        "model_dir": str(model_dir),
        "model_files": {
            str(path.relative_to(model_dir)): _sha256(path)
            for path in model_files
        },
        "prompt": "ground_truth_box",
        "class_labels_used": False,
        "images_total": len(records),
        "images_ready": len(valid),
        "mask_files": len(mask_files),
        "mask_cache_sha256": _cache_digest(mask_files),
        "prompt_chunk_size": args.prompt_chunk_size,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if len(valid) != len(records):
        raise RuntimeError(
            f"SAM mask generation incomplete: {len(valid)}/{len(records)} images"
        )
    print(f"SAM mask cache ready: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
