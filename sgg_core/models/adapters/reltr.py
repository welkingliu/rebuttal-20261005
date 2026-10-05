"""Official RelTR inference bridge for the VG-150 SGDet checkpoint.

The upstream model predicts a fixed set of subject/object relation queries.
Each query is represented as two entity nodes in the common evaluator so the
official subject and object boxes, class logits, and predicate logits remain
paired without heuristic cross-query deduplication.
"""

from __future__ import annotations

from argparse import Namespace
import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.transforms import functional as tvf

from sgg_core.models.official_adapter import fingerprint_tensors


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    ).clamp(0.0, 1.0)


def _scene_graph_from_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert one RelTR image output to the common sparse SGG schema."""
    required = ("sub_logits", "obj_logits", "sub_boxes", "obj_boxes", "rel_logits")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(f"RelTR output is missing: {missing}")

    tensors = {
        key: outputs[key][0] if outputs[key].ndim == 3 else outputs[key]
        for key in required
    }
    count = int(tensors["rel_logits"].shape[0])
    if any(int(tensor.shape[0]) != count for tensor in tensors.values()):
        raise ValueError("RelTR relation-query outputs have inconsistent lengths")

    # The final upstream entity class is no-object. The common evaluator does
    # not carry that class, so preserve its mass separately as box objectness;
    # softmax over the remaining logits then recovers the official conditional
    # class score and objectness * class_score recovers the upstream score.
    # The released VG head contains IDs 0..150 followed by no-object. ID 0 is
    # already the VG background slot, so the remaining 1..150 columns align
    # directly with the shared VG-150 evaluator.
    entity_scores = torch.cat(
        (tensors["sub_logits"][:, :-1], tensors["obj_logits"][:, :-1]), dim=0
    )
    sub_objectness = 1.0 - tensors["sub_logits"].softmax(dim=-1)[:, -1]
    obj_objectness = 1.0 - tensors["obj_logits"].softmax(dim=-1)[:, -1]
    box_scores = torch.cat((sub_objectness, obj_objectness), dim=0)

    # Upstream RelTR evaluates softmax(rel_logits[1:-1]): index 0 is the
    # background predicate and the final index is no-relation. Retain a dummy
    # background column for the common 51-class VG schema without allowing it
    # to alter the foreground normalization.
    relation_foreground = tensors["rel_logits"][:, 1:-1]
    relation_background = relation_foreground.new_full((count, 1), -1e4)
    relation_scores = torch.cat((relation_background, relation_foreground), dim=1)
    boxes = torch.cat(
        (_cxcywh_to_xyxy(tensors["sub_boxes"]),
         _cxcywh_to_xyxy(tensors["obj_boxes"])),
        dim=0,
    )
    query_ids = torch.arange(count, device=boxes.device, dtype=torch.long)
    pairs = torch.stack((query_ids, query_ids + count), dim=1)
    return {
        "pred_boxes": boxes,
        "pred_entity_scores": entity_scores,
        "pred_box_scores": box_scores,
        "pred_rel_pairs": pairs,
        "pred_rel_scores": relation_scores,
    }


def _model_args(device: torch.device, config: dict[str, Any]) -> Namespace:
    values = {
        "dataset": "vg",
        "device": str(device),
        "lr_backbone": 1e-5,
        "backbone": "resnet50",
        "dilation": False,
        "position_embedding": "sine",
        "enc_layers": 6,
        "dec_layers": 6,
        "dim_feedforward": 2048,
        "hidden_dim": 256,
        "dropout": 0.1,
        "nheads": 8,
        "num_entities": 100,
        "num_triplets": 200,
        "pre_norm": False,
        "aux_loss": True,
        "set_cost_class": 1.0,
        "set_cost_bbox": 5.0,
        "set_cost_giou": 2.0,
        "set_iou_threshold": 0.7,
        "bbox_loss_coef": 5.0,
        "giou_loss_coef": 2.0,
        "rel_loss_coef": 1.0,
        "eos_coef": 0.1,
        "return_interm_layers": False,
    }
    allowed = set(values)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unsupported RelTR config values: {unknown}")
    values.update(config)
    if values["dataset"] != "vg":
        raise ValueError(
            "This adapter currently supports only the ontology-aligned VG checkpoint"
        )
    return Namespace(**values)


class RelTRVGAdapter(nn.Module):
    """Thin inference-only wrapper around the pinned official RelTR source."""

    def __init__(self, checkpoint: str, device: torch.device, config: dict[str, Any]):
        super().__init__()
        self.device = torch.device(device)

        # Upstream initializes an ImageNet-pretrained backbone before restoring
        # the complete RelTR checkpoint. Disable that redundant network request.
        backbone_module = importlib.import_module("models.backbone")
        backbone_module.is_main_process = lambda: False
        build_model = importlib.import_module("models").build_model
        model, _, _ = build_model(_model_args(self.device, config))

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # torch<2.0 has no weights_only argument
            payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TypeError(f"RelTR checkpoint has no model state_dict: {checkpoint_path}")
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    def _prepare_image(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch.get("image")
        if not isinstance(image, torch.Tensor):
            raise KeyError("RelTR raw-image inference requires batch['image']")
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("RelTR adapter requires image batch_size=1")
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("batch['image'] must have shape [3,H,W]")
        image = image.to(self.device, dtype=torch.float32)
        if not bool(torch.isfinite(image).all()):
            raise ValueError("batch['image'] contains non-finite values")
        height, width = map(int, image.shape[-2:])
        resize_short = 800
        if max(height, width) / max(min(height, width), 1) * resize_short > 1333:
            resize_short = int(round(1333 * min(height, width) / max(height, width)))
        image = tvf.resize(image, resize_short, antialias=True)
        image = tvf.normalize(image, _IMAGENET_MEAN, _IMAGENET_STD)
        return image.unsqueeze(0)

    def _forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = self._prepare_image(batch)
        outputs = self.model(image)
        return image, outputs

    def predict_scene_graph(self, batch: dict[str, Any], task: str) -> dict[str, torch.Tensor]:
        if str(task).lower() != "sgdet":
            raise NotImplementedError("The released RelTR checkpoint supports SGDet only")
        _, outputs = self._forward(batch)
        return _scene_graph_from_outputs(outputs)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "RelTR sparse relation queries are not aligned with annotated GT pairs; "
            "use predict_scene_graph(..., task='sgdet')"
        )

    def extract_node_features(self, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError(
            "RelTR does not expose GT-node-aligned features in its official inference API"
        )

    def diagnostic_input_fingerprint(self, batch: dict[str, Any]) -> str:
        return fingerprint_tensors(image=self._prepare_image(batch))


def create_adapter(*, checkpoint: str, checkpoints: dict[str, str], device,
                   config: dict[str, Any], diagnostic_task: str) -> RelTRVGAdapter:
    if str(diagnostic_task).lower() != "sgdet":
        raise ValueError("RelTR's released checkpoint is an SGDet checkpoint")
    if set(checkpoints) != {"sgdet"} or checkpoints["sgdet"] != checkpoint:
        raise ValueError("RelTR manifest must declare exactly one SGDet checkpoint")
    return RelTRVGAdapter(checkpoint=checkpoint, device=torch.device(device), config=config)
