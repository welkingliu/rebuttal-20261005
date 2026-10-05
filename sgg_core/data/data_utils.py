"""
utils/data_utils.py
====================
Data loading utilities for the SGG Diagnostic Protocol.

Provides:
  build_synthetic_loader  — generates random batches matching the VG schema
  build_vg_test_loader    — wraps real VG annotation files (no roi_features.hdf5 needed)

--- No roi_features.hdf5 mode ---
When roi_features.hdf5 is unavailable (the common situation), we construct
a high-dimensional proxy feature vector per object from:
  (a) Box geometry features  — 32-d sinusoidal encoding of (x1,y1,x2,y2,cx,cy,w,h,area,ar)
  (b) Label embedding        — 256-d learned-like embedding via deterministic hashing
  (c) Relative spatial features vs all other boxes — 64-d pairwise geometry
  (d) Image-crop statistics  — mean/std/histogram from raw pixels (if VG_100K available)

These are concatenated and projected to 4096-d to match the original feature dim.
Proxy features are provided only for software tests and explicitly labeled
dataset controls. They are not a substitute for detector RoI features or raw
image encoders and must not support paper claims about visual grounding.

Visual Genome batch schema
--------------------------
{
    "visual_features": Tensor [N, 4096]   RoI proxy features
    "union_features":  Tensor [M, 4096]   Union-box proxy features
    "boxes":           Tensor [N, 4]      xyxy normalised [0,1]
    "entity_labels":   Tensor [N]         entity class ids (0=bg)
    "rel_pairs":       Tensor [M, 2]      (subj_idx, obj_idx)
    "rel_labels":      Tensor [M]         predicate class ids (0=bg)
    "graph_adj":       Tensor [N, N]      binary adjacency
    "num_nodes":       int
    "image_id":        int
    "feature_source":  str                "roi_hdf5" | "geometry_proxy" | "image_crop"
}
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_ENTITY_CLASSES = 151
NUM_REL_CLASSES    = 51
ROI_FEAT_DIM       = 4096  # keep consistent with original pipeline dim
_PROJECTION_CACHE = {}
VG_CORRUPT_IMAGE_IDS = {1592, 1722, 4616, 4617}


def vg_cxcywh_to_xyxy(boxes: np.ndarray, scale: float = 1024.0) -> np.ndarray:
    """Convert canonical VG-SGG ``boxes_1024`` values to clipped xyxy."""
    values = np.asarray(boxes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("VG boxes must have shape [N,4]")
    if not np.isfinite(values).all():
        raise ValueError("VG boxes contain non-finite values")
    center = values[:, :2]
    size = np.maximum(values[:, 2:], 0.0)
    converted = np.concatenate((center - size / 2.0, center + size / 2.0), axis=1)
    return np.clip(converted, 0.0, float(scale)).astype(np.float32, copy=False)


def vg_boxes_to_normalized_xyxy(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
    scale: float = 1024.0,
) -> np.ndarray:
    """Recover canonical VG boxes in the original image coordinate system."""
    width, height = int(image_width), int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid VG image size: width={width}, height={height}")
    absolute = vg_cxcywh_to_xyxy(boxes, scale=scale)
    absolute *= max(width, height) / float(scale)
    absolute[:, [0, 2]] = np.clip(absolute[:, [0, 2]], 0.0, float(width))
    absolute[:, [1, 3]] = np.clip(absolute[:, [1, 3]], 0.0, float(height))
    normalizer = np.asarray([width, height, width, height], dtype=np.float32)
    return (absolute / normalizer).astype(np.float32, copy=False)


def _fixed_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    key = (int(in_dim), int(out_dim), int(seed))
    if key not in _PROJECTION_CACHE:
        rng = np.random.default_rng(seed=seed)
        matrix = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        _PROJECTION_CACHE[key] = matrix / np.sqrt(max(in_dim, 1))
    return _PROJECTION_CACHE[key]


def load_rgb_image(path: Optional[Path]):
    """Load one RGB PIL image, deterministically recovering truncated JPEGs.

    Returns ``(image, decode_status)``. Recovery is explicit so callers can
    retain provenance instead of silently accepting a damaged file.
    """
    if path is None or not Path(path).is_file():
        return None, "missing"
    try:
        from PIL import Image, ImageFile
    except ImportError:
        return None, "pillow_unavailable"

    def decode():
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
            return image.copy()

    try:
        return decode(), "strict"
    except OSError:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            return decode(), "truncated_recovery"
        except OSError:
            return None, "decode_failed"
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def load_image_tensor(path: Optional[Path]):
    """Load an RGB image as float32 [C,H,W], or return None when unavailable."""
    image, _ = load_rgb_image(path)
    if image is None:
        return None
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def find_vg_image(image_roots: list[Path], image_id: int | str) -> Optional[Path]:
    """Resolve a VG image across the two official sibling image archives."""
    filenames = [f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
    for root in image_roots:
        for filename in filenames:
            candidate = Path(root) / filename
            if candidate.is_file():
                return candidate
    return None


# ── Geometry-based proxy feature extractor ─────────────────────────────────────

def _sinusoidal_encode(values: np.ndarray, dim: int = 32) -> np.ndarray:
    """
    Encode scalar values using sinusoidal positional encoding.
    values : [N] or [N, K]
    returns: [N, dim] or [N, K*dim]
    """
    if values.ndim == 1:
        values = values[:, None]       # [N, 1]
    N, K = values.shape
    freqs = 2.0 ** np.arange(dim // 2)  # [dim/2]
    # [N, K, dim/2]
    args  = values[:, :, None] * freqs[None, None, :]
    enc   = np.concatenate([np.sin(args), np.cos(args)], axis=-1)  # [N, K, dim]
    return enc.reshape(N, K * dim).astype(np.float32)


def _label_embed(labels: np.ndarray, out_dim: int = 256,
                 num_classes: int = NUM_ENTITY_CLASSES) -> np.ndarray:
    """
    Deterministic label embedding via frequency-based hashing.
    Each class maps to a fixed unit-sphere vector — reproducible, no training needed.
    labels: [N] int array
    """
    rng_states = [np.random.default_rng(int(lbl) + 1) for lbl in labels]
    embs = np.stack([r.standard_normal(out_dim) for r in rng_states], axis=0)
    norms = np.linalg.norm(embs, axis=-1, keepdims=True).clip(1e-8)
    return (embs / norms).astype(np.float32)


def _box_geometry_features(boxes_norm: np.ndarray) -> np.ndarray:
    """
    Extract rich geometry features from normalised xyxy boxes [N, 4].
    Returns [N, F] float32 with F = 10 * 32 = 320 (sinusoidal over 10 scalars).
    """
    x1, y1, x2, y2 = boxes_norm[:, 0], boxes_norm[:, 1], \
                      boxes_norm[:, 2], boxes_norm[:, 3]
    cx  = (x1 + x2) / 2
    cy  = (y1 + y2) / 2
    w   = (x2 - x1).clip(0)
    h   = (y2 - y1).clip(0)
    ar  = (w / h.clip(1e-4)).clip(0, 10)     # aspect ratio
    area = (w * h).clip(0)

    scalars = np.stack([x1, y1, x2, y2, cx, cy, w, h, area, ar], axis=1)  # [N, 10]
    return _sinusoidal_encode(scalars, dim=32)   # [N, 320]


def _pairwise_spatial_features(boxes_norm: np.ndarray, pairs: np.ndarray,
                                feat_dim: int = 64) -> np.ndarray:
    """
    Relative spatial features between subject and object boxes.
    Returns [M, feat_dim].
    """
    if pairs.shape[0] == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)

    s_idx = pairs[:, 0]
    o_idx = pairs[:, 1]
    s_boxes = boxes_norm[s_idx]   # [M, 4]
    o_boxes = boxes_norm[o_idx]   # [M, 4]

    # Relative offsets and scale ratios
    rel_x1  = (o_boxes[:, 0] - s_boxes[:, 0])
    rel_y1  = (o_boxes[:, 1] - s_boxes[:, 1])
    rel_x2  = (o_boxes[:, 2] - s_boxes[:, 2])
    rel_y2  = (o_boxes[:, 3] - s_boxes[:, 3])
    s_w = (s_boxes[:, 2] - s_boxes[:, 0]).clip(1e-4)
    s_h = (s_boxes[:, 3] - s_boxes[:, 1]).clip(1e-4)
    o_w = (o_boxes[:, 2] - o_boxes[:, 0]).clip(1e-4)
    o_h = (o_boxes[:, 3] - o_boxes[:, 1]).clip(1e-4)
    scale_w = np.log((o_w / s_w).clip(1e-4))
    scale_h = np.log((o_h / s_h).clip(1e-4))

    scalars = np.stack([rel_x1, rel_y1, rel_x2, rel_y2,
                        scale_w, scale_h, o_w, o_h], axis=1)  # [M, 8]
    return _sinusoidal_encode(scalars, dim=8)   # [M, 64]


def _image_crop_features(img_path: Path, boxes_xyxy_abs: np.ndarray,
                          feat_dim: int = 128) -> Optional[np.ndarray]:
    """
    Extract per-box image-crop statistics (mean/std/histogram per channel).
    Returns [N, feat_dim] or None if image not available.
    Requires PIL; silently skips if unavailable.
    """
    try:
        from PIL import Image
        import warnings
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W, _ = img.shape
        N = boxes_xyxy_abs.shape[0]
        feats = []
        bins = 16
        for box in boxes_xyxy_abs:
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 <= x1 or y2 <= y1:
                feats.append(np.zeros(feat_dim, dtype=np.float32))
                continue
            crop = img[y1:y2, x1:x2, :]   # [h, w, 3]
            mean = crop.mean(axis=(0, 1))  # [3]
            std  = crop.std(axis=(0, 1))   # [3]
            # per-channel histogram
            hists = []
            for c in range(3):
                h, _ = np.histogram(crop[:, :, c], bins=bins, range=(0, 1))
                hists.append(h.astype(np.float32) / (h.sum() + 1e-6))
            hist_vec = np.concatenate(hists)   # [48]
            feat = np.concatenate([mean, std, hist_vec])  # [54]
            # pad / truncate to feat_dim
            if feat.shape[0] < feat_dim:
                feat = np.concatenate([feat,
                    np.zeros(feat_dim - feat.shape[0], dtype=np.float32)])
            feats.append(feat[:feat_dim])
        return np.stack(feats, axis=0)
    except Exception:
        return None


def build_proxy_features(boxes_norm: np.ndarray,
                          labels: np.ndarray,
                          pairs: np.ndarray,
                          out_dim: int = ROI_FEAT_DIM,
                          img_path: Path = None,
                          img_w: int = 1, img_h: int = 1) -> tuple:
    """
    Build [N, out_dim] proxy object features and [M, out_dim] union features
    from box geometry + label embeddings (+ optional image crops).

    Returns: (node_feats [N, out_dim], union_feats [M, out_dim], source_str)
    """
    N = boxes_norm.shape[0]
    M = pairs.shape[0] if pairs.ndim == 2 else 0

    # ── Node features ──────────────────────────────────────────────────────
    geo  = _box_geometry_features(boxes_norm)          # [N, 320]
    lbl  = _label_embed(labels, out_dim=256)            # [N, 256]
    base = np.concatenate([geo, lbl], axis=1)           # [N, 576]

    source = "geometry_proxy"

    # Try image crops for richer signal
    if img_path is not None and img_path.exists():
        abs_boxes = boxes_norm * np.array([img_w, img_h, img_w, img_h])
        crop_feat = _image_crop_features(img_path, abs_boxes, feat_dim=128)
        if crop_feat is not None:
            base   = np.concatenate([base, crop_feat], axis=1)  # [N, 704]
            source = "image_crop"

    # Project to out_dim via deterministic linear projection
    # Use a fixed seeded random matrix — same projection every call
    in_dim    = base.shape[1]
    proj_mat  = _fixed_projection(in_dim, out_dim, seed=99999)
    node_feats = (base @ proj_mat).astype(np.float32)  # [N, out_dim]

    # ── Union (pair) features ───────────────────────────────────────────────
    if M == 0:
        union_feats = np.zeros((0, out_dim), dtype=np.float32)
    else:
        pair_geo = _pairwise_spatial_features(boxes_norm, pairs, feat_dim=64)  # [M, 64]
        s_idx, o_idx = pairs[:, 0], pairs[:, 1]
        s_feat = node_feats[s_idx]                                              # [M, out_dim]
        o_feat = node_feats[o_idx]                                              # [M, out_dim]

        # Union box geometry
        ux1 = np.minimum(boxes_norm[s_idx, 0], boxes_norm[o_idx, 0])
        uy1 = np.minimum(boxes_norm[s_idx, 1], boxes_norm[o_idx, 1])
        ux2 = np.maximum(boxes_norm[s_idx, 2], boxes_norm[o_idx, 2])
        uy2 = np.maximum(boxes_norm[s_idx, 3], boxes_norm[o_idx, 3])
        u_labels = labels[s_idx]   # use subject label for union embedding
        union_boxes = np.stack([ux1, uy1, ux2, uy2], axis=1)
        union_geo   = _box_geometry_features(union_boxes)         # [M, 320]
        union_lbl   = _label_embed(u_labels, out_dim=256)         # [M, 256]
        union_base  = np.concatenate([union_geo, union_lbl], axis=1)  # [M, 576]

        proj2 = _fixed_projection(576, out_dim, seed=88888)
        union_feats = union_base @ proj2                          # [M, out_dim]

    return node_feats, union_feats, source


# ── Synthetic Data ──────────────────────────────────────────────────────────────

class SyntheticSGGDataset(Dataset):
    """
    Generates plausible random batches for unit-testing the diagnostic pipeline.
    Distributions are calibrated to match VG statistics:
    - Avg 11.5 objects per image
    - Avg 6.2 relations per image
    - Skewed entity/predicate distributions
    """

    def __init__(self, num_samples: int = 500, seed: int = 42):
        self.num_samples = num_samples
        self.ontology_id = "synthetic:vg150-like-v1"
        self.num_entity_classes = NUM_ENTITY_CLASSES
        self.num_predicate_classes = NUM_REL_CLASSES
        rng = np.random.default_rng(seed)

        # Pre-generate scene sizes
        self.num_nodes_list = rng.integers(4, 20, size=num_samples)
        self.seeds          = rng.integers(0, 2**31, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng   = np.random.default_rng(self.seeds[idx])
        N     = int(self.num_nodes_list[idx])
        M     = max(1, int(rng.integers(2, max(3, N * 2))))  # pairs

        # Skewed entity distribution (person is most frequent)
        entity_probs = np.ones(NUM_ENTITY_CLASSES)
        entity_probs[1] = 20   # person
        entity_probs[2] = 10   # car
        entity_probs    = entity_probs / entity_probs.sum()
        entity_labels   = rng.choice(NUM_ENTITY_CLASSES, size=N, p=entity_probs)

        # Skewed predicate distribution (on/near/has are most frequent)
        pred_probs    = np.ones(NUM_REL_CLASSES) * 0.5
        pred_probs[1] = 30   # "on"
        pred_probs[2] = 20   # "has"
        pred_probs[3] = 15   # "near"
        pred_probs    = pred_probs / pred_probs.sum()
        rel_labels    = rng.choice(NUM_REL_CLASSES, size=M, p=pred_probs)

        # Random relation pairs (no self-loops)
        rel_pairs = []
        for _ in range(M):
            s = rng.integers(0, N)
            o = rng.integers(0, N)
            while o == s and N > 1:
                o = rng.integers(0, N)
            rel_pairs.append([s, o])
        rel_pairs = np.array(rel_pairs)

        # Boxes: random xyxy in [0,1]
        x1y1 = rng.uniform(0, 0.8, size=(N, 2))
        wh    = rng.uniform(0.05, 0.4, size=(N, 2))
        x2y2  = np.clip(x1y1 + wh, 0, 1)
        boxes = np.concatenate([x1y1, x2y2], axis=1)

        # Sparse visual features (simulating RoI-pool; low rank for bias signal)
        rank = max(2, N // 4)
        A    = rng.standard_normal((N, rank)).astype(np.float32)
        B    = rng.standard_normal((rank, ROI_FEAT_DIM)).astype(np.float32)
        vis  = ((A @ B) / np.sqrt(rank)).astype(np.float32)

        U    = rng.standard_normal((M, rank)).astype(np.float32)
        uni  = ((U @ B) / np.sqrt(rank)).astype(np.float32)

        # Adjacency
        adj  = np.zeros((N, N), dtype=np.float32)
        for s, o in rel_pairs:
            adj[s, o] = 1.0
            adj[o, s] = 1.0

        result = {
            "visual_features": torch.from_numpy(vis),
            "union_features":  torch.from_numpy(uni),
            "boxes":           torch.from_numpy(boxes.astype(np.float32)),
            "entity_labels":   torch.from_numpy(entity_labels.astype(np.int64)),
            "rel_pairs":       torch.from_numpy(rel_pairs.astype(np.int64)),
            "rel_labels":      torch.from_numpy(rel_labels.astype(np.int64)),
            "graph_adj":       torch.from_numpy(adj),
            "num_nodes":       N,
            "image_id":        idx,
            "feature_source":  "synthetic",
            "dataset":         "synthetic",
            "ontology_id":     "synthetic:vg150-like-v1",
            "num_entity_classes": NUM_ENTITY_CLASSES,
            "num_predicate_classes": NUM_REL_CLASSES,
        }
        return result


def _collate_fn(batch):
    """Return one graph without silently discarding a larger image batch."""
    if len(batch) != 1:
        raise ValueError(
            "SGG loaders require image batch_size=1 because graphs have variable "
            "numbers of nodes and relations. Use one process per GPU plus gradient "
            "accumulation for a larger effective batch."
        )
    return batch[0]


def build_synthetic_loader(num_samples: int = 500,
                            seed: int = 42) -> DataLoader:
    dataset = SyntheticSGGDataset(num_samples=num_samples, seed=seed)
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      collate_fn=_collate_fn, num_workers=0)


# ── Real VG Data ───────────────────────────────────────────────────────────────

class VGTestDataset(Dataset):
    """
    Visual Genome test-set loader — works WITHOUT roi_features.hdf5.

    Required files under data_root:
      VG_SGG_with_attri.h5          — scene graph annotations + boxes  (~500 MB)
      VG_SGG_dicts_with_attri.json  — vocabulary / class mappings       (~1 MB)

    Optional (improves proxy feature quality):
      image_data.json               — image width/height metadata       (~3 MB)
      VG_100K/                      — raw images (enables crop features)

    Feature modes (auto-selected):
      "roi_hdf5"       — real RoI-pool features if roi_features.hdf5 present
      "image_crop"     — per-box crop statistics from raw images (if VG_100K present)
      "geometry_proxy" — box geometry + label embeddings only (minimum requirement)
    """

    def __init__(self, data_root: str, num_samples: int = 500,
                 split: int = 2, include_proxy_features: bool = True,
                 require_relations: bool = True,
                 include_raw_images: bool = True):
        """
        split: 0=train, 1=val, 2=test
        """
        self.data_root   = Path(data_root)
        self.num_samples = num_samples
        self.split_id    = split
        self.include_proxy_features = bool(include_proxy_features)
        self.require_relations = bool(require_relations)
        self.include_raw_images = bool(include_raw_images)
        self.dataset_name = "vg"
        self._fallback   = False

        self._init_vg()

    def _init_vg(self):
        import h5py, json

        # ── Required files ───────────────────────────────────────────────────
        annot_candidates = [
            self.data_root / "VG_SGG_with_attri.h5",
            self.data_root / "VG-SGG-with-attri.h5",   # alternate naming
            self.data_root / "VG-SGG.h5",
        ]
        dict_candidates = [
            self.data_root / "VG_SGG_dicts_with_attri.json",
            self.data_root / "VG-SGG-dicts-with-attri.json",
            self.data_root / "VG-SGG-dicts.json",
        ]

        annot_path = next((p for p in annot_candidates if p.exists()), None)
        dict_path  = next((p for p in dict_candidates  if p.exists()), None)

        if annot_path is None:
            raise FileNotFoundError(
                f"Cannot find VG_SGG_with_attri.h5 in {self.data_root}\n"
                f"  Tried: {[str(p) for p in annot_candidates]}"
            )
        if dict_path is None:
            raise FileNotFoundError(
                f"Cannot find VG_SGG_dicts_with_attri.json in {self.data_root}\n"
                f"  Tried: {[str(p) for p in dict_candidates]}"
            )

        self.annot_file = h5py.File(annot_path, "r")

        with open(dict_path, "r") as f:
            self.sgg_dict = json.load(f)
        canonical = json.dumps(self.sgg_dict, sort_keys=True, separators=(",", ":"))
        import hashlib
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        self.ontology_id = f"vg150:{digest}"
        self.num_entity_classes = NUM_ENTITY_CLASSES
        self.num_predicate_classes = NUM_REL_CLASSES

        # ── Optional: roi_features.hdf5 ──────────────────────────────────────
        feat_path = self.data_root / "roi_features.hdf5"
        self.feat_file = h5py.File(feat_path, "r") if feat_path.exists() else None

        # ── Optional: raw images directory ───────────────────────────────────
        img_dir_candidates = [
            self.data_root / "VG_100K",
            self.data_root / "VG_100K_2",
            self.data_root / "images" / "VG_100K",
            self.data_root / "images" / "VG_100K_2",
            self.data_root / "images",
            self.data_root.parent / "VG_100K",
            self.data_root.parent / "VG_100K_2",
            self.data_root.parent / "images" / "VG_100K",
            self.data_root.parent / "images" / "VG_100K_2",
        ]
        self.img_dirs = []
        seen_dirs = set()
        for candidate in img_dir_candidates:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved not in seen_dirs:
                self.img_dirs.append(candidate)
                seen_dirs.add(resolved)
        # Kept for compatibility with callers that inspect the former single root.
        self.img_dir = self.img_dirs[0] if self.img_dirs else None

        # ── Build index for selected split ───────────────────────────────────
        split_arr  = self.annot_file["split"][:]
        self.index_to_image_meta = {}
        img_data_candidates = [
            self.data_root / "image_data.json",
            self.data_root.parent / "image_data.json",
            self.data_root.parent / "v1.2" / "image_data.json",
        ]
        img_data_path = next((p for p in img_data_candidates if p.is_file()), None)
        if img_data_path is not None:
            with open(img_data_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            if len(raw_meta) == len(split_arr):
                self.index_to_image_meta = dict(enumerate(raw_meta))
            else:
                filtered_meta = [
                    item for item in raw_meta
                    if int(item.get("image_id", -1)) not in VG_CORRUPT_IMAGE_IDS
                ]
                if len(filtered_meta) == len(split_arr):
                    self.index_to_image_meta = dict(enumerate(filtered_meta))
                    print(
                        "  [VGTestDataset] aligned image_data.json after removing "
                        "the four canonical corrupt VG images"
                    )
                else:
                    print(
                        f"  [VGTestDataset][WARN] {img_data_path} has {len(raw_meta)} "
                        f"records but H5 has {len(split_arr)}. Raw-image ID mapping is "
                        "disabled instead of guessing by row index."
                    )
        split_mask = split_arr == self.split_id
        candidate_indices = np.where(split_mask)[0]
        valid_indices = []
        skipped_no_boxes = 0
        for img_idx in candidate_indices:
            first_box = int(self.annot_file["img_to_first_box"][img_idx])
            last_box = int(self.annot_file["img_to_last_box"][img_idx])
            if first_box < 0 or first_box > last_box:
                skipped_no_boxes += 1
                continue
            if self.require_relations:
                first_rel = int(self.annot_file["img_to_first_rel"][img_idx])
                last_rel = int(self.annot_file["img_to_last_rel"][img_idx])
                if first_rel < 0 or first_rel > last_rel:
                    continue
            valid_indices.append(int(img_idx))
            if len(valid_indices) >= self.num_samples:
                break
        self.image_indices = np.asarray(valid_indices, dtype=np.int64)
        if len(self.image_indices) == 0:
            raise ValueError(
                f"VG split={self.split_id} has no images with annotated boxes in {self.data_root}"
            )

        # ── Report what we have ───────────────────────────────────────────────
        feat_mode = (
            "roi_hdf5" if self.include_proxy_features and self.feat_file
            else "image_crop" if (
                self.include_proxy_features and self.img_dir
                and self.index_to_image_meta
            )
            else "geometry_proxy" if self.include_proxy_features
            else "annotation_only"
        )
        print(f"  [VGTestDataset] split={self.split_id}  "
              f"n_images={len(self.image_indices)}  "
              f"feature_mode={feat_mode}")
        if skipped_no_boxes:
            print(f"  [VGTestDataset] skipped_no_box_images={skipped_no_boxes}")
        if self.include_proxy_features and self.feat_file is None:
            print(f"  [VGTestDataset] roi_features.hdf5 NOT found → "
                  f"using proxy features ({feat_mode})")
            print(f"                  Use this for diagnostics or negative controls; "
                  f"final visual-grounding claims need RoI/image features.")

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx: int) -> dict:
        img_idx = int(self.image_indices[idx])
        af      = self.annot_file

        # boxes_1024 uses a 1024-pixel max-side scale, so the image dimensions
        # are needed before annotations can be normalized.
        meta = self.index_to_image_meta.get(img_idx, {})
        img_id = int(meta["image_id"]) if "image_id" in meta else None
        img_w  = int(meta.get("width",  1024))
        img_h  = int(meta.get("height", 1024))

        # ── Box / entity annotations ─────────────────────────────────────────
        first_box = int(af["img_to_first_box"][img_idx])
        last_box  = int(af["img_to_last_box"][img_idx])

        if first_box < 0 or first_box > last_box:
            raise IndexError(
                f"Invalid VG index {img_idx}: no annotated boxes. "
                "This should have been filtered during dataset initialization."
            )

        boxes_slice = slice(first_box, last_box + 1)
        # Canonical VG-SGG stores boxes_1024 as [cx, cy, width, height].
        boxes_raw = af["boxes_1024"][boxes_slice].astype(np.float32)
        boxes_norm = vg_boxes_to_normalized_xyxy(boxes_raw, img_w, img_h)
        labels     = af["labels"][boxes_slice, 0].astype(np.int64)
        N          = len(labels)

        # ── Relation annotations ─────────────────────────────────────────────
        first_rel = int(af["img_to_first_rel"][img_idx])
        last_rel  = int(af["img_to_last_rel"][img_idx])

        if first_rel < 0 or first_rel > last_rel:
            rel_pairs  = np.zeros((0, 2), dtype=np.int64)
            rel_labels = np.zeros(0,      dtype=np.int64)
        else:
            rel_slice  = slice(first_rel, last_rel + 1)
            rel_pairs  = af["relationships"][rel_slice].astype(np.int64)
            rel_labels = af["predicates"][rel_slice, 0].astype(np.int64)
            # Relationships store global box indices; make them image-local
            rel_pairs  = rel_pairs - first_box
            rel_pairs  = rel_pairs.clip(0, N - 1)

        M = rel_pairs.shape[0]

        # Find image file path
        img_path = (
            find_vg_image(self.img_dirs, img_id)
            if img_id is not None else None
        )

        # ── Adjacency matrix ─────────────────────────────────────────────────
        adj = np.zeros((N, N), dtype=np.float32)
        for pair in rel_pairs:
            s, o = int(pair[0]), int(pair[1])
            if 0 <= s < N and 0 <= o < N:
                adj[s, o] = 1.0
                adj[o, s] = 1.0

        result = {
            "boxes":           torch.from_numpy(boxes_norm),
            "entity_labels":   torch.from_numpy(labels),
            "rel_pairs":       torch.from_numpy(rel_pairs),
            "rel_labels":      torch.from_numpy(rel_labels),
            "graph_adj":       torch.from_numpy(adj),
            "num_nodes":       N,
            "image_id":        img_id if img_id is not None else img_idx,
            "dataset_index":   img_idx,
            "image_id_source": "image_data_json" if img_id is not None else "dataset_index",
            "feature_source":  "annotation_only",
            "dataset":         "vg",
            "ontology_id":     self.ontology_id,
            "num_entity_classes": NUM_ENTITY_CLASSES,
            "num_predicate_classes": NUM_REL_CLASSES,
        }
        if self.include_proxy_features:
            if self.feat_file is not None:
                vis = np.asarray(
                    self.feat_file["features"][first_box:last_box + 1],
                    dtype=np.float32,
                )
                uni = np.zeros((max(M, 1), ROI_FEAT_DIM), dtype=np.float32)
                feat_source = "roi_hdf5"
            else:
                vis, uni, feat_source = build_proxy_features(
                    boxes_norm, labels, rel_pairs,
                    out_dim=ROI_FEAT_DIM,
                    img_path=img_path,
                    img_w=img_w, img_h=img_h,
                )
                if M == 0:
                    uni = np.zeros((1, ROI_FEAT_DIM), dtype=np.float32)
            result["visual_features"] = torch.from_numpy(vis)
            result["union_features"] = torch.from_numpy(uni)
            result["feature_source"] = feat_source
        if img_path is not None:
            result["image_path"] = str(img_path)
        if self.include_raw_images:
            image_tensor = load_image_tensor(img_path)
            if image_tensor is not None:
                result["image"] = image_tensor
        return result

def build_vg_test_loader(data_root: str,
                          num_samples: int = 500,
                          batch_size: int = 1,
                          split: int = 2,
                          include_proxy_features: bool = True,
                          require_relations: bool = True,
                          include_raw_images: bool = True) -> DataLoader:
    """
    Build a DataLoader over the VG test split.

    Works with ONLY VG_SGG_with_attri.h5 + VG_SGG_dicts_with_attri.json.
    Automatically upgrades feature quality if VG_100K/ or roi_features.hdf5 are present.
    """
    dataset = VGTestDataset(data_root=data_root,
                             num_samples=num_samples,
                             split=split,
                             include_proxy_features=include_proxy_features,
                             require_relations=require_relations,
                             include_raw_images=include_raw_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=_collate_fn, num_workers=0)
