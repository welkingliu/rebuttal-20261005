"""
utils/oi_data_utils.py
======================
Open Images V6/V7 Dataset Loader for the SGG Diagnostic Protocol
=================================================================

Open Images uses a fundamentally different annotation schema from Visual Genome.
This module handles all the differences and bridges them into the shared batch
format used by Steps 2–4.

Key differences vs VG:
  ┌─────────────────────┬────────────────────────┬──────────────────────────┐
  │ Property            │ Visual Genome          │ Open Images V6/V7        │
  ├─────────────────────┼────────────────────────┼──────────────────────────┤
  │ Entity classes      │ 150 integer IDs        │ 600 MID strings (/m/...) │
  │ Predicate classes   │ 50 integer IDs         │ 30 string labels         │
  │ Annotation format   │ .h5 + JSON vocab       │ CSV files                │
  │ Box storage         │ 1024-normalised xyxy   │ normalised XMin/XMax/    │
  │                     │                        │ YMin/YMax (per object)   │
  │ Relation row format │ pairs table            │ one row = one relation   │
  │                     │                        │ with both boxes inline   │
  │ Image files         │ VG_100K/               │ AWS S3 split into        │
  │                     │                        │ train_0…train_f shards   │
  └─────────────────────┴────────────────────────┴──────────────────────────┘

Required files under oi_data_root/
-----------------------------------
  annotations/
    class-descriptions-boxable.csv      ← 600 boxable MID → name mappings
    oidv6-validation-annotations-vrd.csv ← used as default test split
  Optional:
    oidv6-test-annotations-vrd.csv
    oidv6-train-annotations-vrd.csv
    oidv6-relationships-description.csv ← relationship label → human name

  images/ (optional — enables image-crop proxy features)
    <image_id>.jpg

Use ``tools/download_openimages.py`` so the annotation names, exact image
selection, and coverage manifests stay consistent with this loader.

Batch schema (identical keys to VG loader for full model compatibility):
  visual_features : [N, 4096]
  union_features  : [M, 4096]
  boxes           : [N, 4]     xyxy normalised [0,1]
  entity_labels   : [N]        integer IDs (mapped from MIDs)
  rel_pairs       : [M, 2]
  rel_labels      : [M]        integer IDs (mapped from predicate strings)
  graph_adj       : [N, N]
  num_nodes       : int
  image_id        : str
  feature_source  : str
  dataset         : "openimages"
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sgg_core.data.data_utils import (
    build_proxy_features, load_image_tensor, _collate_fn, ROI_FEAT_DIM,
)


# ── OI constants ───────────────────────────────────────────────────────────────

OI_NUM_ENTITY_CLASSES = 602   # 601 official SGG classes + background
OI_NUM_REL_CLASSES    = 31    # 30 official SGG predicates + background

# Canonical OI-VRD predicates (index 0 = background). The relationship
# description CSV is preferred because its row order defines the ontology.
OI_PREDICATES = [
    "background",
    "at", "holds", "wears", "surf", "hang", "drink", "holding_hands",
    "on", "ride", "dance", "skateboard", "catch", "highfive",
    "inside_of", "eat", "cut", "contain", "handshake", "kiss",
    "talk_on_phone", "interacts_with", "under", "hug", "throw", "hits",
    "snowboard", "is", "kick", "ski", "plays", "read",
]

OI_PRED_TO_ID: Dict[str, int] = {p: i for i, p in enumerate(OI_PREDICATES)}


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class OIVocabulary:
    """
    Maps OI MID strings → integer entity IDs and predicate strings → integer IDs.
    Exposes a sgg_dict property for graph_audit vocabulary lookup compatibility.
    """

    _mid_cache = {}

    @classmethod
    def _annotation_mids(cls, paths: List[Path], cache_path: Path) -> set:
        signature = [
            {"path": str(path), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ]
        cache_key = tuple(
            (item["path"], item["size"], item["mtime_ns"])
            for item in signature
        )
        if cache_key in cls._mid_cache:
            return cls._mid_cache[cache_key]
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("source_signature") == signature:
                cls._mid_cache[cache_key] = set(payload.get("mids", []))
                return cls._mid_cache[cache_key]
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass

        mids = set()
        for path in paths:
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    mids.add(row["LabelName1"].strip())
                    mids.add(row["LabelName2"].strip())
        cls._mid_cache[cache_key] = mids
        payload = {
            "source_signature": signature,
            "mids": sorted(mids),
        }
        temporary = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            temporary.replace(cache_path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return mids

    def __init__(self, class_desc_csv: Path,
                 rel_desc_csv: Optional[Path] = None,
                 annotation_csvs: Optional[List[Path]] = None,
                 ontology_json: Optional[Path] = None):
        # Entity vocab: MID → int
        self.mid_to_id:   Dict[str, int] = {}
        self.id_to_name:  Dict[int, str] = {0: "background"}
        self.mid_to_name: Dict[str, str] = {}

        display_names = {}
        if class_desc_csv.exists():
            with open(class_desc_csv, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue
                    mid, name = row[0].strip(), row[1].strip()
                    if mid.lower() == "labelname":
                        continue
                    display_names[mid] = name
        else:
            raise FileNotFoundError(f"OpenImages class descriptions not found: {class_desc_csv}")

        ontology_json = Path(ontology_json) if ontology_json else None
        self.is_official_sgg_ontology = bool(
            ontology_json is not None and ontology_json.is_file()
        )
        if self.is_official_sgg_ontology:
            payload = json.loads(ontology_json.read_text(encoding="utf-8"))
            object_categories = payload.get("object_categories", [])
            predicate_names = payload.get("predicate_categories", [])
            if len(object_categories) != 601 or len(predicate_names) != 30:
                raise ValueError(
                    "Official OI-SGG ontology must contain 601 objects and "
                    f"30 predicates: {ontology_json}"
                )
            for index, record in enumerate(object_categories, start=1):
                mid = str(record["mid"]).strip()
                name = str(record["name"]).strip()
                if not mid or mid in self.mid_to_id:
                    raise ValueError(f"Invalid or duplicate OI MID: {mid!r}")
                self.mid_to_id[mid] = index
                self.mid_to_name[mid] = name
                self.id_to_name[index] = name
            predicate_names = [
                str(name).strip().lower().replace(" ", "_")
                for name in predicate_names
            ]
        else:
            annotation_csvs = sorted({
                Path(path).resolve()
                for path in (annotation_csvs or []) if Path(path).is_file()
            })
            annotation_mids = self._annotation_mids(
                annotation_csvs,
                class_desc_csv.parent / ".sgg_ontology_cache.json",
            ) if annotation_csvs else set()
            all_mids = sorted(set(display_names) | annotation_mids)
            for index, mid in enumerate(all_mids, start=1):
                self.mid_to_id[mid] = index
                self.mid_to_name[mid] = display_names.get(mid, mid)
                self.id_to_name[index] = self.mid_to_name[mid]

            predicate_names = []
            if rel_desc_csv is not None and rel_desc_csv.exists():
                with open(rel_desc_csv, newline="", encoding="utf-8") as f:
                    for row in csv.reader(f):
                        if row and row[0].strip():
                            predicate_names.append(row[0].strip().lower())
            else:
                predicate_names = OI_PREDICATES[1:]
        self.pred_to_id = {
            label: index for index, label in enumerate(predicate_names, start=1)
        }
        self.pred_id_to_name = {0: "background"}
        self.pred_id_to_name.update({
            index: label for label, index in self.pred_to_id.items()
        })

        self.num_entity_classes = max(self.mid_to_id.values(), default=600) + 1
        self.num_pred_classes   = max(self.pred_to_id.values()) + 1

    def mid_to_int(self, mid: str) -> int:
        if mid not in self.mid_to_id:
            raise KeyError(f"OpenImages MID is outside the declared ontology: {mid}")
        return self.mid_to_id[mid]

    def predicate_to_int(self, label: str) -> int:
        key = label.strip().lower()
        if key not in self.pred_to_id:
            raise KeyError(f"OpenImages predicate is outside the declared ontology: {key}")
        return self.pred_to_id[key]

    def supports_predicate(self, label: str) -> bool:
        return label.strip().lower() in self.pred_to_id

    @property
    def sgg_dict(self) -> dict:
        """Compatible with graph_audit._build_vocab_from_loader()."""
        return {
            "idx_to_label":     {str(k): v for k, v in self.id_to_name.items()},
            "idx_to_predicate": {str(k): v for k, v in self.pred_id_to_name.items()},
        }


# ── CSV annotation index ───────────────────────────────────────────────────────

class OIAnnotationIndex:
    """
    Parses a VRD CSV and builds a per-image index.

    CSV columns (V6 format):
      ImageID, LabelName1, LabelName2,
      XMin1, XMax1, YMin1, YMax1,
      XMin2, XMax2, YMin2, YMax2,
      RelationshipLabel
    """

    def __init__(self, vrd_csv: Path, vocab: OIVocabulary,
                 max_images: Optional[int] = None):
        self.vocab  = vocab
        self._index: Dict[str, List[dict]] = defaultdict(list)

        print(f"  [OIIndex] Parsing {vrd_csv.name} …", flush=True)
        image_ids = set()
        skipped_predicates = 0
        with open(vrd_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not vocab.supports_predicate(row["RelationshipLabel"]):
                    skipped_predicates += 1
                    continue
                image_ids.add(row["ImageID"].strip())
        ordered_ids = sorted(image_ids)
        if max_images is not None:
            ordered_ids = ordered_ids[:max(0, int(max_images))]
        selected_ids = set(ordered_ids)

        row_count = 0
        with open(vrd_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not vocab.supports_predicate(row["RelationshipLabel"]):
                    continue
                iid = row["ImageID"].strip()
                if iid not in selected_ids:
                    continue
                self._index[iid].append({
                    "mid1": row["LabelName1"].strip(),
                    "mid2": row["LabelName2"].strip(),
                    # OI box format: XMin, YMin=YMin1, XMax, YMax (xyxy)
                    "box1": (float(row["XMin1"]), float(row["YMin1"]),
                             float(row["XMax1"]), float(row["YMax1"])),
                    "box2": (float(row["XMin2"]), float(row["YMin2"]),
                             float(row["XMax2"]), float(row["YMax2"])),
                    "pred": row["RelationshipLabel"].strip().lower(),
                })
                row_count += 1

        self._image_ids = [iid for iid in ordered_ids if iid in self._index]

        print(f"  [OIIndex] {len(self._image_ids)} images, "
              f"{row_count} relation rows loaded, "
              f"{skipped_predicates} non-SGG rows skipped.")

    def __len__(self) -> int:
        return len(self._image_ids)

    def image_id(self, idx: int) -> str:
        return self._image_ids[idx]

    def relations(self, image_id: str) -> List[dict]:
        return self._index.get(image_id, [])


# ── Scene graph builder ────────────────────────────────────────────────────────

def _build_scene_graph(relations: List[dict],
                        vocab: OIVocabulary
                        ) -> Tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert OI relation list → (boxes, entity_labels, rel_pairs, rel_labels, adj).

    OI encodes both subject and object boxes inline per relation row.
    We deduplicate objects by (MID, rounded_box) key, then index pairs.
    """
    if not relations:
        boxes      = np.array([[0.1,0.1,0.5,0.5],[0.5,0.5,0.9,0.9]], np.float32)
        entity_lbl = np.array([1, 2], np.int64)
        rel_pairs  = np.array([[0, 1]], np.int64)
        rel_labels = np.array([1], np.int64)
        adj        = np.array([[0,1],[1,0]], np.float32)
        return boxes, entity_lbl, rel_pairs, rel_labels, adj

    # Deduplicate objects: (mid, box_3dp) → index
    key_to_idx:  Dict[tuple, int] = {}
    obj_boxes:   List[tuple]      = []
    obj_labels:  List[int]        = []

    def _get_or_add(mid: str, box: tuple) -> int:
        key = (mid, tuple(round(v, 3) for v in box))
        if key not in key_to_idx:
            idx = len(obj_boxes)
            key_to_idx[key] = idx
            obj_boxes.append(box)
            obj_labels.append(vocab.mid_to_int(mid))
        return key_to_idx[key]

    rel_pairs_list:  List[Tuple[int, int]] = []
    rel_labels_list: List[int]             = []

    for rel in relations:
        s_idx = _get_or_add(rel["mid1"], rel["box1"])
        o_idx = _get_or_add(rel["mid2"], rel["box2"])
        p_id  = vocab.predicate_to_int(rel["pred"])
        if s_idx != o_idx and p_id > 0:
            rel_pairs_list.append((s_idx, o_idx))
            rel_labels_list.append(p_id)

    N = len(obj_boxes)
    if N == 0:
        return _build_scene_graph([], vocab)

    boxes      = np.array(obj_boxes,  dtype=np.float32).clip(0.0, 1.0)
    entity_lbl = np.array(obj_labels, dtype=np.int64)

    if rel_pairs_list:
        rel_pairs  = np.array(rel_pairs_list,  dtype=np.int64)
        rel_labels = np.array(rel_labels_list, dtype=np.int64)
    else:
        rel_pairs  = np.zeros((0, 2), dtype=np.int64)
        rel_labels = np.zeros(0,      dtype=np.int64)

    adj = np.zeros((N, N), dtype=np.float32)
    for s, o in rel_pairs:
        if 0 <= s < N and 0 <= o < N:
            adj[s, o] = 1.0
            adj[o, s] = 1.0

    return boxes, entity_lbl, rel_pairs, rel_labels, adj


# ── Dataset class ──────────────────────────────────────────────────────────────

class OpenImagesVRDDataset(Dataset):
    """
    Open Images V6/V7 VRD test-set loader.

    Implements the identical batch schema as VGTestDataset — drop-in compatible
    with all SGG model wrappers and all three audit steps.
    """

    SPLIT_FILES = {
        "train":      ["oidv6-train-annotations-vrd.csv", "train-annotations-vrd.csv", "train/vrd.csv"],
        "validation": ["oidv6-validation-annotations-vrd.csv", "validation-annotations-vrd.csv", "validation/vrd.csv"],
        "val":        ["oidv6-validation-annotations-vrd.csv", "validation-annotations-vrd.csv", "validation/vrd.csv"],
        "test":       ["oidv6-test-annotations-vrd.csv", "test-annotations-vrd.csv", "test/vrd.csv"],
    }

    def __init__(self, data_root: str, split: str = "validation",
                 num_samples: int = 500, include_proxy_features: bool = True,
                 include_raw_images: bool = True):
        self.data_root = Path(data_root)
        self.include_proxy_features = bool(include_proxy_features)
        self.include_raw_images = bool(include_raw_images)
        ann_dir        = self.data_root / "annotations"
        img_dir        = self.data_root / "images"

        split_key = split.lower()
        candidates = self.SPLIT_FILES.get(split_key)
        if candidates is None:
            raise ValueError(f"Unknown split '{split}'. Use: {list(self.SPLIT_FILES)}")
        vrd_csv = self._find(ann_dir, candidates, required=False)
        if vrd_csv is None or not vrd_csv.exists():
            raise FileNotFoundError(
                f"OpenImages split '{split}' is unavailable in {ann_dir}. "
                f"Expected one of: {candidates}. Cross-split fallback is disabled "
                "because it can leak evaluation annotations into fitted controls."
            )

        # ── Vocabulary ─────────────────────────────────────────────────────────
        # Prefer the 600-class boxable vocabulary.  The similarly named
        # oidv6-class-descriptions.csv contains the full ~20k image-level
        # ontology and is incompatible with Open Images VRD detector heads.
        class_csv = self._find(ann_dir, [
            "class-descriptions-boxable.csv",
            "oidv7-class-descriptions-boxable.csv",
            "oidv6-class-descriptions-boxable.csv"])
        rel_csv = self._find(ann_dir, [
            "oidv6-relationships-description.csv",
            "relationships_description.csv"], required=False)

        ontology_csvs = {
            ann_dir / name
            for names in self.SPLIT_FILES.values()
            for name in names
            if (ann_dir / name).exists()
        }
        ontology_json = ann_dir / "oi_sgg_ontology.json"
        self.vocab = OIVocabulary(
            class_csv, rel_csv, annotation_csvs=list(ontology_csvs),
            ontology_json=ontology_json,
        )
        self.sgg_dict = self.vocab.sgg_dict   # for graph_audit vocab lookup
        canonical = repr(sorted(self.vocab.mid_to_id.items())) + repr(
            sorted(self.vocab.pred_to_id.items())
        )
        self.ontology_id = "openimages:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16]
        self.num_entity_classes = self.vocab.num_entity_classes
        self.num_predicate_classes = self.vocab.num_pred_classes

        self._index = OIAnnotationIndex(vrd_csv, self.vocab, num_samples)
        self.split = "validation" if split_key == "val" else split_key
        self.img_dir = img_dir if img_dir.exists() else None

        feat_mode = (
            "image_crop" if self.include_proxy_features and self.img_dir
            else "geometry_proxy" if self.include_proxy_features
            else "annotation_only"
        )
        print(f"  [OI] {len(self._index)} images  "
              f"entities={self.vocab.num_entity_classes}  "
              f"predicates={self.vocab.num_pred_classes}  "
              f"feature_mode={feat_mode}")

    @staticmethod
    def _find(directory: Path, candidates: List[str],
              required: bool = True) -> Path:
        for name in candidates:
            p = directory / name
            if p.exists():
                return p
        if required:
            raise FileNotFoundError(
                f"Required file not found in {directory}. "
                f"Tried: {candidates}")
        return directory / candidates[0]

    def _locate_image(self, image_id: str) -> Optional[Path]:
        if self.img_dir is None:
            return None
        # Canonical layout is images/<split>/<ImageID>.jpg. The root fallback
        # preserves compatibility with data downloaded by older project code.
        roots = (self.img_dir / self.split, self.img_dir)
        for suffix in [".jpg", ".jpeg", ".png"]:
            for root in roots:
                p = root / f"{image_id}{suffix}"
                if p.exists():
                    return p
                for shard in (image_id[0].lower(), image_id[0].upper()):
                    p = root / shard / f"{image_id}{suffix}"
                    if p.exists():
                        return p
        return None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        iid       = self._index.image_id(idx)
        relations = self._index.relations(iid)

        boxes, entity_lbl, rel_pairs, rel_labels, adj = \
            _build_scene_graph(relations, self.vocab)

        N = boxes.shape[0]
        M = rel_pairs.shape[0]

        img_path = self._locate_image(iid)
        result = {
            "boxes":           torch.from_numpy(boxes),
            "entity_labels":   torch.from_numpy(entity_lbl),
            "rel_pairs":       torch.from_numpy(rel_pairs),
            "rel_labels":      torch.from_numpy(rel_labels),
            "graph_adj":       torch.from_numpy(adj),
            "num_nodes":       N,
            "image_id":        iid,
            "feature_source":  "annotation_only",
            "dataset":         "openimages",
            "ontology_id":     self.ontology_id,
            "num_entity_classes": self.vocab.num_entity_classes,
            "num_predicate_classes": self.vocab.num_pred_classes,
        }
        if self.include_proxy_features:
            vis, uni, feat_src = build_proxy_features(
                boxes, entity_lbl, rel_pairs,
                out_dim=ROI_FEAT_DIM,
                img_path=img_path,
                img_w=1, img_h=1,   # OI boxes are already normalised
            )
            if M == 0:
                uni = np.zeros((1, ROI_FEAT_DIM), dtype=np.float32)
            result["visual_features"] = torch.from_numpy(vis)
            result["union_features"] = torch.from_numpy(uni)
            result["feature_source"] = feat_src
        if img_path is not None:
            result["image_path"] = str(img_path)
        if self.include_raw_images:
            image_tensor = load_image_tensor(img_path)
            if image_tensor is not None:
                result["image"] = image_tensor
        return result


# ── Public API ─────────────────────────────────────────────────────────────────

def build_oi_loader(data_root:   str,
                    split:       str = "validation",
                    num_samples: int = 500,
                    batch_size:  int = 1,
                    include_proxy_features: bool = True,
                    include_raw_images: bool = True) -> DataLoader:
    """
    Build a DataLoader for Open Images VRD.
    Works with annotation CSVs only — no images required.

    Args:
        data_root   : directory containing annotations/ subdirectory
        split       : "train" | "validation" | "test"
        num_samples : maximum images to evaluate
        batch_size  : always 1 for SGG diagnostic protocol
    """
    dataset = OpenImagesVRDDataset(
        data_root=data_root, split=split, num_samples=num_samples,
        include_proxy_features=include_proxy_features,
        include_raw_images=include_raw_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=_collate_fn, num_workers=0)


def check_oi_files(data_root: str) -> dict:
    """Check which OI files are present; print status and download hints."""
    root    = Path(data_root)
    ann_dir = root / "annotations"
    img_dir = root / "images"

    status = {"ok": True, "missing": [], "feat_mode": "geometry_proxy"}

    required = {
        "Boxable class descriptions":
            ["class-descriptions-boxable.csv",
             "oidv7-class-descriptions-boxable.csv",
             "oidv6-class-descriptions-boxable.csv"],
        "VRD annotations (validation or test CSV)":
            ["oidv6-validation-annotations-vrd.csv", "oidv6-test-annotations-vrd.csv",
             "validation-annotations-vrd.csv", "test-annotations-vrd.csv",
             "oidv6-train-annotations-vrd.csv", "validation/vrd.csv", "test/vrd.csv"],
    }
    optional = {
        "Relationship descriptions": ["oidv6-relationships-description.csv", "relationships_description.csv"],
    }

    print(f"\n  {'─'*60}")
    print(f"  Open Images file check:  {root}")
    print(f"  {'─'*60}")

    for label, candidates in required.items():
        found = next((ann_dir / c for c in candidates
                      if (ann_dir / c).exists()), None)
        tag = f"  ← REQUIRED" if not found else ""
        print(f"  [{'OK' if found else 'MISS'}] {label}")
        print(f"         {'✓  '+found.name if found else '✗  not found'+tag}")
        if not found:
            status["ok"] = False
            status["missing"].append(label)

    print()
    for label, candidates in optional.items():
        found = next((ann_dir / c for c in candidates
                      if (ann_dir / c).exists()), None)
        print(f"  [{'OK' if found else '  '}] {label}: "
              f"{'found' if found else 'not found (optional)'}")

    img_ok = img_dir.exists()
    print(f"  [{'OK' if img_ok else '  '}] Images directory ({img_dir.name}/): "
          f"{'found → image_crop features' if img_ok else 'not found → geometry_proxy'}")
    if img_ok:
        status["feat_mode"] = "image_crop"

    if not status["ok"]:
        print(f"\n  To download required files:")
        print(f"    mkdir -p {ann_dir}")
        print(f"    python tools/download_openimages.py --out_dir {root} "
              "--split validation --annotations_only")

    return status
