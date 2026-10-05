import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sgg_core.data.data_utils import (
    vg_boxes_to_normalized_xyxy,
    vg_cxcywh_to_xyxy,
)
from sgg_core.data.gqa_psg_data_utils import build_gqa_loader, build_psg_loader
from sgg_core.data.oi_data_utils import OIVocabulary
from sgg_core.data.vrd_data_utils import build_vrd_loader


class DatasetOntologyTest(unittest.TestCase):
    def test_vg_boxes_1024_are_center_size_not_xyxy(self):
        boxes = np.asarray([
            [511.0, 356.0, 1023.0, 713.0],
            [560.0, 578.0, 925.0, 372.0],
        ], dtype=np.float32)
        converted = vg_cxcywh_to_xyxy(boxes)
        np.testing.assert_allclose(
            converted[0], [0.0, 0.0, 1022.5, 712.5], atol=1e-5
        )
        np.testing.assert_allclose(
            converted[1], [97.5, 392.0, 1022.5, 764.0], atol=1e-5
        )

    def test_vg_boxes_are_recovered_with_max_image_side(self):
        box = np.asarray([[512.0, 256.0, 512.0, 256.0]], dtype=np.float32)
        normalized = vg_boxes_to_normalized_xyxy(
            box, image_width=800, image_height=400
        )
        np.testing.assert_allclose(
            normalized[0], [0.25, 0.25, 0.75, 0.75], atol=1e-6
        )

    def test_openimages_vocabulary_is_exposed_to_graph_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = root / "classes.csv"
            relations = root / "relations.csv"
            classes.write_text("/m/person,Person\n/m/car,Car\n")
            relations.write_text("on\nnear\n")
            vocab = OIVocabulary(classes, relations)
            self.assertEqual(vocab.sgg_dict["idx_to_predicate"]["1"], "on")
            self.assertIn("Person", vocab.sgg_dict["idx_to_label"].values())

    def test_psg_xyxy_boxes_and_full_predicate_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = Path(tmp) / "psg.json"
            payload = {
                "thing_classes": ["person", "car"],
                "stuff_classes": [],
                "predicate_classes": [f"relation-{i}" for i in range(60)],
                "data": [{
                    "image_id": 1,
                    "file_name": "one.jpg",
                    "width": 100,
                    "height": 200,
                    "segments_info": [
                        {"id": 10, "category_id": 0},
                        {"id": 11, "category_id": 1},
                    ],
                    "annotations": [
                        {"bbox": [10, 20, 50, 60], "bbox_mode": 0, "category_id": 0},
                        {"bbox": [50, 100, 100, 200], "bbox_mode": 0, "category_id": 1},
                    ],
                    "relations": [[0, 1, 59]],
                }],
            }
            annotation.write_text(json.dumps(payload))
            loader = build_psg_loader(str(annotation), num_samples=1)
            batch = next(iter(loader))
            self.assertEqual(loader.dataset.num_predicate_classes, 61)
            self.assertEqual(int(batch["rel_labels"][0]), 60)
            self.assertAlmostEqual(float(batch["boxes"][0, 0]), 0.1)
            self.assertAlmostEqual(float(batch["boxes"][0, 3]), 0.3)

    def test_gqa_eval_uses_training_ontology(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.json"
            eval_path = Path(tmp) / "eval.json"
            train = {
                "train": {
                    "width": 100, "height": 100,
                    "objects": {
                        "1": {"name": "person", "x": 0, "y": 0, "w": 20, "h": 20,
                              "relations": [{"name": "beside", "object": "2"}]},
                        "2": {"name": "car", "x": 30, "y": 0, "w": 20, "h": 20,
                              "relations": []},
                    },
                }
            }
            evaluation = {
                "eval": {
                    "width": 100, "height": 100,
                    "objects": {
                        "1": {"name": "car", "x": 0, "y": 0, "w": 20, "h": 20,
                              "relations": [{"name": "beside", "object": "2"}]},
                        "2": {"name": "person", "x": 30, "y": 0, "w": 20, "h": 20,
                              "relations": []},
                    },
                }
            }
            train_path.write_text(json.dumps(train))
            eval_path.write_text(json.dumps(evaluation))
            train_loader = build_gqa_loader(
                str(train_path), 1, vocabulary_path=str(train_path)
            )
            eval_loader = build_gqa_loader(
                str(eval_path), 1, vocabulary_path=str(train_path)
            )
            self.assertEqual(
                train_loader.dataset.ontology_id, eval_loader.dataset.ontology_id
            )
            self.assertEqual(
                train_loader.dataset.obj_vocab, eval_loader.dataset.obj_vocab
            )

            annotation_only = build_gqa_loader(
                str(eval_path), 1, vocabulary_path=str(train_path),
                include_proxy_features=False,
            )
            batch = next(iter(annotation_only))
            self.assertNotIn("visual_features", batch)
            self.assertEqual(batch["feature_source"], "annotation_only")

    def test_gqa_sampling_continues_past_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "available.jpg").write_bytes(b"present")

            def graph():
                return {
                    "width": 100, "height": 100,
                    "objects": {
                        "1": {"name": "person", "x": 0, "y": 0,
                              "w": 20, "h": 20,
                              "relations": [{"name": "beside", "object": "2"}]},
                        "2": {"name": "car", "x": 30, "y": 0,
                              "w": 20, "h": 20, "relations": []},
                    },
                }

            scene_graph = root / "gqa.json"
            scene_graph.write_text(json.dumps({
                "missing": graph(),
                "available": graph(),
            }))
            loader = build_gqa_loader(
                str(scene_graph), 1, image_root=str(images),
                include_proxy_features=False, include_raw_images=False,
            )
            self.assertEqual(loader.dataset.items[0][0], "available")

    def test_psg_sampling_continues_past_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "available.jpg").write_bytes(b"present")

            def record(image_id, file_name):
                return {
                    "image_id": image_id,
                    "file_name": file_name,
                    "width": 100,
                    "height": 100,
                    "segments_info": [
                        {"id": 10, "category_id": 0},
                        {"id": 11, "category_id": 1},
                    ],
                    "annotations": [
                        {"bbox": [0, 0, 20, 20], "bbox_mode": 0,
                         "category_id": 0},
                        {"bbox": [30, 0, 50, 20], "bbox_mode": 0,
                         "category_id": 1},
                    ],
                    "relations": [[0, 1, 0]],
                }

            annotation = root / "psg.json"
            annotation.write_text(json.dumps({
                "thing_classes": ["person", "car"],
                "predicate_classes": ["beside"],
                "data": [
                    record(1, "missing.jpg"),
                    record(2, "available.jpg"),
                ],
            }))
            loader = build_psg_loader(
                str(annotation), 1, image_root=str(images),
                include_proxy_features=False, include_raw_images=False,
            )
            self.assertEqual(loader.dataset.items[0][0], 2)

    def test_vrd_keeps_all_seventy_predicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "json_dataset"
            annotations.mkdir()
            (annotations / "objects.json").write_text(
                json.dumps([f"object-{i}" for i in range(100)])
            )
            (annotations / "predicates.json").write_text(
                json.dumps([f"predicate-{i}" for i in range(70)])
            )
            relation = {
                "predicate": 69,
                "subject": {"category": 99, "bbox": [10, 30, 20, 60]},
                "object": {"category": 0, "bbox": [40, 80, 50, 100]},
            }
            (annotations / "annotations_test.json").write_text(
                json.dumps({"one.jpg": [relation]})
            )
            loader = build_vrd_loader(str(root), "test", 1)
            batch = next(iter(loader))
            self.assertEqual(loader.dataset.num_predicate_classes, 71)
            self.assertEqual(int(batch["rel_labels"][0]), 70)
            self.assertEqual(int(batch["entity_labels"].max()), 100)


if __name__ == "__main__":
    unittest.main()
