import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sgg_core.tools.download_openimages import (
    PROFILES,
    _valid_annotation,
    build_plan,
    get_image_ids_from_rel_csv,
    image_coverage,
    required_annotation_names,
)
from sgg_core.data.oi_data_utils import OpenImagesVRDDataset
from sgg_core.data.data_utils import find_vg_image, load_rgb_image
from sgg_core.tools.prepare_reviewer_datasets import _check_psg_images
from scripts.build_psg_official_test_split import build_split


VRD_COLUMNS = [
    "ImageID", "LabelName1", "LabelName2",
    "XMin1", "XMax1", "YMin1", "YMax1",
    "XMin2", "XMax2", "YMin2", "YMax2",
    "RelationshipLabel",
]


def _write_vrd(path: Path, image_ids):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VRD_COLUMNS)
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({
                "ImageID": image_id,
                "LabelName1": "/m/person",
                "LabelName2": "/m/car",
                "XMin1": 0.1, "XMax1": 0.4,
                "YMin1": 0.1, "YMax1": 0.4,
                "XMin2": 0.5, "XMax2": 0.9,
                "YMin2": 0.5, "YMax2": 0.9,
                "RelationshipLabel": "on",
            })


class DataPreparationTest(unittest.TestCase):
    def test_truncated_jpeg_recovery_is_explicit_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.jpg"
            pixels = np.random.default_rng(7).integers(
                0, 256, size=(64, 64, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(path, format="JPEG", quality=90)
            path.write_bytes(path.read_bytes()[:-10])

            first, first_status = load_rgb_image(path)
            second, second_status = load_rgb_image(path)

            self.assertEqual(first_status, "truncated_recovery")
            self.assertEqual(second_status, first_status)
            self.assertIsNotNone(first)
            self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_vg_image_lookup_checks_both_official_sibling_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "VG_100K"
            second = root / "VG_100K_2"
            first.mkdir()
            second.mkdir()
            expected = second / "42.jpg"
            expected.write_bytes(b"image")
            self.assertEqual(find_vg_image([first, second], 42), expected)

    def test_vrd_preparation_does_not_require_unused_attributes(self):
        required = required_annotation_names(["train", "validation"])
        self.assertNotIn("oidv6-attributes-description.csv", required)
        self.assertIn("oidv6-train-annotations-vrd.csv", required)
        self.assertIn("oidv6-validation-annotations-vrd.csv", required)

    def test_image_selection_matches_loader_lexicographic_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = Path(tmp) / "vrd.csv"
            _write_vrd(annotation, ["b", "a", "c", "a"])
            self.assertEqual(
                get_image_ids_from_rel_csv(annotation), ["a", "b", "c"]
            )

    def test_coverage_is_for_selected_ids_not_directory_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp)
            (images / "a.jpg").write_bytes(b"present")
            (images / "unrelated.jpg").write_bytes(b"present")
            coverage = image_coverage(images, ["a", "b"])
            self.assertEqual(coverage["present"], ["a"])
            self.assertEqual(coverage["missing"], ["b"])

    def test_split_coverage_accepts_legacy_flat_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_root = Path(tmp) / "images"
            split_root = images_root / "validation"
            split_root.mkdir(parents=True)
            (images_root / "legacy.jpg").write_bytes(b"present")
            (split_root / "canonical.jpg").write_bytes(b"present")
            coverage = image_coverage(
                split_root, ["canonical", "legacy"],
                fallback_dirs=[images_root],
            )
            self.assertEqual(coverage["present"], ["canonical", "legacy"])

    def test_openimages_loader_prefers_split_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations"
            split_images = root / "images" / "validation"
            annotations.mkdir()
            split_images.mkdir(parents=True)
            (annotations / "class-descriptions-boxable.csv").write_text(
                "LabelName,DisplayName\n/m/person,Person\n/m/car,Car\n"
            )
            (annotations / "oidv6-relationships-description.csv").write_text(
                "on,on\n"
            )
            _write_vrd(
                annotations / "oidv6-validation-annotations-vrd.csv", ["a"]
            )
            (split_images / "a.jpg").write_bytes(b"split")
            (root / "images" / "a.jpg").write_bytes(b"legacy")
            dataset = OpenImagesVRDDataset(
                str(root), "validation", 1,
                include_proxy_features=False, include_raw_images=False,
            )
            self.assertEqual(dataset._locate_image("a"), split_images / "a.jpg")

    def test_http_error_payload_is_not_a_valid_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oidv6-validation-annotations-vrd.csv"
            path.write_text("<Error><Code>AccessDenied</Code></Error>")
            self.assertFalse(_valid_annotation(path, path.name))

    def test_reduced_profile_has_full_validation_and_bounded_train(self):
        args = argparse.Namespace(
            profile="reduced_2gpu", max_images=None, split=None,
            include_train=False,
        )
        self.assertEqual(build_plan(args), PROFILES["reduced_2gpu"])
        self.assertEqual(build_plan(args)["train"], 1500)
        self.assertEqual(build_plan(args)["validation"], 0)

    def test_boxable_vocabulary_wins_over_full_class_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "class-descriptions-boxable.csv").write_text(
                "LabelName,DisplayName\n/m/person,Person\n/m/car,Car\n"
            )
            (annotations / "oidv6-class-descriptions.csv").write_text(
                "/m/unrelated,Unrelated image-level class\n"
            )
            (annotations / "oidv6-relationships-description.csv").write_text(
                "on,on\n"
            )
            _write_vrd(
                annotations / "oidv6-validation-annotations-vrd.csv", ["a"]
            )
            dataset = OpenImagesVRDDataset(
                str(root), "validation", 1,
                include_proxy_features=False, include_raw_images=False,
            )
            self.assertNotIn("/m/unrelated", dataset.vocab.mid_to_id)
            self.assertEqual(dataset.vocab.mid_to_name["/m/person"], "Person")

    def test_full_psg_coverage_does_not_hide_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "val2017").mkdir()
            (root / "panoptic_val2017").mkdir()
            (root / "val2017" / "present.jpg").write_bytes(b"image")
            (root / "panoptic_val2017" / "present.png").write_bytes(b"mask")
            annotation = root / "psg.json"
            records = [
                {
                    "image_id": 1,
                    "file_name": "val2017/missing.jpg",
                    "pan_seg_file_name": "panoptic_val2017/missing.png",
                    "segments_info": [{"id": 1}],
                    "relations": [[0, 0, 0]],
                },
                {
                    "image_id": 2,
                    "file_name": "val2017/present.jpg",
                    "pan_seg_file_name": "panoptic_val2017/present.png",
                    "segments_info": [{"id": 2}],
                    "relations": [[0, 0, 0]],
                },
            ]
            annotation.write_text(json.dumps({"data": records}))

            sampled = _check_psg_images(annotation, root, 1, False)
            complete = _check_psg_images(annotation, root, 0, False)

            self.assertTrue(sampled["complete"])
            self.assertEqual(sampled["present"], 1)
            self.assertEqual(sampled["selection_skipped_missing_images"], 1)
            self.assertEqual(complete["expected"], 2)
            self.assertEqual(complete["present"], 1)
            self.assertEqual(complete["missing"], 1)
            self.assertFalse(complete["complete"])

    def test_psg_official_split_uses_declared_test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "psg.json"
            source.write_text(json.dumps({
                "test_image_ids": [2, 3],
                "predicate_classes": ["on"],
                "data": [
                    {"image_id": 1, "relations": [[0, 0, 0]]},
                    {"image_id": 2, "relations": [[0, 0, 0]]},
                    {"image_id": 3, "relations": []},
                ],
            }))
            derived = build_split(source, expected_nonempty=1)
            self.assertEqual(
                [record["image_id"] for record in derived["data"]], [2, 3]
            )
            self.assertEqual(
                derived["_sgg_derivation"]["nonempty_relation_graphs"], 1
            )


if __name__ == "__main__":
    unittest.main()
