import json
import tempfile
import unittest
from pathlib import Path

import torch

from sgg_core.data.shared_vg_ontology import (
    build_exact_mapping,
    load_vg_ontology,
    project_batch_to_vg,
)


class _Dataset:
    sgg_dict = {
        "idx_to_label": {"1": "cat", "2": "traffic-light", "3": "unknown"},
        "idx_to_predicate": {"1": "on", "2": "next to"},
    }


class SharedVGOntologyTests(unittest.TestCase):
    def test_exact_mapping_and_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            dictionary = Path(directory) / "vg.json"
            dictionary.write_text(json.dumps({
                "label_to_idx": {"cat": 1, "traffic light": 2},
                "predicate_to_idx": {"on": 1},
            }))
            ontology = load_vg_ontology(dictionary)
        mapping = build_exact_mapping(_Dataset(), ontology)
        self.assertEqual(mapping["object_map"], {1: 1, 2: 2})
        self.assertEqual(mapping["predicate_map"], {1: 1})
        batch = {
            "boxes": torch.tensor([
                [0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0],
                [0.2, 0.2, 0.3, 0.3],
            ]),
            "entity_labels": torch.tensor([1, 2, 3]),
            "rel_pairs": torch.tensor([[0, 1], [0, 2], [1, 0]]),
            "rel_labels": torch.tensor([1, 1, 2]),
            "image": torch.zeros(3, 8, 8),
            "image_id": "one",
            "dataset": "synthetic",
            "ontology_id": "synthetic:1",
        }
        projected, report = project_batch_to_vg(batch, mapping, ontology)
        self.assertIsNotNone(projected)
        self.assertEqual(projected["entity_labels"].tolist(), [1, 2])
        self.assertEqual(projected["rel_pairs"].tolist(), [[0, 1]])
        self.assertEqual(projected["rel_labels"].tolist(), [1])
        self.assertEqual(report["retained_relations"], 1)
        self.assertEqual(report["skipped_relations"], {
            "unmapped_endpoint": 1, "unmapped_predicate": 1,
        })
