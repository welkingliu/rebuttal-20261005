import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgg_core.experiments.experiment_1 import (
    ObjectPairReasoner, _all_ordered_pairs, _graph_adjacency, materialize,
)
from sgg_core.data.data_utils import _collate_fn


class ObjectNodeProtocolTest(unittest.TestCase):
    def test_nodes_are_objects_and_pairs_are_directed(self):
        pairs = _all_ordered_pairs(3)
        self.assertEqual(pairs.shape, (6, 2))
        self.assertIn([0, 1], pairs.tolist())
        self.assertIn([1, 0], pairs.tolist())
        self.assertNotIn([0, 0], pairs.tolist())

    def test_pair_head_predicts_one_distribution_per_pair(self):
        model = ObjectPairReasoner(16, 32, 7, depth=2, mode="gcn")
        model.eval()
        object_features = torch.randn(3, 16)
        pairs = _all_ordered_pairs(3)
        union_features = torch.randn(pairs.size(0), 16)
        boxes = torch.tensor([
            [0.0, 0.0, 0.2, 0.2],
            [0.3, 0.0, 0.5, 0.2],
            [0.0, 0.3, 0.2, 0.5],
        ])
        adjacency = _graph_adjacency(3, torch.tensor([[0, 1]]))
        logits, probes = model(
            object_features, union_features, boxes, adjacency, pairs
        )
        self.assertEqual(logits.shape, (6, 7))
        self.assertEqual(len(probes), 4)

        node, split_probes = model.encode_nodes(object_features, adjacency)
        split_logits = model.score_pairs(node, union_features, boxes, pairs)
        self.assertTrue(torch.allclose(logits, split_logits))
        self.assertEqual(len(split_probes), 4)

    def test_collate_rejects_silent_multi_image_batch(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            _collate_fn([{"image_id": 1}, {"image_id": 2}])

    def test_raw_materialization_fails_with_image_context(self):
        batch = {
            "boxes": torch.tensor([
                [0.0, 0.0, 0.2, 0.2],
                [0.3, 0.0, 0.5, 0.2],
            ]),
            "entity_labels": torch.tensor([1, 2]),
            "rel_pairs": torch.tensor([[0, 1]]),
            "rel_labels": torch.tensor([1]),
            "image_id": 42,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "image_id=42"):
                materialize(
                    [batch], object(), Path(tmp) / "cache.pt", allow_proxy=False
                )


if __name__ == "__main__":
    unittest.main()
