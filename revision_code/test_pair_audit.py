import unittest
import torch

from reproduction_pair_audit import ChunkUnion, concat_chunks, SPEC
from evaluate_reference_pairs import NonZeroShot


class FakeUnion(torch.nn.Module):
    def forward(self, features, proposals, pairs=None):
        return pairs[0].float() * 2, pairs[0].float() + 1


class PairAuditTests(unittest.TestCase):
    def test_eval_chunk_is_noop(self):
        module = FakeUnion().eval()
        pairs = [torch.arange(22).view(11, 2)]
        expected = module([], [None], pairs)
        chunk = ChunkUnion(module, 3)
        actual = module([], [None], pairs)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        chunk.close()
        self.assertTrue(torch.equal(module([], [None], pairs)[0], expected[0]))

    def test_training_chunk_refused(self):
        module = FakeUnion().train()
        chunk = ChunkUnion(module, 3)
        with self.assertRaises(RuntimeError):
            module([], [None], [torch.ones(5, 2)])
        chunk.close()

    def test_only_validation(self):
        self.assertEqual(SPEC["split"], "val")
        self.assertEqual(SPEC["images"], 5000)
        self.assertNotIn("test", SPEC["arms"])

    def test_official_zero_coordinate_order(self):
        seen = NonZeroShot({(10, 20, 3)})
        self.assertNotIn((10, 3, 20), seen)
        self.assertIn((20, 3, 10), seen)
        self.assertIn((10, 4, 20), seen)


if __name__ == "__main__":
    unittest.main()
