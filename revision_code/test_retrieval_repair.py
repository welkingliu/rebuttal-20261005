import unittest

import torch

from retrieval_repair import nearest_posterior, fusion


class RetrievalTests(unittest.TestCase):
    def test_nearest_labels_and_normalization(self):
        keys = torch.eye(4)
        labels = torch.tensor([1, 2, 3, 4])
        q = nearest_posterior(keys[[2, 0]], keys, labels, k=1)
        self.assertTrue(torch.equal(q.argmax(-1), torch.tensor([3, 1])))
        self.assertTrue(torch.allclose(q.sum(-1), torch.ones(2)))
        self.assertTrue(torch.equal(q[:, 0], torch.zeros(2)))

    def test_zero_update_is_exact(self):
        z = torch.tensor([[0., 1., 0.]])
        out, _ = fusion(z, torch.tensor([[0., 0., 1.]]), 0., .9)
        self.assertTrue(torch.equal(out, z))

    def test_high_confidence_identity_untouched(self):
        z = torch.tensor([[0., 10., 0.]])
        out, eligible = fusion(z, torch.tensor([[0., 0., 1.]]), .5, .7)
        self.assertFalse(bool(eligible.any()))
        self.assertTrue(torch.equal(out, z))

    def test_preserves_background_not_maximum_object_score(self):
        z = torch.tensor([[1., 2., 1.7]])
        out, eligible = fusion(z, torch.tensor([[0., 0., 1.]]), .5, .7)
        self.assertTrue(bool(eligible.all()))
        old, new = z.softmax(-1), out.softmax(-1)
        self.assertTrue(torch.allclose(old[:, 0], new[:, 0]))
        self.assertFalse(torch.allclose(old[:, 1:].max(-1)[0], new[:, 1:].max(-1)[0]))
        self.assertEqual(int(new.argmax(-1)), 2)

    def test_agreement_does_not_calibrate_scores(self):
        z = torch.tensor([[1., 2., 1.7]])
        out, eligible = fusion(z, torch.tensor([[0., 1., 0.]]), .5, .7)
        self.assertFalse(bool(eligible.any()))
        self.assertTrue(torch.equal(out, z))


if __name__ == "__main__":
    unittest.main()
