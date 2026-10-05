import unittest

import torch
from torch.nn import functional as F

from vo_math import image_folds
from vp_math import split_fold, make_head, weighted_nll, fusion_logits, fit_head


class ProposalExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.old = dict(weight=torch.randn(150, 768)*.01, bias=torch.randn(150)*.01)

    def test_nested_image_isolation(self):
        ids = [str(i) for i in range(3000)]; mapping = image_folds(ids)
        all_held = []
        for fold in range(5):
            split = split_fold(ids, mapping, fold)
            self.assertEqual([len(split[k]) for k in ["training", "fit", "inner_validation", "held"]], [2400, 2160, 240, 600])
            self.assertFalse(set(split["fit"]) & set(split["inner_validation"]))
            self.assertFalse(set(split["training"]) & set(split["held"]))
            self.assertEqual(set(split["training"]), set(split["fit"]+split["inner_validation"]))
            all_held.extend(split["held"])
        self.assertEqual(len(set(all_held)), 3000)

    def test_foreground_initialization_matches_old(self):
        x = torch.randn(9, 768); h = make_head(self.old)
        expected = F.linear(x, self.old["weight"], self.old["bias"]).softmax(-1)
        torch.testing.assert_close(h(x)[:, 1:].softmax(-1), expected)
        self.assertTrue(torch.equal(h.weight[0], torch.zeros(768)))

    def test_ambiguous_has_no_gradient(self):
        logits = torch.randn(3, 151, requires_grad=True)
        y = torch.tensor([1, -1, 0]); loss, num, den = weighted_nll(logits, y)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(151)))
        self.assertAlmostEqual(float(den), 1.25)
        self.assertTrue(float(logits.grad[0].abs().sum()) > 0)

    def test_preserved_background_and_distribution(self):
        b = torch.randn(8, 151); q = torch.randn(8, 151).softmax(-1)
        p = fusion_logits(b, q).softmax(-1)
        torch.testing.assert_close(p[:, 0], b.softmax(-1)[:, 0])
        torch.testing.assert_close(p.sum(-1), torch.ones(8))
        self.assertTrue(torch.equal(fusion_logits(b, q, alpha=0), b))

    def test_old_fusion_matches_original_helper(self):
        from vj_mac import probabilities
        b = torch.randn(8, 151); q = torch.randn(8, 150).softmax(-1)
        torch.testing.assert_close(fusion_logits(b, q).softmax(-1), probabilities(b, q)[2])

    def test_full_fusion_uses_actual_background(self):
        b = torch.randn(8, 151); q = torch.randn(8, 151).softmax(-1)
        torch.testing.assert_close(fusion_logits(b, q, False).softmax(-1), .5*b.softmax(-1)+.5*q)
        with self.assertRaises(ValueError): fusion_logits(b, q[:, 1:], False)

    def test_real_gradient_and_fixed_refit(self):
        x = F.normalize(torch.randn(32, 768), dim=-1); y = torch.arange(32) % 151
        trained, record = fit_head(x, y, self.old, epochs=2, device="cpu")
        self.assertEqual(record["selected_epochs"], 2)
        self.assertFalse(torch.equal(trained["weight"][1:], self.old["weight"]))


if __name__ == "__main__": unittest.main()
