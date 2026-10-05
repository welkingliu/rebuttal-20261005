import unittest
import torch
from torch.nn import functional as F
from repair_panel_math import fit_ridge, ridge_logits, make_statistics, prior_logits


class PanelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = F.normalize(torch.randn(12, 768), dim=-1)
        self.n = torch.randn(12, 151); self.y = torch.tensor([1, 2, 3, -1, 0, 1, 2, 3, 1, 2, 0, -1])

    def test_kernel_landmarks_only_positive_and_solve_stable(self):
        state = fit_ridge(self.x, self.n, self.y, True, device="cpu")
        self.assertTrue(bool((self.y[state["landmark_row_indices"]] > 0).all()))
        self.assertLess(state["relative_solve_residual"], 1e-6)
        self.assertTrue(torch.equal(ridge_logits(self.x, self.n, state, 0), self.n))
        out = ridge_logits(self.x, self.n, state, .5).softmax(-1)
        torch.testing.assert_close(out[:, 0], self.n.softmax(-1)[:, 0])

    def test_linear_ridge_has_intercept(self):
        state = fit_ridge(self.x, self.n, self.y, False, device="cpu")
        self.assertEqual(state["coefficients"].shape, (769, 150))
        self.assertIsNone(state["landmarks"])

    def test_directed_training_statistics(self):
        raw = dict(gt_labels=torch.tensor([1, 2]), gt_relations=torch.tensor([[0, 1, 3]]))
        state = make_statistics([raw])
        self.assertEqual(float(state["counts"][3, 0, 1]), 1)
        self.assertEqual(float(state["counts"][3, 1, 0]), 0)
        self.assertEqual(state["fitting_images"], 1)

    def test_prior_inference_requires_no_gt_and_preserves_background(self):
        state = make_statistics([dict(gt_labels=torch.tensor([1, 2]), gt_relations=torch.tensor([[0, 1, 3]]))])
        raw = dict(pairs=torch.tensor([[0, 1]]), relation_logits=torch.randn(1, 51))
        expert = dict(weight=torch.randn(150, 768), bias=torch.zeros(150))
        n = self.n[:2]; x = self.x[:2]
        out = prior_logits(x, n, raw, state, expert, 1, True)
        torch.testing.assert_close(out.softmax(-1)[:, 0], n.softmax(-1)[:, 0])
        self.assertTrue(torch.equal(prior_logits(x, n, {}, {}, {}, 0, True), n))


if __name__ == "__main__": unittest.main()
