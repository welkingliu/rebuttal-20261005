import unittest

import torch

from independent_identity_expert import combine, summary, SPEC


class IndependentExpertTests(unittest.TestCase):
    def test_uncertain_disagreement_uses_actual_new_scores(self):
        baseline = torch.tensor([[0., 1., .9]])
        expert = torch.tensor([[.01, .99]])
        changed, mask = combine(baseline, expert, .5)
        self.assertTrue(bool(mask.all()))
        self.assertEqual(int(changed.argmax(-1)), 2)
        self.assertTrue(torch.allclose(changed[:, 0], baseline.softmax(-1)[:, 0]))
        self.assertFalse(torch.allclose(changed[:, 1:].max(-1)[0], baseline.softmax(-1)[:, 1:].max(-1)[0]))

    def test_confident_native_and_uncertain_expert_are_untouched(self):
        baseline = torch.tensor([[0., 9., 0.], [0., 1., .9]])
        expert = torch.tensor([[.01, .99], [.51, .49]])
        changed, mask = combine(baseline, expert, .5)
        self.assertFalse(bool(mask.any()))
        self.assertTrue(torch.equal(changed, baseline.softmax(-1)))

    def test_same_label_is_not_just_recalibrated(self):
        baseline = torch.tensor([[0., 1., .9]])
        changed, mask = combine(baseline, torch.tensor([[.99, .01]]), .25)
        self.assertFalse(bool(mask.any()))
        self.assertTrue(torch.equal(changed, baseline.softmax(-1)))

    def test_finite_development_only_design(self):
        self.assertEqual(len(SPEC["candidates"]), 2)
        self.assertEqual(SPEC["requirements"]["identity_gain_min"], .005)
        self.assertIn("no automatic", SPEC["stop"])
        self.assertIn("gate or final test", SPEC["stop"])


if __name__ == "__main__":
    unittest.main()
