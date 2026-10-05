import unittest

import torch

from vi_protocol import fold_map, SPEC
from vi_selective import candidate, router_features, informative, fuse, image_rows


class SelectiveFusionTests(unittest.TestCase):
    def router(self, bias):
        return dict(mean=torch.zeros(10), scale=torch.ones(10), weight=torch.zeros(1, 10), bias=torch.tensor([bias]))

    def test_image_folds_are_balanced_stable_and_disjoint(self):
        ids = [str(i) for i in range(5000)]
        m = fold_map(ids)
        self.assertEqual(m, fold_map(ids[::-1]))
        self.assertEqual([list(m.values()).count(i) for i in range(5)], [1000] * 5)
        for fold in range(5):
            hold = {i for i in ids if m[i] == fold}
            fit = {i for i in ids if m[i] != fold}
            self.assertFalse(hold & fit)
            self.assertEqual(len(hold | fit), 5000)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            fold_map(["1", "1"])

    def test_changed_probabilities_not_old_scores(self):
        logits = torch.tensor([[0., 1., .9]])
        expert = torch.tensor([[.01, .99]])
        new, mask, _ = fuse(logits, expert, self.router(10.))
        self.assertTrue(mask.item())
        self.assertEqual(new[:, 1:].argmax(-1).item(), 1)
        self.assertTrue(torch.allclose(new[:, 0], logits.softmax(-1)[:, 0]))
        self.assertTrue(torch.allclose(new.sum(-1), torch.ones(1)))
        self.assertFalse(torch.allclose(new[:, 1:].max(-1).values, logits.softmax(-1)[:, 1:].max(-1).values))

    def test_rejected_update_is_exact_native_probability(self):
        logits = torch.tensor([[0., 1., .9]])
        new, mask, _ = fuse(logits, torch.tensor([[.01, .99]]), self.router(-10.))
        self.assertFalse(mask.item())
        self.assertTrue(torch.equal(new, logits.softmax(-1)))

    def test_agreement_does_not_only_recalibrate(self):
        logits = torch.tensor([[0., 1., .9]])
        new, mask, _ = fuse(logits, torch.tensor([[.99, .01]]), self.router(10.))
        self.assertFalse(mask.item())
        self.assertTrue(torch.equal(new, logits.softmax(-1)))

    def test_training_targets_are_repair_versus_damage(self):
        p = torch.tensor([[.6, .4], [.6, .4], [.6, .4], [.6, .4]])
        changed = torch.tensor([[0., .4, .6], [0., .4, .6], [0., .6, .4], [0., .6, .4]])
        useful, target = informative(p, changed, torch.tensor([2, 1, 2, 1]))
        self.assertEqual(useful.tolist(), [True, True, False, False])
        self.assertEqual(target[useful].tolist(), [1., 0.])

    def test_features_finite_at_probability_extremes(self):
        x = router_features(torch.tensor([[1., 0.], [.5, .5]]), torch.tensor([[0., 1.], [.5, .5]]))
        self.assertEqual(tuple(x.shape), (2, 10))
        self.assertTrue(torch.isfinite(x).all())

    def test_metrics_preserve_image_grouping(self):
        base = torch.zeros(3, 151); base[:, 1] = 2.
        prob = base.softmax(-1)
        rows = image_rows(["a", "b"], ["a", "a", "b"], torch.tensor([1, 2, 1]), base,
                          {"native": prob}, {"native": torch.zeros(3, dtype=torch.bool)})
        self.assertEqual([r["objects"] for r in rows["native"]], [2, 1])
        self.assertEqual(sum(r["correct"] for r in rows["native"]), 2)

    def test_original_acceptance_not_relaxed(self):
        self.assertEqual(SPEC["gate"]["object_gain_min"], .005)
        self.assertEqual(SPEC["gate"]["relation_noninferiority_margin"], .005)
        self.assertEqual(SPEC["requirements"]["correction_precision_min"], .6)
        self.assertEqual(SPEC["probe_epochs"], 40)
        self.assertIn("no automatic", SPEC["stop"])


if __name__ == "__main__":
    unittest.main()
