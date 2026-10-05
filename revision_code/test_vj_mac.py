import copy
import unittest

import torch

from vj_mac import SPEC, prototypes, visual_features, probabilities, confidence_features, route, group_folds, records, aggregate, screen_checks


class VisualVerifierTests(unittest.TestCase):
    def test_prototypes_exclude_whole_fold(self):
        x = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
        y = torch.tensor([1, 1, 2, 2]); folds = torch.tensor([0, 1, 0, 1])
        c, n = prototypes(x[folds != 0], y[folds != 0])
        self.assertTrue(torch.equal(c[0], torch.tensor([0., 1.])))
        self.assertEqual(n[0], 1)
        altered = x.clone(); altered[folds == 0] *= -1
        d, _ = prototypes(altered[folds != 0], y[folds != 0])
        self.assertTrue(torch.equal(c, d))

    def test_visual_evidence_depends_on_class_meaning_not_only_confidence(self):
        p = torch.tensor([[.8, .2], [.8, .2]])
        cand = torch.tensor([[0., .2, .8], [0., .2, .8]])
        x = torch.tensor([[1., 0.], [0., 1.]])
        v, eligible = visual_features(x, p, cand, torch.eye(2), torch.tensor([8, 8]))
        self.assertTrue(bool(eligible.all()))
        self.assertLess(v[0, 2], 0); self.assertGreater(v[1, 2], 0)

    def test_no_selection_is_exact_noop_and_missing_support_abstains(self):
        b = torch.tensor([[-1., 2., 0.]])
        q = torch.tensor([[.001, .999]])
        native, p, _ = probabilities(b, q)
        feats = confidence_features(p, q)
        r = dict(mean=torch.zeros(10), scale=torch.ones(10), weight=torch.zeros((1, 10)), bias=torch.tensor([10.]))
        out, mask, _ = route(b, q, feats, r, torch.tensor([False]))
        self.assertFalse(mask.item()); self.assertTrue(torch.equal(out, native))
        out, mask, _ = route(b, q, feats, r, torch.tensor([True]))
        self.assertTrue(mask.item()); self.assertEqual(out[0, 0], native[0, 0])
        self.assertAlmostEqual(float(out.sum()), 1., places=6)

    def test_group_isolation(self):
        data = dict(image_ids=["a", "b"], offsets=[0, 2, 3], labels=torch.ones(3))
        self.assertEqual(group_folds(data, dict(a=0, b=1)).tolist(), [0, 0, 1])
        broken = copy.deepcopy(data); broken["image_ids"][1] = "a"
        with self.assertRaises(ValueError):
            group_folds(broken, dict(a=0))

    def test_denominators_and_undefined_precision(self):
        data = dict(image_ids=["a", "b"], offsets=[0, 2, 3], labels=torch.tensor([1, 2, 1]),
                    baseline=torch.tensor([[0., 2., 1.], [0., 1., 2.], [0., 1., 2.]]))
        prob = data["baseline"].softmax(-1)
        result = aggregate(records(data, prob, torch.zeros(3, dtype=torch.bool)))
        self.assertAlmostEqual(result["top1"], 2 / 3)
        self.assertIsNone(result["repair_precision"])
        self.assertEqual(result["descriptive_paired_image_ci95"], [0., 0.])

    def test_gate_cannot_be_relaxed(self):
        native = dict(macro_accuracy=.7)
        low = dict(delta=.00499, repair_precision=.9, macro_accuracy=.8)
        self.assertFalse(all(screen_checks(low, native).values()))
        self.assertEqual(SPEC["gate"]["object_gain_min"], .005)
        self.assertEqual(SPEC["gate"]["relation_noninferiority_margin"], .005)
        self.assertTrue(SPEC["gate"]["requires_both_tasks"])


if __name__ == "__main__":
    unittest.main()
