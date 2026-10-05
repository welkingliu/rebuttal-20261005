"""Post-registration independent checks of four-scope reporting semantics."""
import unittest
import numpy as np
import torch

from vw_experiment import scopes, scope_summary, calibration_bins


class Prediction:
    def __init__(self):
        self.fields = {"pred_labels": torch.tensor([1, 2, 2, 3]),
                       "pred_scores": torch.tensor([.8, .7, .6, .5])}

    def get_field(self, name):
        return self.fields[name]


class ReportingTests(unittest.TestCase):
    def test_four_distinct_denominators_and_readouts(self):
        logits = torch.zeros(4, 151)
        logits[torch.arange(4), torch.tensor([1, 1, 3, 3])] = 5
        rec = dict(target=np.array([1, 1, 2, 0]), native_endpoint=np.array([True, False, True, False]),
                   unique_proposal=np.array([0, 2]), unique_labels=np.array([1, 2]), gt_objects=3)
        r = scopes(rec, logits, Prediction())
        self.assertEqual(r["pre_nms"], dict(objects=3, correct=2))
        self.assertEqual(r["post_nms"], dict(objects=3, correct=2))
        self.assertEqual(r["unique_pre"], dict(objects=2, correct=1))
        self.assertEqual(r["unique_post"], dict(objects=2, correct=2))
        self.assertEqual(r["native_top50_endpoint"], dict(objects=2, correct=2))
        self.assertEqual(r["not_native_top50_endpoint"], dict(objects=1, correct=0))
        self.assertEqual(sum(b[0] for b in r["fg_conditional_bins"]), 3)
        self.assertEqual(sum(b[0] for b in r["emitted_bins"]), 3)
        totals = scope_summary([r, r])
        self.assertEqual(totals["unique_matched"], 4)
        self.assertEqual(totals["gt_objects"], 6)
        self.assertEqual(totals["post_nms"]["accuracy"], 2/3)

    def test_probability_one_last_bin(self):
        bins = calibration_bins(np.array([0., 1.]), np.array([False, True]))
        self.assertEqual(bins[0], [1., 0., 0.])
        self.assertEqual(bins[14], [1., 1., 1.])

    def test_empty_scope_is_undefined_not_zero(self):
        logits = torch.zeros(4, 151)
        rec = dict(target=np.zeros(4, dtype=int), native_endpoint=np.zeros(4, dtype=bool),
                   unique_proposal=np.array([], dtype=int), unique_labels=np.array([], dtype=int), gt_objects=0)
        r = scope_summary([scopes(rec, logits, Prediction())])
        self.assertIsNone(r["post_nms"]["accuracy"])
        self.assertIsNone(r["unique_post"]["accuracy"])
        self.assertIsNone(r["fg_conditional_ece"])


if __name__ == "__main__":
    unittest.main()
