import unittest

import torch

from ve_failure_audit import transitions, confidence_interval, cosine, feature_row
from ve_native import SharedContextAdapter


class FailureAuditTests(unittest.TestCase):
    def test_repairs_and_damage_are_not_net_counts(self):
        row = transitions([1, 2, 3, 4], [2, 2, 1, 4], [1, 3, 2, 4])
        self.assertEqual(row["repaired"], 1)
        self.assertEqual(row["damaged"], 1)
        self.assertEqual(row["wrong_to_different_wrong"], 1)
        self.assertEqual(row["label_flips"], 3)
        self.assertEqual(row["baseline_correct"], row["updated_correct"])

    def test_mismatched_labels_fail(self):
        with self.assertRaises(ValueError):
            transitions([1], [1, 2], [1])

    def test_paired_ci_uses_object_denominator(self):
        rows = [dict(updated_correct=1, baseline_correct=0, objects=2),
                dict(updated_correct=2, baseline_correct=0, objects=4)]
        result = confidence_interval(rows)
        self.assertEqual(result["delta"], .5)
        self.assertEqual(result["paired_image_bootstrap_95_ci"], [.5, .5])

    def test_cosine_zero_and_opposed(self):
        self.assertIsNone(cosine(torch.zeros(2), torch.ones(2)))
        self.assertAlmostEqual(cosine(torch.ones(2), -torch.ones(2)), -1., places=6)

    def test_zero_feature_update(self):
        torch.manual_seed(17)
        x = torch.randn(5, 8)
        head = SharedContextAdapter(8, 3)
        classifier = torch.nn.Linear(8, 4)
        native = classifier(x)
        y = torch.tensor([1, 2, 3, 1, 2])
        row, logits = feature_row(x, head, classifier, y, native)
        self.assertTrue(torch.equal(logits, native))
        self.assertEqual(row["label_flips"], 0)
        self.assertEqual(row["relative_update_rms_sum"], 0)


if __name__ == "__main__":
    unittest.main()
