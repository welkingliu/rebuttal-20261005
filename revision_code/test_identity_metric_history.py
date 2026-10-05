import unittest

import numpy as np

from audit_identity_metric_history import ece, foreground_confidence, paired_counts


class HistoryAuditTests(unittest.TestCase):
    def test_units(self):
        b = [dict(image_id="a", n=5956, c=4411)]
        c = [dict(image_id="a", n=5956, c=4434)]
        r = paired_counts(b, c, "n", "c", 100)
        self.assertEqual(r["net_correct"], 23)
        self.assertAlmostEqual(r["delta_percentage_points"], .38616521155)
        self.assertAlmostEqual(r["relative_improvement_percent"], 100 * 23 / 4411)

    def test_denominator_guard(self):
        with self.assertRaises(ValueError):
            paired_counts([dict(image_id="a", n=10, c=3)], [dict(image_id="a", n=9, c=4)], "n", "c")

    def test_duplicate_guard(self):
        rows = [dict(image_id="a", n=10, c=3)] * 2
        with self.assertRaises(ValueError):
            paired_counts(rows, rows, "n", "c")

    def test_empty_denominator(self):
        rows = [dict(image_id="a", n=0, c=0)]
        with self.assertRaises(ValueError):
            paired_counts(rows, rows, "n", "c")

    def test_ratio_of_counts_not_mean_of_image_accuracies(self):
        b = [dict(image_id="a", n=1, c=0), dict(image_id="b", n=99, c=0)]
        c = [dict(image_id="a", n=1, c=1), dict(image_id="b", n=99, c=0)]
        self.assertEqual(paired_counts(b, c, "n", "c", 100)["delta_fraction"], .01)

    def test_background_shift_not_foreground_identity(self):
        logits = np.array([[2., 3., 1.], [2., 1., 3.]])
        changed = logits.copy()
        changed[:, 0] += 3
        full, conditional = foreground_confidence(logits)
        full2, conditional2 = foreground_confidence(changed)
        np.testing.assert_allclose(conditional, conditional2)
        self.assertTrue((full2 < full).all())
        self.assertAlmostEqual(ece(conditional, [1, 1]), ece(conditional2, [1, 1]))
        self.assertNotEqual(ece(full, [1, 1]), ece(full2, [1, 1]))

    def test_noop_paired_interval(self):
        rows = [dict(image_id=str(i), n=i+1, c=i) for i in range(10)]
        r = paired_counts(rows, list(reversed(rows)), "n", "c", 100)
        self.assertEqual(r["paired_image_95ci_pp"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
