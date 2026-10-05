import unittest

import numpy as np

from evidence_completion import paired_ratio, selected_ids, transitions


class EvidenceTests(unittest.TestCase):
    def test_zero_change(self):
        result = paired_ratio([1, 2], [1, 2], [3, 4], draws=20)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["paired_image_95ci"], [0, 0])

    def test_image_ratio_not_mean_of_ratios(self):
        result = paired_ratio([1, 2], [0, 0], [1, 100], draws=20)
        self.assertAlmostEqual(result["value"], 3/101)

    def test_transition_counts_not_net_difference(self):
        r = transitions([2, 1, 3, 1], [1, 2, 1, 2], [2, 1, 3, 1], [1, 2, 3, 2], [1, 1, 1, 0])
        self.assertEqual(r["positive_proposals"], 3)
        self.assertEqual(r["pre_corrections"], 2)
        self.assertEqual(r["post_corrections"], 1)
        self.assertEqual(r["post_regressions"], 1)
        self.assertEqual(sum(r["correctness_pattern_counts"]), 3)

    def test_hash_selection_independent_of_input_order(self):
        a = ["1", "4", "9", "3"]
        self.assertEqual(selected_ids(a, 3, "test:"), selected_ids(a[::-1], 3, "test:"))

    def test_empty_denominator_rejected(self):
        with self.assertRaises(ValueError): paired_ratio([0], [0], [0])


if __name__ == "__main__": unittest.main()
