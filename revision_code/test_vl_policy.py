import copy
import unittest

from vl_policy import candidate_order, improves_identity, preserves_relations


class FeasibilityRules(unittest.TestCase):
    def setUp(self):
        self.base = dict(positive_objects=10, post_nms_correct=5,
                         recalls=[.4], class_recalls=[[.5, None, .3]])

    def test_identity_strict_and_denominator_fixed(self):
        trial = copy.deepcopy(self.base)
        self.assertFalse(improves_identity(self.base, trial))
        trial["post_nms_correct"] += 1
        self.assertTrue(improves_identity(self.base, trial))
        trial["positive_objects"] += 1
        with self.assertRaises(ValueError):
            improves_identity(self.base, trial)

    def test_per_class_protection_not_just_mean(self):
        trial = copy.deepcopy(self.base)
        self.assertTrue(preserves_relations(self.base, trial, 0))
        trial["class_recalls"] = [[.6, None, .2]]
        self.assertFalse(preserves_relations(self.base, trial, 0))

    def test_recall_and_support_protection(self):
        trial = copy.deepcopy(self.base)
        trial["recalls"] = [.39]
        self.assertFalse(preserves_relations(self.base, trial, 0))
        trial["recalls"] = [.4]
        trial["class_recalls"][0][1] = 0.
        with self.assertRaises(ValueError):
            preserves_relations(self.base, trial, 0)

    def test_fixed_candidate_order(self):
        self.assertEqual(candidate_order([3, 2, 0], [.2, .8, .1, .1]), [2, 3, 0])


if __name__ == "__main__":
    unittest.main()
