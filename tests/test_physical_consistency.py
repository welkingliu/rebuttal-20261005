import unittest

from sgg_core.audits.physical_consistency import predicate_violation, summarise_pvr


class PhysicalConsistencyTest(unittest.TestCase):
    def test_only_two_dimensional_identifiable_predicates_are_checked(self):
        left = [0.0, 0.0, 0.2, 0.2]
        right = [0.8, 0.0, 1.0, 0.2]
        self.assertFalse(predicate_violation("left of", left, right))
        self.assertTrue(predicate_violation("right of", left, right))
        self.assertIsNone(predicate_violation("behind", left, right))

    def test_zero_and_undefined_are_distinct(self):
        undefined = summarise_pvr([], 20, min_checked=2)
        self.assertIsNone(undefined["PVR"])
        self.assertEqual(undefined["pvr_status"], "undefined")
        zero = summarise_pvr(
            [{"checked": 2, "violations": 0}], 10, min_checked=2
        )
        self.assertEqual(zero["PVR"], 0.0)
        self.assertEqual(zero["pvr_status"], "ok")


if __name__ == "__main__":
    unittest.main()
