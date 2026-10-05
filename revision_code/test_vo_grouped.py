import unittest

import numpy as np

from vo_math import image_folds, protection_targets, choose_top, macro_delta, policy_summary


class GroupedUtilityTests(unittest.TestCase):
    def test_balanced_image_folds_independent_of_input_order(self):
        ids = [str(i) for i in range(3000)]
        split = image_folds(ids)
        self.assertEqual(split, image_folds(ids[::-1]))
        self.assertEqual(np.bincount(list(split.values())).tolist(), [600]*5)
        for fold in range(5):
            fit = {i for i in ids if split[i] != fold}
            held = {i for i in ids if split[i] == fold}
            self.assertFalse(fit & held)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): image_folds(["1", "1"])

    def test_strict_vs_image_vs_identity(self):
        dc = np.zeros((4, 50)); dc[0, :2] = [-.1, .2]; dc[1, 0] = -.2
        support = np.zeros(50, dtype=bool); support[:2] = True
        labels = protection_targets([1, 1, -1, 1], [.1, -.1, 0, 0], dc, support)
        self.assertEqual(labels["strict"].tolist(), [False, False, False, True])
        self.assertEqual(labels["image_guard"].tolist(), [True, False, False, True])
        self.assertEqual(labels["identity_only"].tolist(), [True, True, False, True])

    def test_unsupported_change_rejected(self):
        dc = np.ones((1, 50))
        with self.assertRaises(ValueError): protection_targets([1], [0], dc, np.zeros(50, bool))

    def test_effect_includes_relation_only_damage(self):
        dc = np.zeros((2, 50)); dc[0, 0] = -.1
        labels = protection_targets([0, 0], [0, .1], dc, np.ones(50, bool))
        self.assertEqual(labels["effect"].tolist(), [True, False])

    def test_label_free_selection_and_tie_order(self):
        picked = choose_top([0, 0, 1], [9, 3, 2], [1., 1., 3.], [True, True, False], 3)
        self.assertEqual(picked.tolist(), [1, -1, -1])

    def test_mr_is_not_image_macro_average(self):
        dc = np.zeros((3, 50)); dc[:, 0] = [.5, .5, 0]; dc[0, 1] = -.5
        support = np.zeros((3, 50)); support[:, 0] = 1; support[0, 1] = 1
        self.assertAlmostEqual(macro_delta(dc, support), (1/3-.5)/50)
        weights = np.array([[1., 1., 1.], [2., 0., 0.]])
        np.testing.assert_allclose(macro_delta(dc, support, weights), [(1/3-.5)/50, 0])

    def test_summary_keeps_unselected_denominators(self):
        summary = policy_summary(np.array([0, -1]), np.array([0]), np.array([1]),
            np.array([0.]), np.zeros((1, 50)), np.ones((2, 50)), [10, 90], [0, 1], draws=20)
        self.assertEqual(summary["positive_objects"], 100)
        self.assertAlmostEqual(summary["delta"]["object"], .01)
        self.assertFalse(summary["formal_gate_accepted"])

    def test_statistics_match_original_gate_function(self):
        from vc_protocol import paired_gate
        rng = np.random.default_rng(19)
        n = 12
        support = rng.random((n, 50)) > .6
        base_class = rng.random((n, 50))
        updated_class = np.clip(base_class+rng.normal(0, .02, (n, 50)), 0, 1)
        dc = np.where(support, updated_class-base_class, 0)
        gain = rng.integers(-1, 3, n).astype(float)
        dr = rng.normal(0, .01, n)
        counts = rng.integers(10, 30, n)
        base, updated = [], []
        for i in range(n):
            a = dict(image_id=str(i), positive_objects=int(counts[i]), post_nms_correct=5,
                recalls=[.3]*6, class_recalls=[[float(x) if ok else None for x, ok in zip(base_class[i], support[i])]]*6)
            b = dict(a, post_nms_correct=int(5+gain[i]), recalls=[.3+dr[i]]*6,
                class_recalls=[[float(x) if ok else None for x, ok in zip(updated_class[i], support[i])]]*6)
            base.append(a); updated.append(b)
        original = paired_gate(base, updated)
        new = policy_summary(np.arange(n), np.arange(n), gain, dr, dc, support, counts, np.arange(n) % 3)
        for key in ["object", "R50", "mR50"]:
            self.assertAlmostEqual(original["delta"][key], new["delta"][key], places=12)
            self.assertAlmostEqual(original["paired_image_bootstrap_one_sided_95_lower"][key],
                                   new["image_bootstrap_one_sided_95_lower"][key], places=12)

    def test_selection_rejects_cross_image_rows(self):
        with self.assertRaises(ValueError):
            policy_summary(np.array([0, -1]), np.array([1]), np.array([1]), np.array([0.]),
                np.zeros((1, 50)), np.ones((2, 50)), [10, 10], [0, 1], draws=20)


if __name__ == "__main__": unittest.main()
