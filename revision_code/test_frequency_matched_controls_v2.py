import unittest
import numpy as np
from frequency_matched_controls_v2 import replacements


class Controls(unittest.TestCase):
    def test_all_foreground_targets(self):
        counts = np.arange(151)[::-1]
        truth = np.arange(1, 151)
        labels = np.roll(truth, 1)
        bins = np.argsort(np.argsort(-counts[1:], kind='stable'), kind='stable') // 30
        for seed in (17, 23, 31):
            out = replacements(labels, truth, counts, seed, 'rare')
            self.assertTrue(np.all(out != labels) and np.all(out != truth) and np.all(out > 0))
            np.testing.assert_array_equal(bins[out-1], bins[truth-1])
            np.testing.assert_array_equal(out, replacements(labels, truth, counts, seed, 'rare'))

    def test_unchanged_nodes(self):
        truth = np.array([1, 150, 1])
        labels = np.array([1, 2, 1])
        out = replacements(labels, truth, np.zeros(151), 17, 'ties')
        np.testing.assert_array_equal(out != labels, truth != labels)


if __name__ == '__main__':
    unittest.main()
