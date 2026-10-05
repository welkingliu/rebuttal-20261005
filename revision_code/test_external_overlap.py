import unittest

import numpy as np

from external_overlap_audit import correctness, micro_interval, summarize


class ExternalOverlapTest(unittest.TestCase):
    def test_micro_bootstrap_preserves_cluster_weights(self):
        result = micro_interval(np.array([1., 1., 1., 0.]), np.array(["a", "a", "a", "b"]))
        self.assertEqual(result["value"], .75)
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["images"], 2)

    def test_filter_does_not_break_global_endpoint_indices(self):
        payload = dict(object_targets=np.array([1, 0, 1, 0]), relation_targets=np.array([1, 1]),
                       object_image_ids=np.array(["a", "a", "b", "b"]), relation_image_ids=np.array(["a", "b"]),
                       relation_subject=np.array([0, 2]), relation_object=np.array([1, 3]))
        objects = np.array([[0, 1], [1, 0], [0, 1], [0, 1]])
        relations = np.array([[0, 1], [0, 1]])
        values = correctness(payload, objects, relations)
        result = summarize(payload, values, values, {"b"})
        self.assertEqual(result["object_top1"]["value"], .5)
        self.assertEqual(result["triplet_hit_at_1"]["value"], 0.)
        self.assertEqual(result["support"]["relations"], 1)

    def test_reject_cross_image_endpoint(self):
        payload = dict(object_targets=np.array([1, 0]), relation_targets=np.array([1]),
                       object_image_ids=np.array(["a", "b"]), relation_image_ids=np.array(["a"]),
                       relation_subject=np.array([0]), relation_object=np.array([1]))
        with self.assertRaises(ValueError):
            correctness(payload, np.eye(2), np.array([[0, 1]]))


if __name__ == "__main__":
    unittest.main()
