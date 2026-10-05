import unittest
from types import SimpleNamespace

import numpy as np

from scripts.export_openpsg_predictions import convert_result


class OpenPSGExportTest(unittest.TestCase):
    @staticmethod
    def _result(refine_dists=None, rel_pairs=None):
        rel_scores = np.zeros((1, 57), dtype=np.float32)
        rel_scores[0, 7] = 1.0
        return SimpleNamespace(
            refine_bboxes=np.asarray([
                [0.0, 0.0, 100.0, 50.0, 0.8],
                [20.0, 10.0, 80.0, 40.0, 0.6],
            ], dtype=np.float32),
            refine_labels=None,
            labels=np.asarray([1, 133], dtype=np.int64),
            refine_dists=refine_dists,
            rel_pair_idxes=np.asarray(
                rel_pairs if rel_pairs is not None else [[0, 1]],
                dtype=np.int64,
            ),
            rel_dists=rel_scores,
        )

    def test_end_to_end_hard_labels_remain_box_aligned(self):
        output = convert_result(self._result(), height=50, width=100)
        np.testing.assert_allclose(
            output["pred_boxes"][0], [0.0, 0.0, 1.0, 1.0]
        )
        self.assertEqual(output["pred_entity_scores"].shape, (2, 134))
        self.assertEqual(output["pred_rel_scores"].shape, (1, 57))
        self.assertEqual(output["pred_entity_scores"][0].argmax(), 1)
        self.assertEqual(output["pred_entity_scores"][1].argmax(), 133)

    def test_classic_model_preserves_official_object_distribution(self):
        distributions = np.zeros((2, 134), dtype=np.float32)
        distributions[0, 1] = 0.75
        distributions[1, 133] = 0.65
        output = convert_result(
            self._result(refine_dists=distributions), height=50, width=100
        )
        np.testing.assert_array_equal(
            output["pred_entity_scores"], distributions
        )

    def test_relation_pair_must_reference_exported_entity(self):
        with self.assertRaisesRegex(ValueError, "missing object"):
            convert_result(
                self._result(rel_pairs=[[0, 2]]), height=50, width=100
            )


if __name__ == "__main__":
    unittest.main()
