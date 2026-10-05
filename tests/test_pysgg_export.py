import numpy as np
import torch

from scripts.export_pysgg_vg_task import convert_prediction


class _FakePrediction:
    def __init__(self):
        self.size = (100, 100)
        self.bbox = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        self._fields = {
            "pred_labels": torch.tensor([7]),
            "pred_scores": torch.tensor([0.6]),
            "rel_pair_idxs": torch.zeros((0, 2), dtype=torch.long),
            "pred_rel_scores": torch.zeros((0, 51)),
        }

    def convert(self, mode):
        assert mode == "xyxy"
        return self

    def get_field(self, name):
        return self._fields[name]


def test_pysgg_object_confidence_is_not_squared_by_unified_ranking():
    converted = convert_prediction(_FakePrediction())

    np.testing.assert_allclose(converted["pred_boxes"], [[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(converted["pred_entity_scores"][0, 7], 0.6)
    np.testing.assert_allclose(converted["pred_box_scores"], [1.0])

