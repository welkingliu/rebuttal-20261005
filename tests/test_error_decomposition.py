import unittest

import torch

from sgg_core.audits.error_decomposition import GroundingErrorDecompositionAudit


class FakeModel:
    supports_standard_sgg = True
    supported_tasks = ("sgdet",)

    def eval(self):
        return self

    def predict_scene_graph(self, batch, task):
        scores = torch.full((2, 4), -5.0)
        scores[0, 1] = 5.0
        scores[1, 3] = 5.0  # localized but wrong; GT class is 2
        return {
            "pred_boxes": batch["boxes"].clone(),
            "pred_entity_scores": scores,
        }


class ErrorDecompositionTest(unittest.TestCase):
    def test_localization_and_recognition_are_separate(self):
        loader = [{
            "boxes": torch.tensor([
                [0.0, 0.0, 0.3, 0.3], [0.6, 0.6, 1.0, 1.0]
            ]),
            "entity_labels": torch.tensor([1, 2]),
        }]
        result = GroundingErrorDecompositionAudit().run(
            {"fake": FakeModel()}, loader
        )["fake"]
        self.assertEqual(result["metrics"]["localization_recall"], 1.0)
        self.assertEqual(
            result["metrics"]["recognition_accuracy_given_localized"], 0.5
        )
        self.assertEqual(result["metrics"]["grounded_object_recall"], 0.5)

    def test_object_identity_uses_foreground_scores_like_standard_sgg(self):
        class BackgroundHeavyModel(FakeModel):
            def predict_scene_graph(self, batch, task):
                scores = torch.tensor([[10.0, 9.0, -5.0]])
                return {
                    "pred_boxes": batch["boxes"].clone(),
                    "pred_entity_scores": scores,
                }

        loader = [{
            "boxes": torch.tensor([[0.0, 0.0, 0.3, 0.3]]),
            "entity_labels": torch.tensor([1]),
        }]
        result = GroundingErrorDecompositionAudit().run(
            {"fake": BackgroundHeavyModel()}, loader
        )["fake"]
        self.assertEqual(
            result["metrics"]["recognition_accuracy_given_localized"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
