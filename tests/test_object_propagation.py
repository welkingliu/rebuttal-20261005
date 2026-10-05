import unittest

import torch

from sgg_core.audits.object_propagation import ObjectErrorPropagationAudit


class _Model:
    supported_tasks = ("sgcls",)

    def eval(self):
        return self

    def predict_scene_graph(self, batch, task):
        return {
            "pred_entity_scores": torch.tensor([
                [0.0, 4.0, 0.0],
                [0.0, 3.0, 4.0],
                [0.0, 4.0, 3.0],
            ]),
            "pred_rel_pairs": torch.tensor([[0, 1], [1, 2]]),
            "pred_rel_scores": torch.tensor([
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 5.0],
            ]),
        }


class ObjectPropagationTest(unittest.TestCase):
    def test_stratifies_endpoint_correctness(self):
        batch = {
            "entity_labels": torch.tensor([1, 2, 2]),
            "boxes": torch.tensor([
                [0.0, 0.0, 0.2, 0.2],
                [0.3, 0.3, 0.5, 0.5],
                [0.6, 0.6, 0.8, 0.8],
            ]),
            "rel_pairs": torch.tensor([[0, 1], [1, 2]]),
            "rel_labels": torch.tensor([2, 2]),
        }
        result = ObjectErrorPropagationAudit(
            ks=(1,), bootstrap_trials=10
        ).run({"model": _Model()}, [batch])["model"]["tasks"]["sgcls"]
        self.assertEqual(result["groups"]["both_correct"]["support"], 1)
        self.assertEqual(result["groups"]["one_wrong"]["support"], 1)
        self.assertEqual(result["endpoint_and_pair_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
