import unittest

import numpy as np
import torch

from modern_live_control import metrics, aggregate


class ModernTests(unittest.TestCase):
    def fixture(self):
        batch = dict(boxes=torch.tensor([[0., 0., .2, .2], [.5, .5, .8, .8]]),
                     entity_labels=torch.tensor([1, 2]), rel_pairs=torch.tensor([[0, 1]]), rel_labels=torch.tensor([1]))
        logits = np.full((2, 151), -100., dtype=np.float32)
        logits[0, 1] = logits[1, 2] = 0.
        relations = np.zeros((1, 51), dtype=np.float32); relations[0, 1] = .9
        pred = dict(pred_boxes=batch["boxes"].numpy().copy(), pred_entity_scores=logits,
                    pred_box_scores=np.ones(2, dtype=np.float32), pred_rel_pairs=np.array([[0, 1]]), pred_rel_scores=relations)
        return batch, pred

    def test_triplet_identity_and_iou(self):
        batch, pred = self.fixture()
        self.assertEqual(metrics(pred, batch)["R"], [1.] * 6)
        pred["pred_boxes"] += .3
        self.assertEqual(metrics(pred, batch)["R"], [0.] * 6)
        batch, pred = self.fixture()
        pred["pred_entity_scores"] = pred["pred_entity_scores"][::-1].copy()
        self.assertEqual(metrics(pred, batch)["R"], [0.] * 6)

    def test_paired_contrast(self):
        batch, pred = self.fixture()
        perfect = metrics(pred, batch)
        pred["pred_boxes"] += .3
        zero = metrics(pred, batch)
        conditions = {"clean": perfect}
        for strength in [.25, .5, 1.]:
            conditions["key_%.2f" % strength] = zero
            conditions["unrelated_%.2f" % strength] = perfect
        result = aggregate([dict(conditions=conditions)] * 3)
        self.assertEqual(result["contrasts"]["1.0"]["key_minus_unrelated_R"], [-1.] * 6)
        self.assertEqual(result["contrasts"]["1.0"]["paired_image_95CI"], [[-1.] * 6] * 2)


if __name__ == "__main__":
    unittest.main()
