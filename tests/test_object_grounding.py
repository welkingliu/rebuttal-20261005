import math
import unittest

import torch

from sgg_core.audits.object_grounding import (
    evaluate_object_logits,
    paired_accuracy_delta,
    relationship_endpoint_summary,
    train_linear_probe,
)


class ObjectGroundingMetricTests(unittest.TestCase):
    def test_good_mask_error_has_clustered_interval_and_collapse_statistic(self):
        logits = torch.tensor([
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 0, 1])
        result = evaluate_object_logits(
            logits,
            labels,
            image_ids=["a", "a", "b", "b"],
            groups={"head": [0], "body": [1], "tail": []},
            mask_iou=torch.tensor([0.9, 0.9, 0.95, 0.95]),
        )

        concentration = result["prediction_concentration"]
        self.assertEqual(concentration["unique_predicted_classes"], 1)
        self.assertEqual(concentration["most_predicted_class"], 0)
        self.assertEqual(concentration["most_predicted_fraction"], 1.0)

        audit = result["wrong_given_good_mask"]["iou>=0.85"]
        self.assertEqual(audit["support"], 4)
        self.assertEqual(audit["error_rate"], 0.5)
        self.assertEqual(audit["bootstrap_95ci"], [0.5, 0.5])
        self.assertEqual(audit["bootstrap_unit"], "image")
        self.assertEqual(audit["bootstrap_estimand"], "object_micro_error_rate")

    def test_empty_high_iou_bin_uses_nan_interval(self):
        result = evaluate_object_logits(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            image_ids=["a"],
            groups={"head": [0], "body": [], "tail": []},
            mask_iou=torch.tensor([0.2]),
        )
        interval = result["wrong_given_good_mask"]["iou>=0.95"]["bootstrap_95ci"]
        self.assertTrue(all(math.isnan(value) for value in interval))

    def test_cluster_bootstrap_targets_object_micro_accuracy(self):
        labels = []
        logits = []
        image_ids = []
        baseline = []
        candidate = []
        for image_index in range(100):
            objects = 100 if image_index < 50 else 1
            is_correct = image_index < 50
            for _ in range(objects):
                labels.append(0)
                logits.append([5.0, 0.0] if is_correct else [0.0, 5.0])
                image_ids.append(str(image_index))
                baseline.append(1)
                candidate.append(0 if is_correct else 1)

        labels = torch.tensor(labels)
        result = evaluate_object_logits(
            torch.tensor(logits),
            labels,
            image_ids=image_ids,
            groups={"head": [0], "body": [], "tail": []},
        )
        point = result["top1_accuracy"]
        low, high = result["bootstrap_95ci"]["top1_accuracy"]
        self.assertEqual(result["bootstrap_unit"], "image")
        self.assertEqual(result["bootstrap_estimand"], "object_micro_accuracy")
        self.assertGreater(point, 0.98)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)
        self.assertGreater(low, 0.95)

        paired = paired_accuracy_delta(
            torch.tensor(baseline), torch.tensor(candidate), labels, image_ids
        )
        low, high = paired["bootstrap_95ci"]
        self.assertEqual(paired["bootstrap_unit"], "image")
        self.assertGreater(paired["delta_top1"], 0.98)
        self.assertLessEqual(low, paired["delta_top1"])
        self.assertGreaterEqual(high, paired["delta_top1"])
        self.assertGreater(low, 0.95)

    def test_linear_probe_stops_after_validation_plateau(self):
        features = torch.zeros(12, 4)
        labels = torch.tensor([0, 1] * 6)
        train_mask = torch.tensor([True] * 8 + [False] * 4)
        validation_mask = ~train_mask
        _, history = train_linear_probe(
            features, labels, train_mask, validation_mask,
            num_classes=2, seed=17, device="cpu", epochs=50,
            learning_rate=0.0, early_stopping_patience=3,
        )
        self.assertEqual(len(history), 4)
        self.assertTrue(history[0]["improved"])
        self.assertFalse(history[-1]["improved"])

    def test_relation_endpoint_failure_conditions_on_both_masks(self):
        result = relationship_endpoint_summary(
            prediction=torch.tensor([0, 0, 1, 0]),
            labels=torch.tensor([0, 1, 1, 0]),
            graph_records=[{
                "image_id": "a",
                "object_start": 0,
                "object_stop": 4,
                "rel_pairs": torch.tensor([[0, 1], [2, 3]]),
            }],
            mask_iou=torch.tensor([0.90, 0.90, 0.95, 0.80]),
        )
        self.assertEqual(result["endpoint_failure_rate"], 0.5)
        conditioned = result["conditioned_on_both_endpoint_mask_iou"]
        at_85 = conditioned["both_iou>=0.85"]
        self.assertEqual(at_85["support"], 1)
        self.assertEqual(at_85["coverage"], 0.5)
        self.assertEqual(at_85["endpoint_failure_rate"], 1.0)
        self.assertEqual(at_85["bootstrap_estimand"],
                         "relation_micro_endpoint_failure_rate")


if __name__ == "__main__":
    unittest.main()
