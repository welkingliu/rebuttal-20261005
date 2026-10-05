import unittest
from types import SimpleNamespace

import torch

from sgg_core.audits.live_sgcls_validation import LiveSGClsValidationAudit
from sgg_core.mitigation.grounding_regularizer import GroundingDependencyRegularizer
from sgg_core.mitigation.run_mitigation import (
    _acceptance,
    _deduplicate_training_relations,
    _finite,
    _json_default,
    _parameter_snapshot,
    _parameter_update_audit,
)


class MitigationObjectiveTest(unittest.TestCase):
    def test_live_validation_does_not_require_prediction_cache(self):
        class Model:
            name = "live-test"

            def eval(self):
                return self

            def forward_grounding(self, batch):
                entity_scores = torch.full((2, 4), -4.0)
                entity_scores[0, 1] = 4.0
                entity_scores[1, 2] = 4.0
                relation_scores = torch.full((1, 3), -4.0)
                relation_scores[0, 1] = 4.0
                return {
                    "pred_entity_scores": entity_scores,
                    "pred_rel_scores": relation_scores,
                    "pred_rel_pairs": batch["rel_pairs"],
                }

        batch = {
            "boxes": torch.tensor([[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0]]),
            "entity_labels": torch.tensor([1, 2]),
            "rel_pairs": torch.tensor([[0, 1]]),
            "rel_labels": torch.tensor([1]),
        }
        result = LiveSGClsValidationAudit(
            ks=(1, 5, 50), device="cpu", seen_triplets=set()
        ).run(Model(), [batch])
        identity = result["grounding_error_decomposition"]["object_identity"]
        self.assertEqual(identity["top1_accuracy_given_localized"], 1.0)
        self.assertTrue(_finite(identity["ece_15"]))
        self.assertTrue(_finite(
            result["standard_sgg"]["tasks"]["sgcls"]["metrics"]["mR@50"]
        ))

    def test_acceptance_requires_live_protocol_and_object_update(self):
        def payload(top1, ece, mr):
            return {
                "grounding_error_decomposition": {
                    "status": "ok",
                    "errors": [],
                    "object_identity": {
                        "localized_objects": 100,
                        "top1_accuracy_given_localized": top1,
                        "ece_15": ece,
                    },
                },
                "standard_sgg": {
                    "tasks": {"sgcls": {"metrics": {"mR@50": mr}}}
                },
            }

        args = SimpleNamespace(
            minimum_validation_objects=50,
            max_mr_drop=0.01,
            minimum_object_top1_gain=0.005,
            max_object_ece_increase=0.01,
        )
        acceptance = _acceptance(
            payload(0.50, 0.10, 0.20),
            payload(0.51, 0.10, 0.20),
            args,
            parameter_audit={"object_parameters_updated": True},
        )
        self.assertTrue(acceptance["protocol_valid"])
        self.assertTrue(acceptance["passed"])

    def test_parameter_audit_detects_object_head_update(self):
        object_parameter = torch.nn.Parameter(torch.zeros(2))
        relation_parameter = torch.nn.Parameter(torch.zeros(2))
        groups = {
            "object": [object_parameter],
            "relation": [relation_parameter],
        }
        initial = _parameter_snapshot(groups)
        with torch.no_grad():
            object_parameter.add_(1.0)
        audit = _parameter_update_audit(groups, initial)
        self.assertTrue(audit["object_parameters_updated"])
        self.assertFalse(audit["groups"]["relation"]["updated"])

    def test_numpy_scalars_are_json_safe(self):
        import json
        import numpy as np

        self.assertIs(type(_finite(np.float64(1.0))), bool)
        payload = {
            "applicable": np.bool_(True),
            "score": np.float64(0.5),
            "count": np.int64(3),
        }
        encoded = json.dumps(payload, default=_json_default)
        self.assertEqual(
            json.loads(encoded),
            {"applicable": True, "score": 0.5, "count": 3},
        )

    def test_training_relations_follow_one_predicate_per_pair_protocol(self):
        batch = {
            "rel_pairs": torch.tensor([[0, 1], [0, 1], [2, 1], [0, 1]]),
            "rel_labels": torch.tensor([3, 4, 5, 6]),
            "union_features": torch.arange(8).reshape(4, 2),
        }
        first, removed = _deduplicate_training_relations(batch, seed=17)
        second, second_removed = _deduplicate_training_relations(batch, seed=17)

        self.assertEqual(removed, 2)
        self.assertEqual(second_removed, 2)
        self.assertEqual(first["rel_pairs"].tolist(), [[0, 1], [2, 1]])
        self.assertTrue(torch.equal(first["rel_labels"], second["rel_labels"]))
        self.assertTrue(
            torch.equal(first["union_features"], second["union_features"])
        )
        self.assertEqual(first["union_features"].shape, (2, 2))

    def test_objective_is_finite_and_backpropagates(self):
        clean = torch.tensor(
            [[-2.0, 3.0, 0.0], [-2.0, 0.0, 3.0]], requires_grad=True
        )
        mild = (clean.detach() + 0.05).requires_grad_(True)
        ablated = torch.tensor(
            [[-2.0, 4.0, 0.0], [-2.0, 0.0, 4.0]], requires_grad=True
        )
        targets = torch.tensor([1, 2])
        result = GroundingDependencyRegularizer()(clean, mild, ablated, targets)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(float(result["ablated_uncertainty"]), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(clean.grad)
        self.assertIsNotNone(mild.grad)
        self.assertIsNotNone(ablated.grad)

    def test_mild_consistency_does_not_push_clean_target(self):
        def gradients(mild_weight):
            clean = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
            mild = torch.tensor([[0.0, -1.0, 2.0]], requires_grad=True)
            ablated = clean.detach().clone().requires_grad_(True)
            objective = GroundingDependencyRegularizer(
                mild_weight=mild_weight,
                dependency_weight=0.0,
                uncertainty_weight=0.0,
                object_weight=0.0,
                object_consistency_weight=0.0,
                object_calibration_weight=0.0,
            )
            objective(clean, mild, ablated, torch.tensor([1]))["loss"].backward()
            return clean.grad.clone(), None if mild.grad is None else mild.grad.clone()

        clean_without, mild_without = gradients(0.0)
        clean_with, mild_with = gradients(1.0)
        self.assertTrue(torch.allclose(clean_without, clean_with))
        self.assertTrue(
            mild_without is None or float(mild_without.abs().sum()) == 0.0
        )
        self.assertIsNotNone(mild_with)
        self.assertGreater(float(mild_with.abs().sum()), 0.0)

    def test_object_terms_are_supervised_and_differentiable(self):
        relation = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
        objects = torch.tensor(
            [[-2.0, 4.0, 0.0], [-2.0, 0.0, 4.0]], requires_grad=True
        )
        mask_objects = (objects.detach() + 0.1).requires_grad_(True)
        result = GroundingDependencyRegularizer()(
            relation,
            relation + 0.01,
            torch.zeros_like(relation),
            torch.tensor([1]),
            object_logits=objects,
            object_targets=torch.tensor([1, 2]),
            mask_object_logits=mask_objects,
        )
        self.assertEqual(result["num_objects"], 2)
        self.assertTrue(torch.isfinite(result["object_supervised"]))
        result["loss"].backward()
        self.assertIsNotNone(objects.grad)
        self.assertIsNotNone(mask_objects.grad)


if __name__ == "__main__":
    unittest.main()
