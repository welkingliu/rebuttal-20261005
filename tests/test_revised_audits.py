import math
import unittest

import torch

from sgg_core.audits.graph_audit import (
    GraphLevelAudit, _clustered_rate_ci, _matched_control_node,
    ablate_terminal_node,
)
from sgg_core.audits.pair_audit import (
    PairLevelAudit, VisualPerturbation, derive_batch_seed, recall_at_k,
)
from sgg_core.audits.perturbation_sweep import PerturbationSweepAudit


def batch():
    return {
        "image_id": "one",
        "visual_features": torch.randn(3, 8),
        "union_features": torch.randn(2, 8),
        "boxes": torch.tensor([
            [0.0, 0.0, 0.2, 0.2],
            [0.3, 0.0, 0.5, 0.2],
            [0.6, 0.0, 0.8, 0.2],
        ]),
        "entity_labels": torch.tensor([1, 2, 3]),
        "rel_pairs": torch.tensor([[0, 1], [0, 2]]),
        "rel_labels": torch.tensor([1, 2]),
        "graph_adj": torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    }


class StableModel:
    def eval(self):
        return self

    def predict(self, _batch):
        # First relation is clean-correct; second is clean-wrong. Both persist.
        return {"pred_rel_scores": torch.tensor([
            [-5.0, 5.0, -5.0],
            [-5.0, 5.0, -5.0],
        ])}


class RevisedAuditTest(unittest.TestCase):
    def test_clustered_rate_ci_treats_missing_counter_keys_as_zero(self):
        interval = _clustered_rate_ci(
            [{"total": 2}, {"total": 2, "invariant": 1}],
            numerator="invariant",
            denominator="total",
            trials=20,
        )
        self.assertEqual(len(interval), 2)
        self.assertTrue(all(math.isfinite(value) for value in interval))

    def test_pair_hit_k_ranks_only_foreground_predicates(self):
        scores = torch.tensor([[100.0, 3.0, 2.0]])
        self.assertEqual(recall_at_k(scores, torch.tensor([1]), k=1), 1.0)

    def test_stochastic_perturbations_are_reproducible_but_image_specific(self):
        perturbation = VisualPerturbation()
        sample = batch()
        first_seed = derive_batch_seed(17, 0)
        second_seed = derive_batch_seed(17, 1)
        first = perturbation.inject_visual_noise(sample, seed=first_seed)
        repeat = perturbation.inject_visual_noise(sample, seed=first_seed)
        second = perturbation.inject_visual_noise(sample, seed=second_seed)
        self.assertTrue(torch.equal(first["visual_features"], repeat["visual_features"]))
        self.assertFalse(torch.equal(first["visual_features"], second["visual_features"]))

    def test_color_jitter_uses_more_than_seed_parity(self):
        sample = batch()
        sample["image"] = torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8)
        perturbation = VisualPerturbation()
        outputs = [
            perturbation.color_jitter(sample, strength=0.5, seed=seed)["image"]
            for seed in (17, 29, 43)
        ]
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[1], outputs[2]))

    def test_mar_excludes_clean_wrong_predictions(self):
        data = [batch()]
        result = GraphLevelAudit(
            top_k_motifs=2, mine_batches=10,
            min_motif_support=1, min_eval_support=1,
        ).run({"stable": StableModel()}, data, motif_loader=data)["stable"]
        self.assertEqual(result["MAR"], 1.0)
        self.assertEqual(result["PIR"], 1.0)
        self.assertEqual(result["WSR"], 1.0)
        self.assertEqual(result["clean_correct_support"], 1)
        self.assertEqual(result["clean_wrong_support"], 1)
        control = result["matched_negative_control"]
        self.assertEqual(control["PIR_terminal_minus_control"]["support"], 2)
        self.assertEqual(control["PIR_terminal_minus_control"]["delta"], 0.0)

    def test_endpoint_ablation_changes_raw_image_but_preserves_graph(self):
        sample = batch()
        sample["image"] = torch.linspace(0, 1, 3 * 10 * 10).reshape(3, 10, 10)
        ablated = ablate_terminal_node(sample, obj_node_idx=1, pair_indices=[0])
        self.assertFalse(torch.equal(ablated["image"], sample["image"]))
        self.assertTrue(torch.equal(ablated["boxes"], sample["boxes"]))
        self.assertTrue(
            torch.equal(ablated["entity_labels"], sample["entity_labels"])
        )
        self.assertTrue(torch.equal(ablated["rel_pairs"], sample["rel_pairs"]))
        self.assertEqual(int(torch.count_nonzero(
            ablated["visual_features"][1]
        )), 0)
        self.assertEqual(int(torch.count_nonzero(
            ablated["union_features"][0]
        )), 0)

    def test_control_node_is_area_matched_and_not_an_endpoint(self):
        boxes = torch.tensor([
            [0.0, 0.0, 0.2, 0.2],
            [0.0, 0.0, 0.4, 0.4],
            [0.0, 0.0, 0.39, 0.39],
            [0.0, 0.0, 0.1, 0.1],
        ])
        selected = _matched_control_node(
            num_nodes=4, excluded={0, 1}, image_index=0,
            terminal_node=1, boxes=boxes,
        )
        self.assertEqual(selected, 2)

    def test_brr_is_undefined_below_support_threshold(self):
        result = PairLevelAudit(
            recall_ks=[1], primary_k=1,
            min_relations=3, min_clean_recall=0.01,
        ).run({"stable": StableModel()}, [batch()])["stable"]
        self.assertTrue(math.isnan(result["BRR"]))
        self.assertEqual(result["status"], "insufficient_relation_support")
        self.assertEqual(result["perturbation_seeds"], [17, 29, 43])

    def test_dose_response_bootstraps_images_not_image_seed_rows(self):
        result = PerturbationSweepAudit(
            recall_ks=[1], levels=[0.0, 1.0], seeds=[17, 29, 43]
        ).run({"stable": StableModel()}, [batch(), batch()])["stable"]
        endpoint = result["strategies"]["visual_noise"]["curve"]["1.000"]["1"]
        self.assertEqual(endpoint["n_paired"], 2)
        self.assertEqual(
            endpoint["bootstrap_unit"],
            "image_after_averaging_perturbation_seeds",
        )
        self.assertEqual(endpoint["successful_seeds_per_image"]["minimum"], 3)


if __name__ == "__main__":
    unittest.main()
