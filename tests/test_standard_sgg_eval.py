import unittest

import torch

from sgg_core.audits.standard_sgg_eval import (
    RankedTriplets,
    StandardSGGAudit,
    _build_ranked_triplets,
    _matched_gt_indices,
    box_iou,
)


def _batch(image_id, relations):
    boxes = torch.tensor([
        [0.0, 0.0, 0.4, 0.4],
        [0.6, 0.6, 1.0, 1.0],
        [0.0, 0.6, 0.4, 1.0],
        [0.6, 0.0, 1.0, 0.4],
    ])
    labels = torch.tensor([1, 2, 3, 4])
    pairs = torch.tensor([[s, o] for s, o, _ in relations], dtype=torch.long)
    predicates = torch.tensor([p for _, _, p in relations], dtype=torch.long)
    return {
        "image_id": image_id,
        "boxes": boxes,
        "entity_labels": labels,
        "rel_pairs": pairs,
        "rel_labels": predicates,
    }


class FakeSceneGraphModel:
    supports_standard_sgg = True
    implementation_kind = "test_double"

    def __init__(self, failed_images=(), bad_sgdet_boxes=False):
        self.failed_images = set(failed_images)
        self.bad_sgdet_boxes = bad_sgdet_boxes

    def eval(self):
        return self

    def predict_scene_graph(self, batch, task):
        labels = batch["entity_labels"]
        entity_scores = torch.full((labels.numel(), 6), -8.0)
        entity_scores[torch.arange(labels.numel()), labels] = 8.0

        rel_scores = torch.full((batch["rel_labels"].numel(), 5), -8.0)
        for row, predicate in enumerate(batch["rel_labels"]):
            predicted = 4 if batch["image_id"] in self.failed_images else int(predicate)
            rel_scores[row, predicted] = 8.0

        boxes = batch["boxes"].clone()
        if task == "sgdet" and self.bad_sgdet_boxes:
            boxes = boxes + 2.0
        return {
            "pred_boxes": boxes,
            "pred_entity_scores": entity_scores,
            "pred_rel_pairs": batch["rel_pairs"].clone(),
            "pred_rel_scores": rel_scores,
            "pred_box_scores": torch.ones(labels.numel()),
        }


class JointFakeSceneGraphModel(FakeSceneGraphModel):
    supports_joint_task_inference = True

    def __init__(self):
        super().__init__()
        self.joint_calls = 0

    def predict_scene_graph_tasks(self, batch, tasks):
        self.joint_calls += 1
        return {task: self.predict_scene_graph(batch, task) for task in tasks}


class SGDetOnlySceneGraphModel(FakeSceneGraphModel):
    supported_tasks = ("sgdet",)


class StandardSGGEvalTest(unittest.TestCase):
    def test_independent_relation_probabilities_are_not_softmaxed(self):
        batch = _batch("one", [(0, 1, 1)])
        prediction = {
            "pred_boxes": batch["boxes"],
            "pred_entity_scores": torch.nn.functional.one_hot(
                batch["entity_labels"], num_classes=6
            ).float(),
            "pred_rel_pairs": torch.tensor([[0, 1]]),
            "pred_rel_scores": torch.tensor([[0.0, 0.9, 0.8]]),
            "pred_rel_score_mode": "independent_probabilities",
        }
        ranked = _build_ranked_triplets(
            prediction, batch, task="sgdet", graph_constraint=True
        )
        self.assertAlmostEqual(float(ranked.scores[0]), 0.9, places=6)

    def test_one_prediction_matches_duplicate_ground_truth_rows(self):
        ranked = RankedTriplets(
            scores=torch.tensor([1.0]),
            subj_boxes=torch.tensor([[0.0, 0.0, 0.4, 0.4]]),
            obj_boxes=torch.tensor([[0.6, 0.6, 1.0, 1.0]]),
            subj_labels=torch.tensor([1]),
            predicates=torch.tensor([2]),
            obj_labels=torch.tensor([3]),
        )
        gt = {
            "subj_boxes": ranked.subj_boxes.repeat(2, 1),
            "obj_boxes": ranked.obj_boxes.repeat(2, 1),
            "subj_labels": ranked.subj_labels.repeat(2),
            "predicates": ranked.predicates.repeat(2),
            "obj_labels": ranked.obj_labels.repeat(2),
        }
        self.assertEqual(_matched_gt_indices(ranked, gt, 0.5), {0, 1})

    def test_task_limited_model_reports_unsupported_tasks_without_fabricating_results(self):
        loader = [_batch("one", [(0, 1, 1)])]
        result = StandardSGGAudit(ks=[5]).run(
            {"sgdet_only": SGDetOnlySceneGraphModel()}, loader
        )["sgdet_only"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["tasks"]), {"sgdet"})
        self.assertEqual(result["unsupported_tasks"], ["predcls", "sgcls"])
        self.assertEqual(result["tasks"]["sgdet"]["num_images"], 1)

    def test_joint_task_adapter_uses_one_call_per_image(self):
        loader = [_batch("one", [(0, 1, 1)]), _batch("two", [(2, 3, 2)])]
        model = JointFakeSceneGraphModel()
        result = StandardSGGAudit(ks=[5]).run({"joint": model}, loader)["joint"]
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["joint_task_inference"])
        self.assertEqual(model.joint_calls, len(loader))

    def test_perfect_prediction_all_tasks_and_zero_shot(self):
        loader = [_batch("one", [(0, 1, 1), (2, 3, 2)])]
        audit = StandardSGGAudit(
            ks=[1, 5, 10],
            seen_triplets={(1, 1, 2)},
        )
        result = audit.run({"perfect": FakeSceneGraphModel()}, loader)["perfect"]
        self.assertEqual(result["status"], "ok")
        for task in ("predcls", "sgcls", "sgdet"):
            metrics = result["tasks"][task]["metrics"]
            self.assertEqual(metrics["R@5"], 1.0)
            self.assertEqual(metrics["mR@5"], 1.0)
            self.assertEqual(metrics["imR@5"], 1.0)
            self.assertEqual(metrics["zR@5"], 1.0)

    def test_recall_is_image_macro_and_micro_is_reported_separately(self):
        loader = [
            _batch("hit", [(0, 1, 1)]),
            _batch("miss", [(0, 1, 1), (2, 3, 2), (0, 2, 3)]),
        ]
        audit = StandardSGGAudit(ks=[5], tasks=["predcls"])
        metrics = audit.run(
            {"mixed": FakeSceneGraphModel(failed_images={"miss"})}, loader
        )["mixed"]["tasks"]["predcls"]["metrics"]
        self.assertEqual(metrics["R@5"], 0.5)
        self.assertEqual(metrics["microR@5"], 0.25)

    def test_sgdet_requires_both_boxes_to_match_iou(self):
        loader = [_batch("one", [(0, 1, 1)])]
        audit = StandardSGGAudit(ks=[5], tasks=["sgdet"], iou_threshold=0.5)
        metrics = audit.run(
            {"bad_boxes": FakeSceneGraphModel(bad_sgdet_boxes=True)}, loader
        )["bad_boxes"]["tasks"]["sgdet"]["metrics"]
        self.assertEqual(metrics["R@5"], 0.0)
        self.assertEqual(float(box_iou(loader[0]["boxes"], loader[0]["boxes"]).diag().min()), 1.0)


if __name__ == "__main__":
    unittest.main()
