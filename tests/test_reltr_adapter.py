import unittest

import torch

from sgg_core.audits.standard_sgg_eval import _build_ranked_triplets
from sgg_core.models.adapters.reltr import _scene_graph_from_outputs


class RelTRAdapterTest(unittest.TestCase):
    def test_vg_head_preserves_existing_background_slot(self):
        sub_logits = torch.full((1, 152), -10.0)
        obj_logits = torch.full((1, 152), -10.0)
        sub_logits[0, 1] = 10.0
        obj_logits[0, 150] = 10.0
        rel_logits = torch.full((1, 52), -10.0)
        rel_logits[0, 1] = 10.0
        output = _scene_graph_from_outputs({
            "sub_logits": sub_logits,
            "obj_logits": obj_logits,
            "sub_boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2]]),
            "obj_boxes": torch.tensor([[0.75, 0.75, 0.2, 0.2]]),
            "rel_logits": rel_logits,
        })

        self.assertEqual(tuple(output["pred_entity_scores"].shape), (2, 151))
        ranked = _build_ranked_triplets(
            output,
            {
                "boxes": torch.empty((0, 4)),
                "entity_labels": torch.empty(0, dtype=torch.long),
            },
            "sgdet",
            graph_constraint=True,
        )
        self.assertEqual(ranked.subj_labels.tolist(), [1])
        self.assertEqual(ranked.obj_labels.tolist(), [150])
        self.assertEqual(ranked.predicates.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
