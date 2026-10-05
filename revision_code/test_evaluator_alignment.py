import unittest
import torch
from evaluator_audit_aligned import prediction_gt_view


class AlignmentTest(unittest.TestCase):
    def test_compacted_prediction_retains_full_gt(self):
        batch=dict(boxes=torch.tensor([[0.,0.,.2,.2],[0.,.9,.2,.9],[.2,.2,.4,.4]]),
                   entity_labels=torch.tensor([2,3,4]),rel_pairs=torch.tensor([[0,1],[0,2]]),rel_labels=torch.tensor([1,2]))
        pred=dict(pred_boxes=batch["boxes"][[0,2]],gt_entity_indices=torch.tensor([0,2]))
        view,indices=prediction_gt_view(pred,batch)
        self.assertEqual(view["entity_labels"].tolist(),[2,4])
        self.assertEqual(len(batch["rel_pairs"]),2)
        self.assertEqual(len(batch["boxes"]),3)
        pred["gt_entity_indices"]=torch.tensor([0,0])
        with self.assertRaises(RuntimeError):
            prediction_gt_view(pred,batch)


if __name__=="__main__":
    unittest.main()
