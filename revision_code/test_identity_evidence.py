import unittest
import numpy as np
from identity_evidence import geometry_assignment, paired_micro, cell_counts, summarize_cells, image_record


class IdentityEvidenceTest(unittest.TestCase):
    def test_geometry_one_to_one_and_unmatched(self):
        boxes = np.array([[0,0,9,9], [0,0,9,9], [30,30,39,39], [90,90,99,99]])
        gt = np.array([[0,0,9,9], [30,30,39,39]])
        mapping, overlaps = geometry_assignment(boxes, gt)
        self.assertEqual(sorted(mapping[mapping>=0].tolist()), [0,1])
        self.assertEqual(mapping[-1], -1)
        np.testing.assert_equal(overlaps[mapping>=0], 1.)

    def test_zero_support_is_undefined(self):
        self.assertIsNone(paired_micro([[0,0]])["value"])

    def test_repairs_and_damage_do_not_cancel_in_report(self):
        row = cell_counts(np.array([0,1,1]), np.array([1,0,1]), np.array([1,1,1]), np.ones(3,dtype=bool))
        summary = summarize_cells([row])
        self.assertEqual(summary["delta_hit1"]["value"], 0.)
        self.assertEqual(summary["repair_rate_given_clean_wrong"]["value"], 1.)
        self.assertEqual(summary["damage_rate_given_clean_correct"]["value"], .5)

    def test_identity_matching_is_separate_from_predicate(self):
        z = dict(object_gt=np.array([1,2]), object_pred=np.array([1,3]), relation_pairs=np.array([[0,1]]),
                 relation_gt=np.array([4]), clean_prediction=np.array([4]), prediction_clean=np.array([4]),
                 incident_clean=np.array([False]))
        row = image_record(z)
        self.assertEqual(row["fixed_pair_label_accounting"]["lost_to_identity_with_predicate_correct"], 1)
        self.assertEqual(row["conditions"]["clean"]["all"]["correct"], 1)

    def test_image_weighted_micro_not_mean_of_image_ratios(self):
        self.assertAlmostEqual(paired_micro([[9,10],[0,1]])["value"], 9/11)

    def test_candidate_coverage_keeps_duplicate_gt_annotations(self):
        from sgdet_identity import evaluable_pairs
        relations = np.array([[0,1,3],[0,1,4],[1,0,3],[2,1,5]])
        a,b,pairs,counts = evaluable_pairs(relations,np.array([0,1]),np.array([[0,1]]))
        np.testing.assert_equal(a,[0,1]); np.testing.assert_equal(b,[0,0])
        self.assertEqual(counts["missing_endpoint"],1)
        self.assertEqual(counts["missing_native_candidate"],1)

    def test_matching_maximizes_valid_pair_count_before_iou(self):
        # Both predictions prefer GT0, but the first can also match GT1.
        gt = np.array([[0,0,9,9], [4,0,13,9]])
        boxes = np.array([[1,0,10,9], [0,0,7,9]])
        mapping, ious = geometry_assignment(boxes,gt)
        np.testing.assert_equal(mapping,[1,0])
        self.assertTrue((ious>=.5).all())

    def test_corruption_keeps_unmatched_proposals_and_pairs_routes(self):
        import torch
        from sgdet_identity import conditions
        labels=torch.tensor([1,2,3,4]); target=torch.tensor([1,5,3,4]); matched=torch.tensor([True,True,False,True])
        rows=list(conditions(labels,target,matched,np.zeros((151,151),dtype=int),np.ones(151,dtype=int),"test"))
        self.assertEqual(len(rows),58)
        for name,replacement,route in rows:
            if replacement is not None:
                self.assertEqual(replacement[2].item(),3)
            if name.startswith(("confusable","matched_random")):
                self.assertEqual(replacement[1].item(),2)
        for start in range(4,len(rows),3):
            self.assertTrue(torch.equal(rows[start][1],rows[start+1][1]))
            self.assertTrue(torch.equal(rows[start][1],rows[start+2][1]))


if __name__ == "__main__":
    unittest.main()
