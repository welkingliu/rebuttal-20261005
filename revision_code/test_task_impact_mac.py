import copy
import inspect
import unittest

import numpy as np

import task_impact_mac as m


class TaskImpactTests(unittest.TestCase):
    def setUp(self):
        self.item=dict(image_id="unit",boxes=[[0,0,10,10],[60,0,70,10],[0,60,10,70]],
            size=[100,100],labels=[1,2,2])
        self.s=m.geometry(self.item["boxes"],self.item["size"])
        self.p=np.zeros((3,151)); self.p[0,1]=1.; self.p[1:,2]=1.
        self.f=np.ones((4,150,150))*.5
        self.q=dict(target=1,anchor=2,relation="left")

    def test_geometry_orientation_and_no_self_pairs(self):
        self.assertTrue(self.s[0,0,1]); self.assertTrue(self.s[1,1,0])
        self.assertTrue(self.s[2,0,2]); self.assertTrue(self.s[3,2,0])
        self.assertFalse(self.s[:,range(3),range(3)].any())

    def test_gap_is_strict_and_normalized(self):
        s=m.geometry([[0,0,0,0],[5,0,5,0]],[100,100])
        self.assertFalse(s.any())
        with self.assertRaises(ValueError): m.geometry([[5,0,1,2]],[100,100])

    def test_query_construction_deterministic_without_predictions(self):
        a,c=m.make_queries(self.item,[1,2,3,4])
        b,_=m.make_queries(copy.deepcopy(self.item),[4,3,2,1])
        self.assertEqual(a,b)
        self.assertEqual(len(a),6)
        self.assertEqual(len({q["id"] for q in a}),6)
        self.assertTrue(all(c[s]>=2 for s in c))
        for task in a:
            pairs=m.possible_pairs(self.item["labels"],task["query"]["target"],task["query"]["anchor"],
                self.s[m.RELATIONS.index(task["query"]["relation"])])
            self.assertEqual(task["answers"],pairs)
            self.assertEqual(bool(pairs),task["stratum"]=="positive")

    def test_solver_has_no_ground_truth_arguments(self):
        self.assertEqual(list(inspect.signature(m.pair_score).parameters),
            ["query","probabilities","spatial","frequencies","mode"])

    def test_geometry_and_oracle_select_the_correct_pair(self):
        pair,score=m.pair_score(self.q,self.p,self.s,self.f,"soft_geometry")
        self.assertEqual(pair,[0,1]); self.assertEqual(score,1.)
        query=dict(self.q,relation="right")
        self.assertEqual(m.pair_score(query,self.p,self.s,self.f,"soft_geometry"),(None,0.))

    def test_soft_geometry_can_disambiguate_without_rewriting_identity(self):
        p=self.p.copy(); p[0,1]=.45; p[0,3]=.55
        hard,_=m.pair_score(self.q,p,self.s,self.f,"hard_geometry")
        soft,score=m.pair_score(self.q,p,self.s,self.f,"soft_geometry")
        self.assertIsNone(hard); self.assertEqual(soft,[0,1]); self.assertEqual(score,.45)
        self.assertEqual(p[0].argmax(),3)

    def test_frequency_is_not_current_image_geometry(self):
        q=dict(self.q,relation="right")
        pair,score=m.pair_score(q,self.p,self.s,self.f,"soft_frequency")
        self.assertEqual(pair,[0,1]); self.assertEqual(score,.5)

    def test_no_self_pair_even_when_one_region_has_both_classes(self):
        p=np.zeros((2,151)); p[0,1:3]=.5; p[1,3]=1.
        self.assertEqual(m.pair_score(self.q,p,self.s[:,:2,:2],self.f,"soft_identity"),(None,0.))

    def test_calibration_ties_and_equality_abstain(self):
        scores=np.linspace(0,1,100)
        value=m.fit_threshold(scores)
        self.assertEqual(value["false_answers"],5)
        self.assertIsNone(m.returned_pair([0,1],value["threshold"],value["threshold"]))
        self.assertEqual(m.fit_threshold(np.ones(100))["false_answers"],0)
        with self.assertRaises(ValueError): m.fit_threshold([.1]*10)

    def test_multiple_valid_answers_and_negative_denominator(self):
        task=dict(query=self.q,stratum="positive",answers=[[0,1],[0,2]])
        self.assertTrue(m.decision_record(task,[0,2],self.item["labels"],self.s)["positive_pair_success"])
        task.update(stratum="category_absent",answers=[])
        absent=m.decision_record(task,None,self.item["labels"],self.s)
        false=m.decision_record(task,[0,1],self.item["labels"],self.s)
        self.assertTrue(absent["correct"]); self.assertTrue(false["false_answer"])
        row=dict(task,image_id="unit",decisions={name:false for name in m.ARMS})
        counts=m.accumulate(["unit"],[row])
        result=m.metric(counts["hard_geometry"].sum(0))
        self.assertEqual(result["negative_false_answer_rate"],1.)
        self.assertIsNone(result["positive_pair_success"])

    def test_mismatched_geometry_is_not_a_ground_truth_answer(self):
        perm=[1,2,0]; wrong=self.s[:,perm][:,:,perm]
        pair,score=m.pair_score(self.q,self.p,wrong,self.f,"soft_geometry")
        self.assertIsNone(pair); self.assertEqual(score,0.)

    def test_paired_bootstrap_uses_same_images_for_each_arm(self):
        counts=np.array([[1,1,1,2,0,1,0,1,0,1,1],[1,0,0,2,1,1,1,1,0,1,0]])
        _,delta=m.paired_intervals({name:counts.copy() for name in m.ARMS})
        for row in delta.values():
            self.assertEqual(row["positive_delta_ci95"],[0.,0.])
            self.assertEqual(row["negative_false_answer_delta_ci95"],[0.,0.])


if __name__=="__main__": unittest.main()
