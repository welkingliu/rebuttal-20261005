import copy
from pathlib import Path
import tempfile
import time
import unittest

import torch

import vk_mac as m
from vj_mac import group_folds


class CalibrationSourceTests(unittest.TestCase):
    def test_balanced_image_folds_do_not_depend_on_input_order(self):
        ids=[str(i) for i in range(500)]
        a=m.fold_map(ids)
        self.assertEqual(a,m.fold_map(list(reversed(ids))))
        self.assertEqual([list(a.values()).count(i) for i in range(5)],[100]*5)
        with self.assertRaises(ValueError): m.fold_map(["a","a"])

    def test_objects_of_same_image_stay_together(self):
        data=dict(image_ids=["a","b"],offsets=[0,3,5],labels=torch.ones(5))
        a=group_folds(data,dict(a=0,b=1))
        self.assertEqual(a.tolist(),[0,0,0,1,1])

    def test_holdout_features_and_labels_cannot_affect_fit(self):
        torch.manual_seed(3); torch.set_num_threads(1)
        x=torch.randn(220,14); y=(torch.arange(220)%2).float()
        fit=torch.arange(220)<200
        useful=torch.ones(220,dtype=torch.bool); eligible=useful.clone()
        other=x.clone(); other[~fit]+=100
        y2=y.clone(); y2[~fit]=1-y2[~fit]
        with tempfile.TemporaryDirectory() as temp:
            a=m.router_fit(x,y,useful,eligible,fit,Path(temp),"a",time.monotonic())
            b=m.router_fit(other,y2,useful,eligible,fit,Path(temp),"b",time.monotonic())
        for k in a: self.assertTrue(torch.equal(a[k],b[k]),k)

    def test_zero_utility_excluded_and_target_is_conditional(self):
        p=torch.tensor([[.8,.2],[.8,.2],[.8,.2],[.8,.2]])
        candidate=torch.tensor([[0.,.2,.8],[0.,.2,.8],[0.,.8,.2],[0.,.8,.2]])
        useful,target=m.informative_targets(p,candidate,torch.tensor([2,1,1,2]))
        self.assertEqual(useful.tolist(),[True,True,False,False])
        self.assertEqual(target[:2].tolist(),[1.,0.])

    def test_requirements_unchanged(self):
        self.assertEqual(m.SPEC['training']['repair_probability_min'],.6)
        self.assertEqual(m.SPEC['training']['alpha'],.5)
        self.assertEqual(m.SPEC['screening']['identity_gain_min'],.005)
        self.assertEqual(m.SPEC['gate']['relation_noninferiority_margin'],.005)
        self.assertTrue(m.SPEC['gate']['requires_both_tasks'])


if __name__=='__main__': unittest.main()
