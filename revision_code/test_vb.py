"""CPU regression checks; native GPU integration is a queued mandatory gate."""
import copy
import json
import types
import unittest

import numpy as np
import torch

from vb_protocol import ResidualHead, objective, proposal_targets, choose_ids, paired_gate, SPEC, summarize_rows
from vb_native import ContextPatch, OfficialMetrics


class VBTests(unittest.TestCase):
    def test_zero_init_and_gradient(self):
        torch.manual_seed(17)
        head=ResidualHead(8,151)
        x=torch.randn(32,8);z=torch.randn(32,151)
        self.assertTrue(torch.equal(head(x,z),z))
        y=torch.tensor([0,2,3,-1]*8)
        loss,ce,kl=objective(head,x,z,y,1.)
        self.assertLess(abs(float(kl)),1e-6)
        loss.backward()
        self.assertGreater(float(head.linear.weight.grad.abs().sum()),0)
        self.assertIsNone(z.grad)
        head.zero_grad()
        with torch.no_grad():
            head.linear.bias[3]=.2
        _,_,kl=objective(head,x,z,y,1.)
        kl.backward()
        self.assertGreater(float(head.linear.bias.grad.abs().sum()),0)

    def test_match_background_ignore_and_positive(self):
        gt=np.array([[0,0,9,9]],dtype=float)
        boxes=np.array([[0,0,9,9],[100,100,110,110],[0,0,4,9]],dtype=float)
        self.assertEqual(proposal_targets(boxes,gt,np.array([7]),"sgdet").tolist(),[7,0,7])
        boxes[-1]=[0,0,3,9]
        self.assertEqual(proposal_targets(boxes,gt,np.array([7]),"sgdet")[-1],-1)
        self.assertEqual(proposal_targets(gt,gt,np.array([7]),"sgcls").tolist(),[7])
        with self.assertRaises(RuntimeError):
            proposal_targets(boxes,gt,np.array([7]),"sgcls")

    def test_selection_reproducible_and_disjoint(self):
        ids=[str(i) for i in range(2000)]
        a=choose_ids(ids,1500,"VB_val:")
        self.assertEqual(a,choose_ids(ids[::-1],1500,"VB_val:"))
        self.assertFalse(set(a[:500])&set(a[500:]))

    @staticmethod
    def rows(n=1000):
        return [dict(image_id=str(i),positive_correct=6,positive_objects=10,post_nms_correct=6,
                     recalls=[.2]*6,class_recalls=[[.2]+[None]*49 for _ in range(6)],
                     calibration_bins=[[10,6,6]]+[[0,0,0]]*14,nll_sum=10.,brier_sum=5.) for i in range(n)]

    def test_gate_refuses_null_and_degradation(self):
        base=self.rows()
        null=paired_gate(base,base)
        self.assertFalse(null["accepted"])
        change=copy.deepcopy(base)
        for row in change:
            row["positive_correct"]=7
        self.assertTrue(paired_gate(base,change)["accepted"])
        for row in change:
            row["recalls"]=[.1]*6
        self.assertFalse(paired_gate(base,change)["accepted"])
        self.assertFalse(paired_gate(base[:10],change[:10])["accepted"])

    def test_strict_json_and_summary(self):
        rows=self.rows(2)
        encoded=json.dumps(rows,allow_nan=False)
        value=summarize_rows(json.loads(encoded))
        self.assertAlmostEqual(value["R"]["50"],.2)
        self.assertAlmostEqual(value["mR"]["50"],.2/50)
        self.assertAlmostEqual(value["object_top1"],.6)

    def test_context_updates_consumer_but_not_average(self):
        x=torch.randn(3,8);z=torch.randn(3,151)
        labels=z[:,1:].argmax(1)+1
        context=type("LSTMContext",(),{})()
        original=(z,labels,x,torch.arange(3),torch.arange(3),None)
        context.obj_ctx=lambda *args,**kwargs:original
        pred=types.SimpleNamespace(context_layer=context)
        model=types.SimpleNamespace(roi_heads={"relation":types.SimpleNamespace(predictor=pred)})
        box=types.SimpleNamespace(bbox=torch.zeros(3,4),size=(100,100))
        patch=ContextPatch(model);patch.head=ResidualHead(8,151);patch.enabled=True
        out=context.obj_ctx(None,[box])
        self.assertTrue(torch.equal(out[0],z));self.assertTrue(torch.equal(out[1],labels))
        with torch.no_grad():
            patch.head.linear.bias[150]=100
        out=context.obj_ctx(None,[box])
        self.assertTrue(torch.equal(out[1],torch.full((3,),150)))
        avg=context.obj_ctx(None,[box],ctx_average=True)
        self.assertTrue(torch.equal(avg[0],z));self.assertTrue(torch.equal(avg[1],labels))
        patch.close()

    def test_native_metrics_row(self):
        from pysgg.structures.bounding_box import BoxList
        boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
        gt=BoxList(boxes,(100,100),"xyxy")
        gt.add_field("labels",torch.tensor([1,2]))
        gt.add_field("relation_tuple",torch.tensor([[0,1,3]]))
        pred=BoxList(boxes.clone(),(100,100),"xyxy")
        pred.add_field("pred_labels",torch.tensor([1,2]))
        pred.add_field("pred_scores",torch.tensor([.9,.9]))
        pred.add_field("rel_pair_idxs",torch.tensor([[0,1]]))
        rel=torch.zeros(1,51);rel[0,3]=1
        pred.add_field("pred_rel_scores",rel)
        logits=torch.zeros(2,151);logits[0,1]=10;logits[1,2]=10
        row=OfficialMetrics("sgcls").row("sample",pred,gt,{"logits":logits},np.array([1,2]))
        self.assertEqual(row["recalls"],[1.]*6)
        self.assertEqual(row["positive_correct"],2)
        json.dumps(row,allow_nan=False)
        summary=summarize_rows([row])
        self.assertAlmostEqual(summary["mR"]["50"],1/50)
        self.assertEqual(summary["object_macro_accuracy"],1)

    def test_native_nms_permutation(self):
        from pysgg.modeling.roi_heads.relation_head.utils_relation import obj_prediction_nms
        torch.manual_seed(3)
        x=torch.randn(3,8);z=torch.randn(3,151)
        boxes=torch.tensor([[0.,0.,10.,10.],[2.,2.,12.,12.],[50.,50.,60.,60.]]).unsqueeze(1).repeat(1,151,1)
        perm=torch.tensor([2,0,1]);inv=torch.argsort(perm)
        labels=obj_prediction_nms(boxes[perm],z[perm],.3)[inv]
        context=type("LSTMContext",(),{})()
        context.decoder_rnn=types.SimpleNamespace(nms_thresh=.3)
        context.obj_ctx=lambda *a,**k:(z,labels,x,perm,inv,None)
        model=types.SimpleNamespace(roi_heads={"relation":types.SimpleNamespace(predictor=types.SimpleNamespace(context_layer=context))})
        box=types.SimpleNamespace(bbox=boxes[:,0],size=(100,100))
        patch=ContextPatch(model);patch.head=ResidualHead(8,151);patch.enabled=True
        actual=context.obj_ctx(None,[box],boxes_per_cls=boxes)
        self.assertTrue(torch.equal(actual[1],labels))
        patch.close()


if __name__=="__main__":
    unittest.main()
