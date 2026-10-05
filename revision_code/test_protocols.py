"""Small invariance and ranking tests; no assets or new training required."""
import unittest

import numpy as np
import torch
from torch import nn

from identity_intervention import SemanticReplay, conditions, fingerprint
from mitigation_decomposition import matches, softmax
from paired_visual_control import selected_nodes, mask
from PIL import Image


class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_embed2=nn.Embedding(151,4)
    def forward(self,logits):
        labels=logits[:,1:].argmax(1)+1
        return logits,labels,self.obj_embed2(labels)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_layer=Context()
    def forward(self,logits):
        obj,labels,edge=self.context_layer(logits)
        return [obj],[edge+labels[:,None].float()],{}


class ProtocolTests(unittest.TestCase):
    def test_noop_and_route_separation(self):
        torch.manual_seed(17)
        replay=SemanticReplay(Predictor().eval())
        inputs=(torch.randn(8,151),)
        clean,labels=replay.run(inputs)
        noop,_=replay.run(inputs,labels)
        self.assertTrue(torch.equal(clean[1][0],noop[1][0]))
        changed=labels.roll(1)
        embedding,_=replay.run(inputs,changed,"embedding")
        self.assertTrue(torch.equal(clean[0][0],embedding[0][0]))
        self.assertFalse(torch.equal(clean[1][0],embedding[1][0]))
        freq,_=replay.run(inputs,changed,"frequency")
        self.assertTrue(torch.allclose(clean[0][0].sort(1)[0],freq[0][0].sort(1)[0]))

    def test_matched_nodes_and_nested_strengths(self):
        labels=torch.arange(1,9)
        gt=labels.clone()
        gt[0]=10
        confusion=np.ones((151,151),dtype=int)
        counts=np.arange(151)
        cases={name:(replacement,route,seed,strength) for name,replacement,route,seed,strength in
               conditions(labels,gt,confusion,counts,"test")}
        for seed in (17,23,31):
            last=set()
            for fraction in (25,50,100):
                a=cases["confusable_s%d_p%d"%(seed,fraction)][0]
                b=cases["matched_random_s%d_p%d"%(seed,fraction)][0]
                self.assertTrue(torch.equal(a!=labels,b!=labels))
                changed=set((a!=labels).nonzero().flatten().tolist())
                self.assertNotIn(0,changed)
                self.assertTrue(last<=changed)
                last=changed

    def test_temperature_preserves_argmax(self):
        x=np.array([[1.,3.,-2.],[5.,2.,1.]])
        self.assertTrue(np.array_equal(softmax(x).argmax(1),softmax(x/2).argmax(1)))

    def test_fixed_candidate_label_score_crossing(self):
        z=dict(pred_rel_pairs=np.array([[0,1]]),pred_rel_scores=np.array([[.1,.9]]),
               gt_relations=np.array([[0,1,1]]),gt_labels=np.array([1,2]),
               pred_boxes=np.array([[0.,0.,.2,.2],[.5,.5,1.,1.]]),
               gt_boxes=np.array([[0.,0.,.2,.2],[.5,.5,1.,1.]]))
        rows,_=matches(z,np.array([1,2]),np.array([.6,.6]))
        wrong,_=matches(z,np.array([2,2]),np.array([.6,.6]))
        self.assertEqual(rows[0][0],1.)
        self.assertEqual(wrong[0][0],0.)

    def test_fingerprint_detects_geometry_mutation(self):
        x=torch.ones(3,4)
        old=fingerprint([x])
        x[0,0]=0.
        self.assertNotEqual(old,fingerprint([x]))

    def test_area_matched_control_and_zero_strength(self):
        boxes=np.array([[0.,0.,4.,4.],[5.,5.,8.,8.],[1.,5.,5.,9.]])
        nodes,reason=selected_nodes(boxes,np.array([[0,1]]))
        self.assertIsNone(reason)
        self.assertEqual(nodes[:2],(0,2))
        image=Image.fromarray(np.arange(300,dtype=np.uint8).reshape(10,10,3))
        self.assertTrue(np.array_equal(np.asarray(image),np.asarray(mask(image,boxes[0],0.))))
        self.assertFalse(np.array_equal(np.asarray(image),np.asarray(mask(image,boxes[0],1.))))


if __name__=="__main__":
    unittest.main()
