import unittest
from types import SimpleNamespace

import torch
from torch import nn

from vc_native import ProposalPatch
from vc_protocol import ResidualHead


class Proposal:
    def __init__(self):
        self.fields={"predict_logits":torch.randn(3,151)}
        self.bbox=torch.zeros(3,4)
        self.size=(100,100)
    def get_field(self,key):
        return self.fields[key]
    def add_field(self,key,value):
        self.fields[key]=value


class LSTMContext(nn.Module):
    def forward(self,x,proposals,pairs,logger=None,all_average=False,ctx_average=False):
        return proposals[0].get_field("predict_logits").clone()


class TestRouting(unittest.TestCase):
    def setUp(self):
        predictor=nn.Identity()
        predictor.context_layer=LSTMContext()
        self.relation=SimpleNamespace(object_cls_refine=False,predictor=predictor,post_processor=nn.Identity())
        self.patch=ProposalPatch(SimpleNamespace(roi_heads={"relation":self.relation}))
        self.proposal=Proposal()
        self.original=self.proposal.get_field("predict_logits").clone()
        self.x=torch.randn(3,4096)
    def tearDown(self):
        self.patch.close()
    def before(self):
        self.patch._before(None,([self.proposal],None,None,None,self.x))
    def test_zero_identity(self):
        self.patch.enabled=True
        self.patch.head=ResidualHead()
        self.before()
        self.assertTrue(torch.equal(self.original,self.proposal.get_field("predict_logits")))
    def test_nonzero_and_average_restore(self):
        self.patch.enabled=True
        self.patch.head=ResidualHead()
        with torch.no_grad():
            self.patch.head.linear.bias[150]=20
        self.before()
        updated=self.proposal.get_field("predict_logits").clone()
        observed=self.relation.predictor.context_layer(self.x,[self.proposal],None)
        averaged=self.relation.predictor.context_layer(self.x,[self.proposal],None,ctx_average=True)
        self.assertTrue(torch.equal(observed,updated))
        self.assertTrue(torch.equal(averaged,self.original))
        self.assertTrue(torch.equal(self.proposal.get_field("predict_logits"),updated))
        self.patch._post(None,((None,[updated]),))
        with self.assertRaises(RuntimeError):
            self.patch._post(None,((None,[self.original]),))
    def test_float16_features(self):
        head=ResidualHead()
        out=head(self.x.half(),self.original)
        out.sum().backward()
        self.assertTrue(torch.isfinite(head.linear.weight.grad).all())
    def test_unsupported_route(self):
        self.relation.object_cls_refine=True
        with self.assertRaises(RuntimeError):
            ProposalPatch(SimpleNamespace(roi_heads={"relation":self.relation}))


if __name__=="__main__":
    unittest.main()
