import inspect
import math
import unittest
import torch
from torch.nn import functional as F
from vu_math import (PRIMARY,crop_bounds,view_features,risk_features,allocation,
                     endpoint_impact,apply_expert,fit_expert,fit_risk)


class ReobservationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);self.n=torch.randn(11,151);self.x=F.normalize(torch.randn(11,768),dim=-1)
        self.boxes=torch.tensor([[i,i,i+20,i+30] for i in range(11)]).float()
        self.rx=risk_features(self.n,self.boxes,(100,100));width=self.rx.shape[1]
        self.risk=dict(mean=torch.zeros(width),scale=torch.ones(width),weight=torch.zeros(1,width),bias=torch.zeros(1))
        self.pairs=torch.tensor([[0,1],[1,2],[3,4]]);self.rel=torch.randn(3,51)

    def test_crop_is_prediction_geometry_and_clipped(self):
        self.assertEqual(crop_bounds([5,5,14,14],(20,20),1),(5,5,15,15))
        self.assertEqual(crop_bounds([0,0,19,19],(20,20),1.5),(0,0,20,20))
        with self.assertRaises(ValueError):crop_bounds([100,100,110,110],(20,20),1)

    def test_equal_budget_and_random_determinism(self):
        selections={name:allocation(name,self.n,self.rx,self.pairs,self.rel,"77",self.risk)
                    for name in [PRIMARY,"uncertainty_budget","random_budget"]}
        for value in selections.values():self.assertEqual(int(value.sum()),math.ceil(.2*11))
        self.assertTrue(torch.equal(selections["random_budget"],allocation("random_budget",self.n,self.rx,self.pairs,self.rel,"77",self.risk)))

    def test_risk_and_impact_finite(self):
        self.assertEqual(self.rx.shape,(11,156));self.assertTrue(torch.isfinite(self.rx).all())
        impact=endpoint_impact(self.n,self.pairs,self.rel)
        self.assertTrue(torch.all((impact>=0)&(impact<=1)));self.assertEqual(float(impact[8]),0.)

    def test_unselected_exact_and_background_preserved(self):
        state=dict(weight=torch.randn(150,768)*.1,bias=torch.zeros(150))
        selected=torch.arange(11)<3;out=apply_expert(self.n,self.x,state,selected)
        self.assertTrue(torch.equal(out[~selected],self.n[~selected]))
        self.assertTrue(torch.allclose(out.softmax(-1)[:,0],self.n.softmax(-1)[:,0],atol=1e-6))
        self.assertFalse(torch.equal(out[selected],self.n[selected]))

    def test_view_normalization(self):
        value=view_features(dict(siglip_tight=self.x,siglip_context=self.x),"siglip_dual")
        self.assertEqual(value.shape,(11,1536));self.assertTrue(torch.allclose(value.norm(dim=-1),torch.ones(11),atol=1e-6))

    def test_fit_and_inference_do_not_share_label_arguments(self):
        labels=torch.tensor([1,2,1,2,3,3,0,-1,1,2,0])
        state,info=fit_expert(self.x,labels,validation=(self.x[:3],labels[:3]),smoke=True,device="cpu")
        self.assertGreater(info["selected_epoch"],0);self.assertTrue(torch.isfinite(state["weight"]).all())
        risk=fit_risk(self.rx,self.n,labels,smoke=True,device="cpu")
        self.assertEqual(risk["fitting_proposals"],10)
        for function in [allocation,apply_expert,risk_features]:
            self.assertFalse(set(inspect.signature(function).parameters)&{"labels","target","gt","expert_features"})


if __name__=="__main__":unittest.main()
