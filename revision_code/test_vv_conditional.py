import inspect
import unittest
import torch
from torch.nn import functional as F

from vv_math import (TRAINED, make_head, design, apply_delta, project_channels,
                     objective, predict, fit, choose_epoch)
from vp_math import split_fold


class ConditionalRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.native = torch.randn(12,151)
        self.visual = F.normalize(torch.randn(12,1536),dim=-1)
        self.labels = torch.tensor([1,2,3,4,0,0,0,0,-1,-1,8,9])

    def test_zero_is_exact_noop(self):
        for name in TRAINED:
            self.assertTrue(torch.equal(predict(self.visual,self.native,make_head(),name),self.native))

    def test_all_arms_finite_nonzero_gradient(self):
        for name in TRAINED:
            head=make_head()
            loss,ce,kl=objective(self.native,head(design(self.visual,self.native,name)),self.labels,name)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(head.weight.grad.norm()),0.)
            if name=="foreground_visual":self.assertEqual(float(head.weight.grad[0].norm()),0.)

    def test_kl_has_gradient_away_from_teacher(self):
        delta=(torch.randn_like(self.native)*.1).requires_grad_(True)
        _,_,kl=objective(self.native,delta,torch.full((12,),-1),"joint_visual")
        kl.backward()
        self.assertGreater(float(delta.grad.norm()),0.)

    def test_background_and_foreground_projection(self):
        q=self.native+torch.randn_like(self.native)*.3
        fg=project_channels(self.native,q,"foreground").softmax(-1)
        bg=project_channels(self.native,q,"background").softmax(-1)
        self.assertTrue(torch.allclose(fg[:,0],self.native.softmax(-1)[:,0],atol=1e-6))
        self.assertTrue(torch.allclose(fg[:,1:]/fg[:,1:].sum(-1,keepdim=True),q[:,1:].softmax(-1),atol=1e-6))
        self.assertTrue(torch.allclose(bg[:,0],q.softmax(-1)[:,0],atol=1e-6))
        self.assertTrue(torch.allclose(bg[:,1:]/bg[:,1:].sum(-1,keepdim=True),self.native[:,1:].softmax(-1),atol=1e-6))
        self.assertTrue(torch.equal(bg[:,1:].argmax(-1),self.native[:,1:].argmax(-1)))

    def test_foreground_fit_preserves_bg(self):
        state,_=fit(self.visual,self.native,self.labels,"foreground_visual",2,device="cpu")
        head=make_head();head.load_state_dict(state)
        logits=predict(self.visual,self.native,head,"foreground_visual")
        self.assertTrue(torch.allclose(logits.softmax(-1)[:,0],self.native.softmax(-1)[:,0],atol=1e-6))

    def test_posterior_control_ignores_pixels(self):
        head=make_head()
        with torch.no_grad():head.weight.normal_(0,.01)
        a=predict(self.visual,self.native,head,"posterior_only")
        b=predict(torch.randn_like(self.visual),self.native,head,"posterior_only")
        self.assertTrue(torch.equal(a,b))

    def test_joint_background_supervision_reaches_bg_row(self):
        head=make_head();y=torch.zeros(12,dtype=torch.long)
        loss,_,_=objective(self.native,head(design(self.visual,self.native,"joint_visual")),y,"joint_visual")
        loss.backward()
        self.assertGreater(float(head.weight.grad[0].norm()),0.)

    def test_no_label_argument_in_repair(self):
        for function in [predict,design,apply_delta,project_channels]:
            self.assertFalse(set(inspect.signature(function).parameters)&{"labels","target","gt"})

    def test_joint_selector_rejects_relation_harm(self):
        rows=[dict(epoch=0,object=.5,R50=.4,mR50=.1),dict(epoch=1,object=.51,R50=.39,mR50=.1)]
        self.assertEqual(choose_epoch(rows)["selected_epoch"],0)
        rows.append(dict(epoch=2,object=.506,R50=.4,mR50=.1))
        self.assertEqual(choose_epoch(rows)["selected_epoch"],2)

    def test_image_split_disjoint(self):
        ids=[str(i) for i in range(30)];mapping={iid:int(iid)%5 for iid in ids}
        split=split_fold(ids,mapping,0)
        self.assertFalse(set(split["held"])&set(split["training"]))
        self.assertFalse(set(split["fit"])&set(split["inner_validation"]))


if __name__=="__main__":unittest.main()
