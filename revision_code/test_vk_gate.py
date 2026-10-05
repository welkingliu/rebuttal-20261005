import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import vk_gate_math as m
import vj_mac as original
from vk_gate_queue import stage_plan, completion_path, lane_available
from vk_gate_protocol import SPEC


class NativeVKTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17); torch.set_num_threads(1)
        self.features=F.normalize(torch.randn(8,768),dim=-1)
        self.baseline=torch.full((8,151),-8.); self.baseline[:,1]=2.; self.baseline[:,2]=1.5
        self.state=dict(expert_head=dict(weight=torch.zeros(150,768),bias=torch.full((150,),-10.)),
            centers=F.normalize(torch.randn(150,768),dim=-1), counts=torch.ones(150,dtype=torch.long)*100,
            router=dict(weight=torch.zeros(1,14),bias=torch.tensor([10.]),mean=torch.zeros(14),scale=torch.ones(14)),
            fold_map={"metadata":0})
        self.state["expert_head"]["bias"][1]=10.

    def test_no_label_input_and_frozen_constants(self):
        self.assertEqual(list(inspect.signature(m.repair).parameters),["features","baseline","state"])
        self.assertEqual(original.SPEC["alpha"],.5)
        self.assertEqual(original.SPEC["repair_probability_min"],.6)
        self.assertEqual(SPEC["gate_images"],1000)
        self.assertEqual(SPEC["crop_batch"],8)

    def test_native_backend_matches_reference_cache(self):
        from vk_gate_native import configure_backend
        with patch("torch.backends.cudnn", SimpleNamespace(benchmark=True, deterministic=True, allow_tf32=False)), \
             patch("torch.backends.cuda.matmul", SimpleNamespace(allow_tf32=False)):
            configure_backend()
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_matches_original_vj_vk_probability_route(self):
        result=m.repair(self.features,self.baseline,self.state)
        q=F.linear(self.features,**self.state["expert_head"]).softmax(-1)
        _,p,candidate=original.probabilities(self.baseline,q)
        visual,supported=original.visual_features(self.features,p,candidate,self.state["centers"],self.state["counts"])
        inputs=torch.cat([original.confidence_features(p,q),visual],-1)
        expected,mask,conf=original.route(self.baseline,q,inputs,self.state["router"],supported)
        self.assertTrue(torch.equal(result["probabilities"],expected))
        self.assertTrue(torch.equal(result["selected"],mask)); self.assertTrue(mask.all())
        self.assertTrue(torch.equal(result["repair_probability"],conf))
        self.assertTrue(torch.allclose(result["logits"].softmax(-1)[:,0],self.baseline.softmax(-1)[:,0]))

    def test_unselected_logits_are_exact_noop(self):
        self.state["router"]["bias"].fill_(-10.)
        result=m.repair(self.features,self.baseline,self.state)
        self.assertFalse(result["selected"].any())
        self.assertTrue(torch.equal(result["logits"],self.baseline))

    def test_insufficient_class_support_prevents_changes(self):
        self.state["counts"][0]=4
        result=m.repair(self.features,self.baseline,self.state)
        self.assertFalse(result["selected"].any())

    def test_inputs_checked_before_inference(self):
        with self.assertRaises(ValueError): m.repair(self.features*2,self.baseline,self.state)
        with self.assertRaises(ValueError): m.repair(self.features,self.baseline[:,:150],self.state)

    def test_cross_runtime_fixture_validation(self):
        result=m.repair(self.features,self.baseline,self.state)
        fixture=dict(result,features=self.features,baseline=self.baseline)
        self.assertEqual(m.validate_fixture(fixture,self.state)["objects"],8)
        fixture["selected"]=~fixture["selected"]
        with self.assertRaises(RuntimeError): m.validate_fixture(fixture,self.state)

    def test_sgcls_uses_exact_supplied_box_after_resize(self):
        from vk_gate_native import crop_geometry
        gt=SimpleNamespace(size=(200,100),bbox=torch.tensor([[10.,20.,100.,90.]]))
        c=dict(size=(100,50),proposal_boxes=torch.tensor([[5.000001,10.,50.,45.]]))
        self.assertTrue(torch.equal(crop_geometry(c,gt,"sgcls"),gt.bbox))
        c["proposal_boxes"][0,0]=15.
        with self.assertRaises(RuntimeError): crop_geometry(c,gt,"sgcls")

    def test_sgdet_never_substitutes_gt_boxes(self):
        from vk_gate_native import crop_geometry
        gt=SimpleNamespace(size=(200,100),bbox=torch.tensor([[10.,20.,100.,90.]]))
        c=dict(size=(100,50),proposal_boxes=torch.tensor([[1.,2.,10.,9.]]))
        self.assertTrue(torch.equal(crop_geometry(c,gt,"sgdet"),c["proposal_boxes"]*2))

    def test_lane_does_not_bypass_active_r16_queue(self):
        with patch("vk_gate_queue.read",return_value=dict(status="running",pid=123,task="sgdet")), \
             patch("pathlib.Path.exists",return_value=True),patch("vk_gate_queue.process_alive",return_value=True), \
             patch("vk_gate_queue.subprocess.check_output") as call:
            ok,_=lane_available(1)
            self.assertFalse(ok); call.assert_not_called()

    def test_lane_fails_closed_on_gpu_telemetry_failure(self):
        with patch("pathlib.Path.exists",return_value=False), \
             patch("vk_gate_queue.subprocess.check_output",side_effect=OSError("offline")):
            self.assertFalse(lane_available(1)[0])

    def test_order_requires_smoke_before_gate(self):
        plan=stage_plan("sgcls")
        self.assertEqual([r[2] for r in plan],["export","features","evaluate","export","features","evaluate","decision"])
        self.assertTrue(all(r[1] for r in plan[:3])); self.assertFalse(any(r[1] for r in plan[3:]))
        self.assertEqual(completion_path("sgcls",False,"decision").name,"decision.json")

    def test_actual_hook_preserves_predicates_and_rejects_reordering(self):
        from vi_native import FusionPatch
        context=type("TransformerContext",(nn.Module,),{} )(); context.context_obj=nn.Identity()
        predictor=nn.Identity(); predictor.context_layer=context
        model=SimpleNamespace(roi_heads=SimpleNamespace(relation=SimpleNamespace(
            object_cls_refine=True,predictor=predictor,post_processor=nn.Identity())))
        adapter=FusionPatch(model)
        try:
            relations=torch.randn(3,51); boxes=torch.randn(8,4)
            adapter.capture=dict(size=(100,100),proposal_boxes=boxes)
            adapter.update=dict(size=(100,100),boxes=boxes,native_logits=self.baseline,logits=self.baseline+.1)
            adapter.enabled=True
            output=adapter._after(None,(),([self.baseline],[relations],{}))
            self.assertIs(output[1][0],relations)
            self.assertTrue(torch.equal(output[0][0],self.baseline+.1))
            adapter._post(None,(([relations],output[0]),))
            self.assertTrue(adapter.capture["route_checked"])
            adapter.update["boxes"]=boxes+1
            with self.assertRaises(RuntimeError): adapter._after(None,(),([self.baseline],[relations]))
        finally: adapter.close()


if __name__=="__main__": unittest.main()
