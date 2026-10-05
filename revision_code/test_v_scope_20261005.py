"""Synthetic regression checks of the released adapter, not a native-model rerun."""
import unittest
import torch
from torch import nn
from sgg_core.models.adapters.pysgg_live import PySGGLiveAdapter, _identity_linear


class ReadoutScope(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = PySGGLiveAdapter.__new__(PySGGLiveAdapter)
        nn.Module.__init__(self.model)
        self.model.entity_calibrator = _identity_linear(151)
        self.model.relation_calibrator = _identity_linear(51)
        self.raw = dict(pred_entity_scores=torch.randn(8,151).softmax(-1),
                        pred_rel_scores=torch.randn(12,51).softmax(-1))

    def test_unmodified_readout_preserves_probabilities(self):
        output = self.model._calibrate(self.raw)
        for key in self.raw:
            torch.testing.assert_close(output[key].softmax(-1), self.raw[key])

    def test_object_update_does_not_change_predicate_logits(self):
        before = self.model._calibrate(self.raw)
        with torch.no_grad():
            self.model.entity_calibrator.bias[1] += 1
        after = self.model._calibrate(self.raw)
        self.assertTrue(torch.equal(before['pred_rel_scores'],after['pred_rel_scores']))
        self.assertFalse(torch.equal(before['pred_entity_scores'],after['pred_entity_scores']))

    def test_relation_loss_has_no_object_readout_gradient(self):
        loss = self.model._calibrate(self.raw)['pred_rel_scores'].square().mean()
        gradients = torch.autograd.grad(loss,tuple(self.model.entity_calibrator.parameters()),allow_unused=True)
        self.assertTrue(all(g is None for g in gradients))

    def test_object_loss_reaches_readout_not_detached_inputs(self):
        scores = self.model._calibrate(self.raw)['pred_entity_scores']
        loss = nn.functional.cross_entropy(scores,torch.arange(8))
        gradients = torch.autograd.grad(loss,tuple(self.model.entity_calibrator.parameters()))
        self.assertTrue(all(g.abs().sum()>0 for g in gradients))
        self.assertFalse(self.raw['pred_entity_scores'].requires_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
