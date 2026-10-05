import unittest
import torch
from torch.nn import functional as F
from vq_math import ResidualHead, residual_logits, objective, choose_epoch, early_expert_epoch, fit


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = F.normalize(torch.randn(24, 768), dim=-1)
        self.native = torch.randn(24, 151)
        self.y = torch.arange(24) % 151

    def test_zero_is_exact_native_and_has_gradient(self):
        model = ResidualHead(True); delta = model(self.x)
        self.assertTrue(torch.equal(residual_logits(self.native, delta), self.native))
        objective(self.native, delta, self.y)[0].backward()
        self.assertGreater(float(model.weight.grad.abs().sum()), 0)

    def test_background_preserved_and_bound(self):
        model = ResidualHead(True)
        with torch.no_grad(): model.weight.normal_(0, 20)
        delta = model(self.x); actual = residual_logits(self.native, delta).softmax(-1)
        self.assertLessEqual(float(delta.abs().max()), 1.)
        torch.testing.assert_close(actual[:, 0], self.native.softmax(-1)[:, 0])
        torch.testing.assert_close(actual.sum(-1), torch.ones(len(actual)))

    def test_unmatched_targets_are_not_supervised(self):
        delta = torch.randn(24, 150, requires_grad=True)
        loss, ce, kl = objective(self.native, delta, torch.full((24,), -1))
        self.assertEqual(float(ce), 0.)
        torch.testing.assert_close(loss, kl)
        loss.backward(); self.assertGreater(float(delta.grad.abs().sum()), 0)

    def test_no_visual_control_ignores_features(self):
        model = ResidualHead(False)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 150)
        self.assertTrue(torch.equal(model(self.x), model(torch.randn_like(self.x))))

    def test_selector_rejects_relation_damage_and_prefers_earlier_tie(self):
        base = dict(epoch=0, object=.5, R50=.4, mR50=.1)
        rows = [base, dict(epoch=1, object=.51, R50=.394, mR50=.1),
                dict(epoch=2, object=.505, R50=.397, mR50=.099),
                dict(epoch=4, object=.505, R50=.4, mR50=.1)]
        self.assertEqual(choose_epoch(rows)["selected_epoch"], 2)
        self.assertTrue(choose_epoch(rows[:2])["no_op"])
        self.assertTrue(choose_epoch([base, dict(base, epoch=1)])["no_op"])

    def test_early_control_recovers_pre_ten_optimum(self):
        self.assertEqual(early_expert_epoch([dict(epoch=5, inner_weighted_nll=1.),
            dict(epoch=10, inner_weighted_nll=1.2)]), 5)

    def test_real_fitting_changes_foreground_parameters(self):
        state, history = fit(self.x, self.native, self.y, True, 2, device="cpu")
        self.assertEqual(len(history), 2)
        self.assertGreater(float(state["weight"].abs().max()), 0.)


if __name__ == "__main__": unittest.main()
