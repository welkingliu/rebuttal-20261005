import unittest

import torch

from ve_native import SharedContextAdapter, objective_terms, objective, pair_log_scores


class VETests(unittest.TestCase):
    def test_existing_gate_is_not_relaxed(self):
        from ve_protocol import SPEC
        from vc_protocol import SPEC as PREVIOUS
        self.assertEqual({k: v for k, v in SPEC["gate"].items() if k != "candidate"},
                         {k: v for k, v in PREVIOUS["gate"].items() if k != "candidate"})
        self.assertEqual((SPEC["min_epochs"], SPEC["max_epochs"], SPEC["patience"]), (3, 5, 1))

    def test_zero_update_and_bound(self):
        torch.manual_seed(17)
        head = SharedContextAdapter(16, 4)
        x = torch.randn(6, 16)
        self.assertTrue(torch.equal(head(x), x))
        with torch.no_grad():
            head.up.bias.fill_(100)
        limit = .1 * x.square().mean(-1, keepdim=True).sqrt()
        self.assertTrue(bool(((head(x) - x).abs() <= limit + 1e-6).all()))

    def test_losses_have_real_gradients(self):
        torch.manual_seed(17)
        obj = torch.randn(4, 5, requires_grad=True)
        rel = torch.randn(4, 3, requires_grad=True)
        pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        teacher = ([obj.detach() + .3 * torch.randn_like(obj)], [rel.detach() + .3 * torch.randn_like(rel)])
        terms = objective_terms(([obj], [rel]), teacher, torch.tensor([1, 2, 0, -1]), pairs)
        for key, loss in terms.items():
            gradients = torch.autograd.grad(loss, [obj, rel], retain_graph=True, allow_unused=True)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients if g is not None), 0)
        self.assertEqual(objective(terms, "supervised"), terms["object_ce"])
        self.assertTrue(torch.allclose(objective(terms, "relation_aware"), sum(terms.values())))

    def test_equal_teacher_consistency_zero(self):
        o = torch.randn(3, 5)
        r = torch.randn(3, 4)
        p = torch.tensor([[0, 1], [1, 2], [2, 0]])
        t = objective_terms(([o], [r]), ([o], [r]), torch.tensor([1, 2, 3]), p)
        for key in ["object_kl", "predicate_kl", "pair_score_kl"]:
            self.assertLess(abs(float(t[key])), 1e-6)
        self.assertEqual(tuple(pair_log_scores(o, r, p).shape), (3,))


if __name__ == "__main__":
    unittest.main()
