import unittest

import torch

from vf_native import RichReadout, objective_terms


class VFTests(unittest.TestCase):
    def test_exact_zero_initialization(self):
        head = RichReadout(8, 4, 6, 5)
        x, logits = torch.randn(7, 12), torch.randn(7, 5)
        self.assertTrue(torch.equal(head(x, logits), logits))

    def test_every_declared_loss_reaches_head(self):
        torch.manual_seed(17)
        head = RichReadout(8, 4, 6, 5)
        with torch.no_grad():
            head.output.weight.normal_(0, .1)
        x, base = torch.randn(4, 12), torch.randn(4, 5)
        terms = objective_terms(head(x, base), base, torch.tensor([1, 2, 0, -1]),
                                torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.randn(3))
        self.assertEqual(set(terms), {"object_ce", "object_kl", "pair_score_kl"})
        for term in terms.values():
            gradients = torch.autograd.grad(term, tuple(head.parameters()), retain_graph=True)
            self.assertGreater(sum(float(g.square().sum()) for g in gradients), 0)

    def test_no_update_zero_consistency(self):
        base = torch.randn(3, 5)
        terms = objective_terms(base, base, torch.tensor([1, 2, 3]),
                                torch.tensor([[0, 1], [1, 2]]), torch.randn(2))
        self.assertLess(abs(float(terms["object_kl"])), 1e-6)
        self.assertLess(abs(float(terms["pair_score_kl"])), 1e-6)

    def test_no_relaxed_gate_or_test_selection(self):
        from vf_protocol import SPEC
        from ve_protocol import SPEC as VE
        self.assertEqual({k: v for k, v in SPEC["gate"].items() if k != "candidate"},
                         {k: v for k, v in VE["gate"].items() if k != "candidate"})
        self.assertIn("development", SPEC["selection"])
        self.assertIn("no test tonight", SPEC["plan"])


if __name__ == "__main__":
    unittest.main()
