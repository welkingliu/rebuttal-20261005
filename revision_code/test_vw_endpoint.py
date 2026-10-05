import unittest
import numpy as np
import torch

from vw_math import make_head, predict, relation_protection, terms, gradient_audit, fit
from vp_math import split_fold


def fixture():
    torch.manual_seed(9)
    n = 8
    pairs = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j])
    protected = torch.zeros(len(pairs), dtype=torch.bool)
    protected[:5] = True
    return dict(visual=torch.randn(n, 1536), native=torch.randn(n, 151),
                target=torch.arange(1, n+1), endpoint=torch.tensor([True]*4 + [False]*4),
                pairs=pairs, emitted_labels=torch.arange(1, n+1),
                predicate_logprob=torch.full((len(pairs),), -.2), protected=protected)


class EndpointTests(unittest.TestCase):
    def test_noop(self):
        r = fixture()
        self.assertTrue(torch.equal(predict(make_head(), r["visual"], r["native"]), r["native"]))

    def test_zero_protection_at_native(self):
        r = fixture()
        self.assertEqual(float(relation_protection(r["native"], r["native"], r)), 0.)

    def test_nonzero_gradients(self):
        report = gradient_audit(fixture())
        self.assertTrue(all(v > 0 for v in report["loss_parameter_gradient_norms"].values()))

    def test_loss_ablation(self):
        r = fixture()
        updated = r["native"].clone()
        updated[:, 1:] -= .5
        p = terms(updated, r["native"], r["target"], r, "endpoint_relation")
        c = terms(updated, r["native"], r["target"], r, "endpoint_ce")
        self.assertAlmostEqual(float(p["loss"]-c["loss"]), float(p["relation_protection"]), places=5)

    def test_prediction_does_not_accept_labels(self):
        r = fixture()
        head = make_head()
        p = predict(head, r["visual"], r["native"])
        r["target"].fill_(150)
        r["protected"].fill_(False)
        self.assertTrue(torch.equal(p, predict(head, r["visual"], r["native"])))

    def test_no_protected_pairs(self):
        r = fixture()
        r["protected"].fill_(False)
        updated = r["native"].clone().requires_grad_()
        loss = relation_protection(updated, r["native"], r)
        loss.backward()
        self.assertTrue(torch.equal(updated.grad, torch.zeros_like(updated)))

    def test_nested_ids(self):
        ids = [str(i) for i in range(50)]
        split = split_fold(ids, {i: int(i)%5 for i in ids}, 0)
        for a, b in [("held", "training"), ("fit", "inner_validation")]:
            self.assertFalse(set(split[a]) & set(split[b]))
        self.assertEqual(set(split["training"]), set(split["fit"]) | set(split["inner_validation"]))

    def test_training_reaches_parameters(self):
        state, history = fit([fixture()], "endpoint_relation", 1, device="cpu")
        self.assertEqual(len(history), 1)
        self.assertGreater(sum(float(x.abs().sum()) for x in state.values()), 0)


if __name__ == "__main__":
    unittest.main()
