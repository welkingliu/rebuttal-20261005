import unittest

import numpy as np
import torch

from external_coverage import audit, normalize
from vc_protocol import ResidualHead, choose_ids
from vd_experiment import ShrunkHead


class ExtensionTests(unittest.TestCase):
    def test_residual_shrinkage(self):
        head = ResidualHead(4, 3)
        with torch.no_grad():
            head.linear.bias[2] = 2.
        x, z = torch.randn(8, 4), torch.randn(8, 3)
        self.assertTrue(torch.equal(ShrunkHead(head, 0)(x, z), z))
        self.assertTrue(torch.allclose(ShrunkHead(head, .25)(x, z) - z, torch.tensor([0., 0., .5]).expand_as(z)))
        self.assertTrue(torch.allclose(ShrunkHead(head, 1)(x, z), head(x, z)))

    def test_coverage_denominators(self):
        graphs = [("a", {"0": "person", "1": "bicycle", "2": "unknown"},
                   [("0", "on", "1"), ("0", "above", "1"), ("2", "on", "0"), ("2", "unknown", "0")]),
                  ("b", {}, [])]
        r = audit(graphs, {"person", "bicycle"}, {"on"})
        self.assertEqual(r["counts"]["relations"], 4)
        self.assertEqual(r["counts"]["images"], 2)
        self.assertEqual(r["relation_instance_coverage"], .25)
        self.assertEqual(r["counts"]["endpoint_unmapped/predicate_unmapped"], 1)
        self.assertEqual(normalize(" IN_front-of  "), "in front of")

    def test_unused_split(self):
        pool = set(map(str, range(5000)))
        excluded = set(map(str, range(2500)))
        dev = choose_ids(pool - excluded, 500, "VD_development:")
        gate = choose_ids(pool - excluded - set(dev), 1000, "VD_gate:")
        self.assertFalse(set(dev) & set(gate))
        self.assertFalse((set(dev) | set(gate)) & excluded)
        self.assertEqual(len(gate), 1000)


if __name__ == "__main__":
    unittest.main()
