"""Small deterministic contract tests; actual CUDA smoke is a separate queue gate."""
import unittest

import torch

from repro_experiment import DeterministicImages
from repro_protocol import SPEC
from show_status import verdict


class ReproductionTests(unittest.TestCase):
    def test_distributed_sampler_and_resume(self):
        full = list(DeterministicImages(21,0,96))
        left = list(DeterministicImages(21,0,96,rank=0,world_size=2))
        right = list(DeterministicImages(21,0,96,rank=1,world_size=2))
        self.assertEqual(full, [v for pair in zip(left,right) for v in pair])
        resumed = list(DeterministicImages(21,32,96,rank=0,world_size=2))
        self.assertEqual(resumed,left[16:])

    def test_toy_accumulation(self):
        torch.manual_seed(666)
        x,y = torch.randn(16,3),torch.randn(16,1)
        model = torch.nn.Linear(3,1)
        torch.nn.functional.mse_loss(model(x),y).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        for a,b in zip(x,y):
            (torch.nn.functional.mse_loss(model(a),b)/16).backward()
        for a,p in zip(expected,model.parameters()):
            self.assertTrue(torch.allclose(a,p.grad,atol=1e-6))

    def test_frozen_recipe(self):
        self.assertEqual(SPEC["training"]["optimizer_steps"] * SPEC["training"]["effective_images_per_update"],256000)
        self.assertTrue(all(SPEC["flags"].values()))

    def test_execution_not_acceptance(self):
        self.assertIn("NOT",verdict(dict(status="complete",accepted=False)))
        self.assertIn("gap",verdict(dict(status="complete",reproduction_passed=False)))


if __name__ == "__main__":
    unittest.main()
