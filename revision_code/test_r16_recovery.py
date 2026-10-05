import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import torch

from common import ROOT
from r16_recover import finish_pending_validation, pending_is_newer, training_complete


class RecoveryTests(unittest.TestCase):
    def test_pending_validation_not_optimizer_replay(self):
        latest = dict(iteration=2800)
        pending = dict(iteration=3000, pending_validation=True)
        self.assertTrue(pending_is_newer(latest, pending))
        self.assertTrue(pending_is_newer(None, pending))

    def test_completed_validation_not_applied_twice(self):
        self.assertFalse(pending_is_newer(dict(iteration=3000), dict(iteration=3000, pending_validation=True)))
        self.assertFalse(pending_is_newer(dict(iteration=3200), dict(iteration=3000, pending_validation=True)))

    def test_early_stop_goes_straight_to_test(self):
        self.assertTrue(training_complete(dict(iteration=24000, scheduler=dict(stage_count=3))))
        self.assertFalse(training_complete(dict(iteration=24000, scheduler=dict(stage_count=2))))

    def test_budget_exhausted_goes_straight_to_test(self):
        self.assertTrue(training_complete(dict(iteration=75000, scheduler=dict(stage_count=1))))

    def test_pending_validation_preserves_weights_and_sgd_momentum(self):
        from maskrcnn_benchmark.solver.lr_scheduler import WarmupReduceLROnPlateau
        model = torch.nn.Module()
        model.backbone = torch.nn.Identity()
        model.rpn = torch.nn.Identity()
        model.roi_heads = torch.nn.Module()
        model.roi_heads.box = torch.nn.Identity()
        model.roi_heads.relation = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.08, momentum=.9)
        scheduler = WarmupReduceLROnPlateau(optimizer, gamma=.1, patience=2)
        model.roi_heads.relation(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step(None, epoch=2999)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        state = dict(iteration=3000, best=.1, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        with tempfile.TemporaryDirectory(dir=str(ROOT / "tmp"), prefix="r16_recovery_test_") as temp:
            out = Path(temp)
            with patch("r16_recover.r16.build", return_value=(model, None)), \
                 patch("maskrcnn_benchmark.solver.make_optimizer", return_value=optimizer), \
                 patch("maskrcnn_benchmark.solver.make_lr_scheduler", return_value=scheduler), \
                 patch("r16_recover.r16.dataset", return_value=[None]*5000), \
                 patch("r16_recover.restore_rng"), \
                 patch("r16_recover.r16.evaluate", return_value={"R":{"100":.3}}), \
                 patch("torch.cuda.get_rng_state", return_value=torch.get_rng_state()):
                finish_pending_validation("sgcls", out, state, {"test":True})
            saved = torch.load(str(out / "latest.pth"))
            self.assertEqual(saved["iteration"], 3000)
            self.assertEqual(saved["scheduler"]["last_epoch"], 3000)
            self.assertEqual(saved["best"], .3)
            for key, value in weights.items(): self.assertTrue(torch.equal(saved["model"][key], value))
            for key, value in state["optimizer"]["state"].items():
                self.assertTrue(torch.equal(saved["optimizer"]["state"][key]["momentum_buffer"], value["momentum_buffer"]))


if __name__ == "__main__": unittest.main()
