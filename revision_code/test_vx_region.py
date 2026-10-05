import inspect
import unittest
import numpy as np
import torch

from vx_math import balanced_weights, paired_box_iou, objective
from vx_common import SPEC, PRIMARY
from vx_vision import VisualResidual


class RegionTests(unittest.TestCase):
    def test_duplicate_instance_weights(self):
        w = balanced_weights([1, 1, 1, 2, 0, -1], [0, 0, 0, 1, 0, 0], [True, False])
        self.assertAlmostEqual(float(w[:3].sum()), 1.)
        self.assertAlmostEqual(float(w[3]), 1.)
        self.assertAlmostEqual(float(w[4]), .5)
        self.assertEqual(float(w[5]), 0.)

    def test_endpoint_weights_only_training(self):
        w = balanced_weights([1, 1, 2, 0], [0, 0, 1, 0], [True, False], True)
        self.assertEqual(w.tolist(), [1.5, 1.5, 1., .5])
        self.assertEqual(list(inspect.signature(VisualResidual.forward).parameters), ["self", "tokens", "native"])

    def test_background_only(self):
        w = balanced_weights([0, 0], [0, 0], [False])
        self.assertEqual(w.tolist(), [.125, .125])

    def test_bad_supervision_rejected(self):
        with self.assertRaises(ValueError):
            balanced_weights([1], [-1], [True])
        with self.assertRaises(ValueError):
            balanced_weights([-1], [0], [True])

    def test_paired_boxes_pixel_iou(self):
        a = np.array([[0, 0, 9, 9], [0, 0, 9, 9]])
        b = np.array([[0, 0, 9, 9], [10, 10, 19, 19]])
        np.testing.assert_array_equal(paired_box_iou(a, b), [1., 0.])

    def test_ce_has_actual_gradient(self):
        z = torch.randn(3, 151, requires_grad=True)
        meta = dict(target=torch.tensor([1, 2, -1]), native=torch.zeros(3, 151),
                    object_weight=torch.tensor([1., 1., 0.]), diagnostic_weight=torch.tensor([3., 1., 0.]),
                    pairs=torch.tensor([[0, 1]]), emitted_labels=torch.tensor([1, 2, 3]),
                    predicate_logprob=torch.zeros(1), protected=torch.tensor([True]))
        values = objective(z, meta, PRIMARY)
        grad = torch.autograd.grad(values["object_ce"], z, retain_graph=True)[0]
        self.assertGreater(float(grad[:2].abs().sum()), 0)
        self.assertEqual(float(grad[2].abs().sum()), 0)
        values["loss"].backward()
        self.assertTrue(torch.isfinite(z.grad).all())


if __name__ == "__main__":
    unittest.main()
