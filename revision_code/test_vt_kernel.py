import unittest
import torch
from torch.nn import functional as F
from vt_kernel_math import fit_bandwidth, kernel, fit_models, predict_logits
from vp_math import split_fold


class KernelScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.x = F.normalize(torch.randn(40,768), dim=-1)
        self.n = torch.randn(40,151)
        self.y = torch.tensor([1,2,3,4,0,-1,1,2]*5)

    def test_scale_active_and_noop(self):
        states = fit_models(self.x,self.n,self.y,device="cpu")
        for s in states.values():
            self.assertGreater(s["diagnostics"]["fit_mean_kernel_activation"], .1)
            self.assertLess(s["diagnostics"]["relative_solve_residual"], 1e-6)
            self.assertTrue(torch.equal(predict_logits(self.x,self.n,s,0),self.n))
            out = predict_logits(self.x,self.n,s,1)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.allclose(out.softmax(-1)[:,0],self.n.softmax(-1)[:,0],atol=1e-6))
            self.assertGreater(float((out-self.n).abs().max()),1e-3)
        self.assertTrue(torch.equal(states["median_kernel"]["scale"],torch.ones_like(states["median_kernel"]["scale"])))

    def test_bandwidth_uses_fit_landmarks_and_is_not_point07(self):
        b=fit_bandwidth(self.x)
        self.assertGreater(b,.5)
        self.assertTrue(torch.allclose(kernel(self.x,self.x,b).diag(),torch.ones(40),atol=1e-5))

    def test_no_fit_eval_overlap(self):
        ids=[str(i) for i in range(100)]; folds={i:int(i)%5 for i in ids}
        s=split_fold(ids,folds,0)
        self.assertFalse(set(s["fit"]) & set(s["inner_validation"]))
        self.assertFalse(set(s["training"]) & set(s["held"]))
        self.assertEqual(set(s["training"]),set(s["fit"])|set(s["inner_validation"]))

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError): fit_models(self.x*2,self.n,self.y,device="cpu")
        with self.assertRaises(ValueError): fit_bandwidth(torch.ones(3,768)/768**.5)


if __name__ == "__main__": unittest.main()
