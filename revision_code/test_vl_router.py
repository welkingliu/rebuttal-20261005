import unittest
import numpy as np
import torch
from vl_router import box_iou_aligned, predict_logistic, select_action


class RouterRules(unittest.TestCase):
    def test_iou(self):
        a=torch.tensor([[0.,0.,9.,9.],[0.,0.,9.,9.]])
        b=torch.tensor([[0.,0.,9.,9.],[20.,20.,29.,29.]])
        self.assertTrue(torch.equal(box_iou_aligned(a,b),torch.tensor([1.,0.])))

    def test_threshold_and_single_action(self):
        effect=dict(mean=[0.],scale=[1.],weight=[0.],bias=0.)
        safety=dict(mean=[0.],scale=[1.],weight=[1.],bias=0.)
        model=dict(effect=effect,safety=safety,probability_min=.6)
        choice, rows=select_action(model,np.array([[0.],[2.],[2.]]),[9,3,1])
        self.assertEqual(choice,1)
        self.assertEqual(len(rows),3)
        self.assertEqual(select_action(model,np.array([[0.]]),[9])[0],None)

    def test_stable_sigmoid(self):
        m=dict(mean=[0.],scale=[1.],weight=[1.],bias=0.)
        result=predict_logistic(m,np.array([[-10000.],[10000.]]))
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(result[0],1e-10)
        self.assertGreater(result[1],.999)


if __name__ == "__main__": unittest.main()
