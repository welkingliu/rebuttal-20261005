import unittest
import numpy as np
import torch
from vm_localization_router import localization_features, choose_localized


class LocalizationRules(unittest.TestCase):
    def test_input_contract(self):
        f=torch.ones(2,768);b=torch.zeros(2,151);box=torch.tensor([[0.,0.,9.,9.],[0.,0.,4.,4.]])
        x=localization_features(f,b,box,(10,10))
        self.assertEqual(x.shape,(2,776))
        self.assertAlmostEqual(x[0,772],1.)
        self.assertAlmostEqual(x[1,772],.25)

    def test_localization_restricts_action(self):
        model=dict(mean=[0.],scale=[1.],weight=[1.],bias=0.)
        choices=[dict(proposal=0,safety_probability=.9,expected_signed_utility=.9),
                 dict(proposal=1,safety_probability=.8,expected_signed_utility=.4)]
        chosen,_,_=choose_localized(model,np.array([[-2.],[2.]]),[0,1],choices)
        self.assertEqual(chosen,1)
        with self.assertRaises(ValueError):choose_localized(model,np.ones((2,1)),[1,0],choices)


if __name__=="__main__":unittest.main()
