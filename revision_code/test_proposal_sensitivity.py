import unittest

import numpy as np

from r19_proposal_sensitivity import unique_support


class SupportTests(unittest.TestCase):
    def test_duplicate_proposals_do_not_duplicate_gt(self):
        p,g=unique_support(np.array([[.9,.0],[.8,.0],[.0,.7]]))
        self.assertEqual(p.tolist(),[0,2]);self.assertEqual(g.tolist(),[0,1])

    def test_threshold_cardinality_precedes_iou_sum(self):
        p,g=unique_support(np.array([[1.,.5],[.5,.49]]))
        self.assertEqual(len(p),2);self.assertEqual(g.tolist(),[1,0])

    def test_empty_support(self):
        p,g=unique_support(np.zeros((2,0)))
        self.assertEqual(len(p),0);self.assertEqual(len(g),0)


if __name__=="__main__":unittest.main()
