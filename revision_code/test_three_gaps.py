import unittest
import numpy as np
import torch

from r18_ovsgtr import score, bootstrap


class OpenVocabularyMetrics(unittest.TestCase):
    def test_perfect_and_novel_denominators(self):
        boxes=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[40.,0.,50.,10.]])
        labels=torch.tensor([1,2,3])
        rels=np.array([[0,1,1],[1,2,2]])
        probs=torch.zeros(2,51);probs[0,1]=1;probs[1,2]=1
        graph=dict(pred_boxes=boxes,pred_boxes_class=labels,pred_boxes_score=torch.ones(3),
                   all_node_pairs=torch.tensor(rels[:,:2]),all_relation=probs)
        result=score(graph,(boxes,labels,rels),{1,2},{2})
        self.assertEqual(result['all']['R']['50'],1.)
        self.assertEqual(result['novel_object_endpoint']['ground_truth_relations'],1)
        self.assertEqual(result['novel_predicate']['R']['50'],1.)
        self.assertEqual(result['identity_audit']['localized_pairs'],2)
        self.assertTrue(all(p['identity_correct'] for p in result['identity_audit']['pairs']))

    def test_absent_predictions_are_failures_not_skipped(self):
        gt=(torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.]]),torch.tensor([1,2]),np.array([[0,1,1]]))
        graph=dict(pred_boxes=torch.empty(0,4),pred_boxes_class=torch.empty(0,dtype=torch.long),
            pred_boxes_score=torch.empty(0),all_node_pairs=torch.empty(0,2,dtype=torch.long),all_relation=torch.empty(0,51))
        r=score(graph,gt,{1,2},{2})
        self.assertEqual(r['all']['R']['50'],0.)
        self.assertIsNone(r['novel_predicate']['R']['50'])
        self.assertEqual(r['identity_audit']['localized_objects'],0)

    def test_bootstrap_zero_and_undefined_are_distinct(self):
        self.assertEqual(bootstrap([0.,0.])['ci95'],[0.,0.])
        self.assertIsNone(bootstrap([])['mean'])


class PlainMotifsMetrics(unittest.TestCase):
    def test_native_recall_and_predicate_denominators(self):
        from r16_motifs import recall_row
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        boxes=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.]])
        gt=BoxList(boxes,(40,20),mode='xyxy')
        gt.add_field('labels',torch.tensor([1,2]))
        gt.add_field('relation_tuple',torch.tensor([[0,1,1]]))
        pred=BoxList(boxes,(40,20),mode='xyxy')
        pred.add_field('pred_labels',torch.tensor([1,2]))
        pred.add_field('pred_scores',torch.ones(2))
        pred.add_field('rel_pair_idxs',torch.tensor([[0,1]]))
        probs=torch.zeros(1,51);probs[0,1]=1
        pred.add_field('pred_rel_scores',probs)
        result=recall_row(pred,gt,'sgdet')
        self.assertEqual(result['R'],[1.]*6)
        self.assertEqual(result['class_recall'][0][0],1.)
        self.assertIsNone(result['class_recall'][0][1])


if __name__=='__main__':unittest.main()
