"""Read-only native model integration for the V-B residual object head."""
import types

import numpy as np
import torch

from vb_protocol import proposal_targets, KS


class ContextPatch:
    def __init__(self, model):
        self.context=model.roi_heads["relation"].predictor.context_layer
        if type(self.context).__name__!="LSTMContext":
            raise RuntimeError("V-B requires native Motifs object/edge context")
        self.original=self.context.obj_ctx
        self.head=None
        self.temperature=1.0
        self.enabled=False
        self.capture=None
        self.average_calls=0
        self.context.obj_ctx=types.MethodType(self._call,self.context)

    def _call(self, context, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        out=self.original(obj_feats,proposals,obj_labels,boxes_per_cls,ctx_average=ctx_average)
        if ctx_average:
            self.average_calls+=1
            return out
        logits,labels,features,perm,inv_perm,ls=out
        updated=logits
        if self.enabled:
            updated=self.head(features,logits) if self.head is not None else logits/self.temperature
            if boxes_per_cls is not None:
                from pysgg.modeling.roi_heads.relation_head.utils_relation import obj_prediction_nms
                labels=obj_prediction_nms(boxes_per_cls[perm],updated[perm],context.decoder_rnn.nms_thresh)[inv_perm]
            else:
                labels=updated[:,1:].argmax(1)+1
        if len(proposals)!=1:
            raise RuntimeError("V-B image batch must be one")
        self.capture=dict(features=features.detach(),base_logits=logits.detach(),
                          logits=updated.detach(),labels=labels.detach(),
                          proposal_boxes=proposals[0].bbox.detach(),size=proposals[0].size)
        return updated,labels,features,perm,inv_perm,ls

    def close(self):
        self.context.obj_ctx=self.original


def targets(captured, gt, task):
    gt=gt.resize(captured["size"])
    return proposal_targets(captured["proposal_boxes"].cpu().numpy(),gt.bbox.cpu().numpy(),
                            gt.get_field("labels").cpu().numpy(),task)


def compare_predictions(a,b):
    tensors=[(a.bbox,b.bbox)]
    for field in ["pred_labels","pred_scores","rel_pair_idxs","pred_rel_scores"]:
        tensors.append((a.get_field(field),b.get_field(field)))
    errors=[]
    for x,y in tensors:
        if x.shape!=y.shape:
            raise RuntimeError("Zero-update changed native prediction shape")
        if x.dtype in (torch.int32,torch.int64):
            if not torch.equal(x,y):
                raise RuntimeError("Zero-update changed native discrete predictions")
        elif not torch.allclose(x,y,atol=1e-5,rtol=1e-5):
            raise RuntimeError("Zero-update changed native floating predictions")
        errors.append(float((x.float()-y.float()).abs().max()) if x.numel() else 0.)
    return max(errors)


class OfficialMetrics:
    def __init__(self,task):
        from pysgg.data.datasets.evaluation.vg.sgg_eval import SGRecall
        self.task=task
        self.result={}
        self.recall=SGRecall(self.result)
        self.recall.register_container(task)
        self.result[task+"_recall"]={k:[] for k in KS}

    def row(self,iid,pred,gt,captured,target):
        from functools import reduce
        pred=pred.resize(gt.size).to("cpu")
        rel=gt.get_field("relation_tuple").cpu().numpy().astype(int)
        local=dict(pred_rel_inds=pred.get_field("rel_pair_idxs").numpy(),
                   rel_scores=pred.get_field("pred_rel_scores").numpy(),
                   gt_rels=rel,gt_classes=gt.get_field("labels").cpu().numpy(),
                   gt_boxes=gt.bbox.cpu().numpy(),pred_boxes=pred.bbox.numpy(),
                   pred_classes=pred.get_field("pred_labels").numpy(),
                   obj_scores=pred.get_field("pred_scores").numpy())
        local["pred_rel_inds"]=local["pred_rel_inds"][:100]
        local["rel_scores"]=local["rel_scores"][:100]
        if len(local["pred_rel_inds"]):
            local=self.recall.calculate_recall({"iou_thres":.5},local,self.task)
        else:
            local["pred_to_gt"]=[]
        counts=np.bincount(rel[:,2],minlength=51)[1:]
        rr=[]; mr=[]
        for k in KS:
            hits=reduce(np.union1d,local["pred_to_gt"][:k],np.array([],dtype=int)).astype(int)
            hit=np.bincount(rel[hits,2],minlength=51)[1:]
            rr.append(float(len(hits)/len(rel)))
            mr.append([float(h/c) if c else None for h,c in zip(hit,counts)])
        logits=captured["logits"].float().cpu()
        target=torch.from_numpy(target)
        positive=target>0
        probs=logits.softmax(-1)
        conf,labels=probs[:,1:].max(-1);labels+=1
        correct=labels[positive]==target[positive]
        pp=probs[positive]; yy=target[positive]
        bins=np.zeros((15,3))
        c=conf[positive].numpy();h=correct.numpy()
        for j in range(15):
            sel=np.minimum((c*15).astype(int),14)==j
            bins[j]=[sel.sum(),c[sel].sum(),h[sel].sum()]
        nll=float(torch.nn.functional.cross_entropy(logits[positive],yy,reduction="sum")) if positive.any() else 0.
        onehot=torch.nn.functional.one_hot(yy,151).float()
        return dict(image_id=iid,recalls=rr,class_recalls=mr,
                    positive_objects=int(positive.sum()),positive_correct=int(correct.sum()),
                    post_nms_correct=int((pred.get_field("pred_labels")[positive]==yy).sum()),
                    ignored_proposals=int((target<0).sum()),background_proposals=int((target==0).sum()),
                    object_class_correct=np.bincount(yy[correct].numpy(),minlength=151)[1:].tolist(),
                    object_class_count=np.bincount(yy.numpy(),minlength=151)[1:].tolist(),
                    total_proposals=len(target),nll_sum=nll,brier_sum=float((pp-onehot).square().sum()),
                    calibration_bins=bins.tolist())
