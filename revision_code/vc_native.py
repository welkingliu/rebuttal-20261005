"""Residual update on the proposal logits actually used by native TDE output."""
import types

import torch

from vb_native import OfficialMetrics, compare_predictions, targets


class ProposalPatch:
    def __init__(self, model):
        relation = model.roi_heads["relation"]
        if relation.object_cls_refine:
            raise RuntimeError("V-C requires the unchanged native proposal-logit route")
        self.context = relation.predictor.context_layer
        if type(self.context).__name__ != "LSTMContext":
            raise RuntimeError("V-C requires native Motifs context")
        self.original = self.context.forward
        self.head = None
        self.temperature = 1.0
        self.enabled = False
        self.capture = None
        self.average_calls = 0
        self.final_checks = 0
        self.original_logits = None
        self.context.forward = types.MethodType(self._context, self.context)
        self.hooks = [relation.predictor.register_forward_pre_hook(self._before),
                      relation.post_processor.register_forward_pre_hook(self._post)]

    def _before(self, module, inputs):
        proposals, features = inputs[0], inputs[4]
        if len(proposals) != 1 or features.ndim != 2:
            raise RuntimeError("V-C expects one image and pooled ROI features")
        base = proposals[0].get_field("predict_logits")
        self.original_logits = base
        updated = base
        if self.enabled:
            updated = self.head(features, base) if self.head is not None else base / self.temperature
        if not torch.isfinite(updated).all():
            raise RuntimeError("Nonfinite V-C object logits")
        proposals[0].add_field("predict_logits", updated)
        self.capture = dict(features=features.detach(), base_logits=base.detach(),
                            logits=updated.detach(), proposal_boxes=proposals[0].bbox.detach(),
                            size=proposals[0].size)

    def _context(self, context, x, proposals, rel_pair_idxs, logger=None,
                 all_average=False, ctx_average=False):
        if not (ctx_average or all_average):
            return self.original(x, proposals, rel_pair_idxs, logger,
                                 all_average=all_average, ctx_average=ctx_average)
        self.average_calls += 1
        current = proposals[0].get_field("predict_logits")
        proposals[0].add_field("predict_logits", self.original_logits)
        try:
            return self.original(x, proposals, rel_pair_idxs, logger,
                                 all_average=all_average, ctx_average=ctx_average)
        finally:
            proposals[0].add_field("predict_logits", current)

    def _post(self, module, inputs):
        actual = inputs[0][1][0]
        if not torch.equal(actual, self.capture["logits"]):
            raise RuntimeError("Updated object logits were replaced before final postprocessing")
        self.capture["final_logits"] = actual.detach().clone()
        self.final_checks += 1

    def close(self):
        self.context.forward = self.original
        for hook in self.hooks:
            hook.remove()
