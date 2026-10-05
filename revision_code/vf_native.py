"""V-F: decoupled rich-feature identity readout with native score propagation."""
import torch
from torch import nn
from torch.nn import functional as F


class RichReadout(nn.Module):
    def __init__(self, roi=4096, context=512, hidden=256, classes=151):
        super().__init__()
        self.roi, self.context = roi, context
        self.hidden = nn.Linear(roi + context, hidden)
        self.output = nn.Linear(hidden, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, logits):
        roi, context = features.split([self.roi, self.context], dim=-1)
        features = torch.cat([F.layer_norm(roi, (self.roi,)),
                              F.layer_norm(context, (self.context,))], -1)
        return logits + self.output(F.gelu(self.hidden(features)))


def objective_terms(logits, baseline, y, pairs, relation_log_confidence):
    valid = y >= 0
    if not valid.any():
        raise RuntimeError("No supervised proposals")
    weights = logits.new_ones(logits.shape[-1]); weights[0] = .25
    ce = F.cross_entropy(logits[valid], y[valid], weight=weights)
    kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(baseline, -1), reduction="batchmean")
    def scores(x):
        confidence = F.log_softmax(x, -1)[:, 1:].max(-1)[0]
        return confidence[pairs[:, 0]] + confidence[pairs[:, 1]] + relation_log_confidence
    if not len(pairs):
        raise RuntimeError("No native relation pairs")
    rank = F.kl_div(F.log_softmax(scores(logits), 0), F.softmax(scores(baseline), 0), reduction="sum")
    return dict(object_ce=ce, object_kl=kl, pair_score_kl=rank)


class ReadoutPatch:
    def __init__(self, model, head):
        relation = model.roi_heads.relation
        if not relation.object_cls_refine:
            raise RuntimeError("Updated identity would not reach postprocessing")
        self.head, self.enabled, self.capture = head, False, {}
        self.context = relation.predictor.context_layer
        if type(self.context).__name__ != "TransformerContext":
            raise RuntimeError("Only the audited Transformer is eligible")
        self.handles = [relation.predictor.register_forward_pre_hook(self._before),
                        self.context.context_obj.register_forward_hook(self._context),
                        relation.predictor.register_forward_hook(self._after),
                        relation.post_processor.register_forward_pre_hook(self._post)]

    def _before(self, module, inputs):
        if len(inputs[0]) != 1:
            raise RuntimeError("One image per batch required")
        proposal = inputs[0][0]
        self.capture = dict(roi=inputs[4].detach(), pairs=inputs[1][0].detach(),
                            proposal_boxes=proposal.bbox.detach(), size=proposal.size)
        if proposal.has_field("boxes_per_cls"):
            self.capture["boxes_per_cls"] = proposal.get_field("boxes_per_cls").detach()

    def _context(self, module, inputs, output):
        self.capture["context"] = output.detach()

    def _after(self, module, inputs, output):
        features = torch.cat([self.capture["roi"], self.capture["context"]], -1)
        baseline, relations = output[0][0], output[1][0]
        logits = self.head(features, baseline) if self.enabled else baseline
        self.capture.update(features=features.detach(), baseline=baseline.detach(), logits=logits.detach(),
                            relation_logits=relations.detach())
        # The semantic/context and predicate outputs are deliberately unchanged.
        # New object scores still reach native NMS, class-specific boxes and ranking.
        return ([logits], output[1]) + tuple(output[2:])

    def _post(self, module, inputs):
        if not torch.equal(inputs[0][1][0], self.capture["logits"]):
            raise RuntimeError("V-F identity output was bypassed")
        if not torch.equal(inputs[0][0][0], self.capture["relation_logits"]):
            raise RuntimeError("V-F unexpectedly changed predicate logits")
        self.capture["route_checked"] = True

    def close(self):
        for hook in self.handles:
            hook.remove()
