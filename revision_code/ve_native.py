"""Differentiable frozen-predictor replay and native-postprocessing adapter."""
import types

import torch
from torch import nn
from torch.nn import functional as F


class SharedContextAdapter(nn.Module):
    def __init__(self, width=512, hidden=64, scale=.1):
        super().__init__()
        self.down = nn.Linear(width, hidden)
        self.up = nn.Linear(hidden, width)
        self.scale = scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        change = self.up(F.gelu(self.down(F.layer_norm(x, (x.shape[-1],)))))
        rms = x.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return x + self.scale * rms * change.tanh()


class ContextAdapterPatch:
    def __init__(self, model, head):
        relation = model.roi_heads.relation
        if not relation.object_cls_refine:
            raise RuntimeError("V-E requires the classifier output to reach native postprocessing")
        self.context = relation.predictor.context_layer
        if type(self.context).__name__ != "TransformerContext":
            raise RuntimeError("V-E requires the audited TransformerContext")
        self.head, self.enabled, self.capture = head, False, None
        self.original_nms = self.context.nms_per_cls
        self.context.nms_per_cls = types.MethodType(self._nms, self.context)
        self.handles = [self.context.context_obj.register_forward_hook(self._context),
                        relation.predictor.register_forward_pre_hook(self._before),
                        self.context.out_obj.register_forward_hook(self._logits),
                        relation.post_processor.register_forward_pre_hook(self._post)]

    def _nms(self, context, obj_dists, boxes_per_cls, num_objs):
        # Discrete native NMS is intentionally nondifferentiable. Scores retain
        # their gradient for the object and consistency objectives.
        with torch.no_grad():
            return self.original_nms(obj_dists.detach(), boxes_per_cls, num_objs)

    def _context(self, module, inputs, output):
        if not self.enabled:
            return output
        return self.head(output)

    def _before(self, module, inputs):
        if len(inputs[0]) != 1:
            raise RuntimeError("V-E image batch must be one")
        proposal = inputs[0][0]
        self.capture = dict(proposal_boxes=proposal.bbox.detach(), size=proposal.size)

    def _logits(self, module, inputs, output):
        self.capture["logits"] = output.detach()

    def _post(self, module, inputs):
        actual = inputs[0][1][0]
        if not torch.equal(actual.detach(), self.capture["logits"]):
            raise RuntimeError("V-E object update did not reach final postprocessing")
        self.capture["final_route_checked"] = True

    def close(self):
        self.context.nms_per_cls = self.original_nms
        for h in self.handles:
            h.remove()


def pair_log_scores(objects, predicates, pairs):
    confidence = F.log_softmax(objects, -1)[:, 1:].max(-1)[0]
    rel = F.log_softmax(predicates, -1)[:, 1:].max(-1)[0]
    return confidence[pairs[:, 0]] + confidence[pairs[:, 1]] + rel


def objective_terms(student, teacher, targets, pairs, background_weight=.25):
    obj, rel = student[0][0], student[1][0]
    tobj, trel = teacher[0][0].detach(), teacher[1][0].detach()
    valid = targets >= 0
    if not valid.any() or len(pairs) == 0:
        raise RuntimeError("V-E training image has no eligible supervision or pairs")
    weights = obj.new_ones(obj.shape[-1])
    weights[0] = background_weight
    ce = F.cross_entropy(obj[valid], targets[valid], weight=weights)
    okl = F.kl_div(F.log_softmax(obj, -1), F.softmax(tobj, -1), reduction="batchmean")
    rkl = F.kl_div(F.log_softmax(rel, -1), F.softmax(trel, -1), reduction="batchmean")
    student_rank = pair_log_scores(obj, rel, pairs)
    teacher_rank = pair_log_scores(tobj, trel, pairs)
    skl = F.kl_div(F.log_softmax(student_rank, 0), F.softmax(teacher_rank, 0), reduction="sum")
    return dict(object_ce=ce, object_kl=okl, predicate_kl=rkl, pair_score_kl=skl)


def objective(terms, mode):
    if mode == "supervised":
        return terms["object_ce"]
    if mode != "relation_aware":
        raise ValueError(mode)
    return sum(terms.values())
