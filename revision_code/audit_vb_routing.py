"""Check whether the old V-B residual actually reached final object outputs."""
import json
import torch
from common import ROOT, ensure_storage, atomic_json
from native_runtime import load_model, dataset, infer, image_id
from vb_native import ContextPatch, compare_predictions
from vb_protocol import ResidualHead


def main():
    ensure_storage();torch.set_num_threads(4)
    out=ROOT/"results/VB_routing_audit"
    model,cfg,transform,provenance=load_model("tde_motifs","sgcls",out)
    ds=dataset(cfg,"val");patch=ContextPatch(model)
    post=[]
    handle=model.roi_heads["relation"].post_processor.register_forward_pre_hook(
        lambda module,inputs:post.append(inputs[0][1][0].detach().cpu().clone()))
    pred,_=infer(model,cfg,transform,ds,0)
    clean_context=patch.capture["logits"].cpu().clone();clean_final=post[-1]
    patch.enabled=True;patch.head=ResidualHead().cuda().eval()
    with torch.no_grad():patch.head.linear.bias[150]=20
    changed,_=infer(model,cfg,transform,ds,0)
    result=dict(status="complete",image_id=image_id(ds,0),model=provenance,
        object_classification_refine=bool(model.roi_heads["relation"].object_cls_refine),
        context_logits_max_change=float((patch.capture["logits"].cpu()-clean_context).abs().max()),
        final_object_logits_max_change=float((post[-1]-clean_final).abs().max()),
        final_labels_changed=int((pred.get_field("pred_labels")!=changed.get_field("pred_labels")).sum()),
        final_object_scores_max_change=float((pred.get_field("pred_scores")-changed.get_field("pred_scores")).abs().max()),
        protocol="Synthetic development-only routing test; not a mitigation accuracy result")
    result["old_vb_final_object_route_connected"]=result["final_object_logits_max_change"]>0
    atomic_json(out/"summary.json",result);print(json.dumps(result,indent=2),flush=True)
    handle.remove();patch.close()


if __name__=="__main__":main()
