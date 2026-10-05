"""Visual re-observation and prediction-only equal-budget allocation."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vp_math import fusion_logits

PRIMARY = "diagnostic_budget"
ARMS = ["native", "dino_all", "siglip_tight_all", "siglip_dual_all", PRIMARY,
        "uncertainty_budget", "random_budget"]
EXPERTS = ["dino", "siglip_tight", "siglip_dual"]


def crop_bounds(box, size, scale):
    x0,y0,x1,y1 = [float(v) for v in box]; w,h=size
    if not all(math.isfinite(v) for v in [x0,y0,x1,y1]) or x1<x0 or y1<y0:
        raise ValueError("Invalid predicted box")
    cx,cy=(x0+x1+1)/2,(y0+y1+1)/2
    bw,bh=(x1-x0+1)*scale,(y1-y0+1)*scale
    out=(max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),
         min(w,math.ceil(cx+bw/2)),min(h,math.ceil(cy+bh/2)))
    if out[2]<=out[0] or out[3]<=out[1]: raise ValueError("Empty detector crop")
    return out


def view_features(bundle, name):
    if name=="dino": return bundle["features"]
    if name=="siglip_tight": return bundle["siglip_tight"]
    if name=="siglip_dual": return torch.cat([bundle["siglip_tight"],bundle["siglip_context"]],-1)/2**.5
    raise ValueError("Unknown expert")


def fit_expert(x,y,validation=None,epochs=None,smoke=False,callback=None,device="cuda"):
    torch.manual_seed(17); model=nn.Linear(x.shape[1],150).to(device)
    nn.init.zeros_(model.weight); nn.init.zeros_(model.bias)
    keep=y>0; x,y=x[keep].to(device),y[keep].to(device)-1
    if not len(y): raise ValueError("No positive fitting proposals")
    if validation is not None:
        vx,vy=validation; valid=vy>0; vx,vy=vx[valid].to(device),vy[valid].to(device)-1
        if not len(vy): raise ValueError("No positive inner proposals")
    elif epochs is None: raise ValueError("Refit epoch not specified")
    limit=epochs if epochs is not None else (3 if smoke else 40)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    gen=torch.Generator().manual_seed(17); best=float("inf"); chosen=0; bad=0; history=[]
    for epoch in range(1,limit+1):
        total=0.; order=torch.randperm(len(y),generator=gen).to(device)
        for ix in order.split(256):
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(x[ix]),y[ix])
            if not torch.isfinite(loss): raise RuntimeError("Invalid expert loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            total+=float(loss.detach())*len(ix)
        row=dict(epoch=epoch,train_ce=total/len(y))
        if validation is not None:
            with torch.no_grad():
                value=sum(float(F.cross_entropy(model(vx[i:i+512]),vy[i:i+512],reduction="sum"))
                          for i in range(0,len(vy),512))/len(vy)
            row["inner_ce"]=value
            if epoch >= (1 if smoke else 3):
                if value<best-1e-6: best,chosen,bad=value,epoch,0
                else: bad+=1
        history.append(row)
        if callback: callback(row,limit)
        if validation is not None and epoch>=3 and bad>=5: break
    return {k:v.detach().cpu() for k,v in model.state_dict().items()},dict(
        selected_epoch=chosen if validation is not None else epochs,history=history,
        fitting_positive=len(y),fitting_classes=int(y.unique().numel()))


def risk_features(native,boxes,size):
    p=native[:,1:].softmax(-1); top=p.topk(2,dim=-1)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)/math.log(150)
    widths=(boxes[:,2]-boxes[:,0]+1).clamp_min(1); heights=(boxes[:,3]-boxes[:,1]+1).clamp_min(1)
    area=(widths*heights/(size[0]*size[1])).clamp(1e-6,1.)
    continuous=torch.stack([top.values[:,0],top.values[:,0]-top.values[:,1],entropy,
        native.softmax(-1)[:,0],area.log(),(widths/heights).log().clamp(-5,5)],-1)
    return torch.cat([continuous,F.one_hot(top.indices[:,0],150).float()],-1)


def fit_risk(x,native,y,smoke=False,device="cuda"):
    keep=y>=0; target=((y>0)&(native[:,1:].argmax(-1)+1!=y)).float()[keep]
    x=x[keep].to(device); target=target.to(device)
    mean=x.mean(0); scale=x.std(0,unbiased=False).clamp_min(.01)
    z=(x-mean)/scale
    torch.manual_seed(17); model=nn.Linear(x.shape[1],1).to(device)
    nn.init.zeros_(model.weight); nn.init.zeros_(model.bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    gen=torch.Generator().manual_seed(17); history=[]
    for epoch in range(2 if smoke else 20):
        total=0.
        for ix in torch.randperm(len(z),generator=gen).to(device).split(1024):
            opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(z[ix]).flatten(),target[ix])
            if not torch.isfinite(loss): raise RuntimeError("Invalid risk loss")
            loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
        history.append(total/len(z))
    return dict(weight=model.weight.detach().cpu(),bias=model.bias.detach().cpu(),
        mean=mean.cpu(),scale=scale.cpu(),history=history,positive_events=int(target.sum()),fitting_proposals=len(target))


def endpoint_impact(native,pairs,relations):
    impact=native.new_zeros(len(native))
    if not len(pairs): return impact
    objects=native.softmax(-1)[:,1:].max(-1)[0]
    score=relations.softmax(-1)[:,1:].max(-1)[0]*objects[pairs[:,0]]*objects[pairs[:,1]]
    order=np.argsort(-score.detach().cpu().numpy(),kind="stable")[:50]
    ix=torch.as_tensor(order.copy(),device=native.device)
    for side in [0,1]: impact.index_add_(0,pairs[ix,side],score[ix])
    return impact/impact.max().clamp_min(1e-12)


def allocation(name,native,risk_x,pairs,relations,image_id,risk):
    count=len(native); budget=min(count,int(math.ceil(.2*count)))
    if name==PRIMARY:
        z=(risk_x-risk["mean"].to(native.device))/risk["scale"].to(native.device)
        probability=F.linear(z,risk["weight"].to(native.device),risk["bias"].to(native.device)).flatten().sigmoid()
        priority=probability*(1+endpoint_impact(native,pairs,relations))
        order=np.argsort(-priority.detach().cpu().numpy(),kind="stable")
    elif name=="uncertainty_budget":
        priority=1-native[:,1:].softmax(-1).max(-1)[0]
        order=np.argsort(-priority.detach().cpu().numpy(),kind="stable")
    elif name=="random_budget":
        seed=int.from_bytes(hashlib.sha256(("vu_random17:"+str(image_id)).encode()).digest()[:8],"little")
        order=np.random.default_rng(seed).permutation(count)
    else: raise ValueError("Unknown budget policy")
    selected=torch.zeros(count,dtype=torch.bool,device=native.device)
    selected[torch.as_tensor(order[:budget].copy(),device=native.device)]=True
    return selected


def apply_expert(native,features,state,selected):
    if selected.shape!=(len(native),): raise ValueError("Selection shape mismatch")
    result=native.clone()
    if selected.any():
        q=F.linear(features[selected],state["weight"].to(features.device),state["bias"].to(features.device)).softmax(-1)
        result[selected]=fusion_logits(native[selected],q,alpha=.5)
    return result
