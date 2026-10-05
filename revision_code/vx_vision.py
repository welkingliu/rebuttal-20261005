"""SigLIP2 frozen-prefix caching and real last-two-block region adaptation."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from common import ensure_storage, sha256, canonical_path, atomic_json
from evidence_completion import read
from vx_common import (OUT, CACHE, WEIGHTS, HERE, SPEC, ARMS, PRIMARY,
                       load, save, progress, finish, checked_done)
from vx_math import objective, tensor_meta
from vq_math import choose_epoch

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def step_layer(layer, hidden):
    value = layer(hidden, attention_mask=None)
    return value[0] if isinstance(value, tuple) else value


def load_vision(p):
    from transformers import AutoModel, AutoImageProcessor
    for path, digest in p["visual_assets"].items():
        if sha256(path) != digest:
            raise RuntimeError("SigLIP assets changed")
    parent = AutoModel.from_pretrained(p["model_dir"], local_files_only=True)
    model = getattr(parent, "vision_model", None)
    if model is None or len(model.encoder.layers) != 12:
        raise RuntimeError("Expected SigLIP2 FixRes 12-block vision model")
    model = model.float().cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(p["model_dir"], local_files_only=True, use_fast=False)
    return model, processor


class VisualResidual(nn.Module):
    def __init__(self, vision, adaptive):
        super().__init__()
        self.layers = copy.deepcopy(vision.encoder.layers[-SPEC["tail_blocks"]:])
        self.norm = copy.deepcopy(vision.post_layernorm).requires_grad_(False)
        self.pool = copy.deepcopy(vision.head).requires_grad_(False)
        self.layers.requires_grad_(adaptive)
        self.readout = nn.Linear(768+151, 151, device="cuda")
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.adaptive = adaptive

    def encode(self, tokens):
        for layer in self.layers:
            tokens = step_layer(layer, tokens)
        return F.normalize(self.pool(self.norm(tokens)).float(), dim=-1)

    def forward(self, tokens, native):
        # Deliberately no GT labels, endpoint masks or IoU/matching arguments.
        if tokens.shape[0] != native.shape[0] or native.shape[1] != 151:
            raise ValueError("Unaligned proposal order/ontology")
        parts = []
        for chunk in tokens.split(SPEC["crop_chunk"]):
            if self.training and self.adaptive and torch.is_grad_enabled():
                parts.append(checkpoint(self.encode, chunk, use_reentrant=False))
            else:
                parts.append(self.encode(chunk))
        features = torch.cat(parts)
        design = torch.cat([features, F.normalize(native.detach().softmax(-1), dim=-1)], -1)
        return native + self.readout(design)


def extract(p, reg, lane, shard):
    from PIL import Image
    from vu_math import crop_bounds
    stage = OUT/lane/("extract%d" % shard)
    if checked_done(stage, reg):
        return
    model, processor = load_vision(p)
    rows = p["inputs"][shard::2]
    manifest, parity = {}, []
    start = time.monotonic()
    import shutil
    if shutil.disk_usage(CACHE.parent).free < 200*1024**3:
        raise RuntimeError("At least 200 GiB free on experiment disk required for float32 prefix cache")
    for n, entry in enumerate(rows, 1):
        iid = entry["image_id"]
        if sha256(entry["raw"]) != entry["raw_sha256"]:
            raise RuntimeError("Raw drift")
        raw = torch.load(entry["raw"], map_location="cpu", weights_only=False)
        image_path = canonical_path(raw["image"])
        image_hash = sha256(image_path)
        path = CACHE/lane/"prefix"/(iid+".pt")
        if path.exists():
            value = torch.load(str(path), map_location="cpu", weights_only=False)
            if (value["protocol_sha256"] != reg or value["image_sha256"] != image_hash
                    or value["raw_sha256"] != entry["raw_sha256"]
                    or not torch.equal(value["crop_boxes"], raw["crop_boxes"])):
                raise RuntimeError("Resumed prefix changed")
            if n == 1:
                if not value.get("prefix_parity"):
                    raise RuntimeError("Resumed first prefix lacks saved forward parity")
                parity.extend(value["prefix_parity"])
        else:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if tuple(image.size) != tuple(raw["image_size"]):
                raise RuntimeError("Image coordinate mismatch")
            crops = [image.crop(crop_bounds(b.tolist(), image.size, 1.)) for b in raw["crop_boxes"]]
            tokens = []
            with torch.no_grad():
                for offset in range(0, len(crops), SPEC["crop_chunk"]):
                    pixels = processor(images=crops[offset:offset+SPEC["crop_chunk"]], return_tensors="pt")["pixel_values"].cuda()
                    hidden = model.embeddings(pixels)
                    for layer in model.encoder.layers[:-SPEC["tail_blocks"]]:
                        hidden = step_layer(layer, hidden)
                    if hidden.shape[1:] != (196, 768) or not torch.isfinite(hidden).all():
                        raise RuntimeError("Invalid prefix tokens")
                    if n == 1 and offset == 0:
                        current = hidden
                        for layer in model.encoder.layers[-SPEC["tail_blocks"]:]:
                            current = step_layer(layer, current)
                        manual = F.normalize(model.head(model.post_layernorm(current)).float(), dim=-1)
                        direct = F.normalize(model(pixel_values=pixels).pooler_output.float(), dim=-1)
                        error = float((manual-direct).abs().max())
                        if not torch.allclose(manual, direct, atol=2e-5, rtol=2e-4):
                            raise RuntimeError("Frozen-prefix/full-forward parity failed")
                        single = F.normalize(model(pixel_values=pixels[:1]).pooler_output.float(), dim=-1)
                        if not torch.allclose(single, direct[:1], atol=2e-5, rtol=2e-4):
                            raise RuntimeError("Crop batch parity failed")
                        parity.append(dict(image_id=iid, split_forward_max_error=error,
                                           batch_max_error=float((single-direct[:1]).abs().max())))
                    tokens.append(hidden.float().cpu())
            value = dict(image_id=iid, protocol_sha256=reg, raw_sha256=entry["raw_sha256"],
                         image_sha256=image_hash, crop_boxes=raw["crop_boxes"], tokens=torch.cat(tokens),
                         dtype="float32", prefix_frozen_blocks=10,
                         prefix_parity=parity if n == 1 else [])
            save(path, value)
        manifest[iid] = dict(sha256=sha256(path), bytes=path.stat().st_size, image_sha256=image_hash)
        if n % 10 == 0 or n == len(rows):
            progress(stage, "frozen_prefix_detected_box_crops", n, len(rows), start)
        if STOP:
            raise RuntimeError("Stopped at completed prefix image boundary; caches retained")
    finish(stage, reg, len(rows), manifest=manifest, parity=parity, gt_used_by_encoder=False,
           bytes=sum(v["bytes"] for v in manifest.values()))


def verified_manifest(p, reg, lane):
    result = {}
    for shard in [0, 1]:
        stage = OUT/lane/("extract%d" % shard)
        if not checked_done(stage, reg):
            raise RuntimeError("Prefix shard incomplete")
        info = read(stage/"summary.json")
        if not info["parity"]:
            raise RuntimeError("Missing prefix parity audit")
        result.update(info["manifest"])
    if set(result) != set(p["image_ids"]):
        raise RuntimeError("Prefix coverage mismatch")
    return result


class Records:
    def __init__(self, p, reg, lane):
        self.reg, self.lane = reg, lane
        self.prefix = verified_manifest(p, reg, lane)
        if not checked_done(OUT/lane/"prepare", reg):
            raise RuntimeError("Native audit incomplete")
        self.meta = read(OUT/lane/"prepare/summary.json")["manifest"]
        self.verified = set()
        self.metadata = {}

    def get(self, iid):
        path = CACHE/self.lane/"prefix"/(iid+".pt")
        meta_path = CACHE/self.lane/"meta"/(iid+".pt")
        if iid not in self.verified:
            if sha256(path) != self.prefix[iid]["sha256"] or sha256(meta_path) != self.meta[iid]["meta_sha256"]:
                raise RuntimeError("Input cache changed: "+iid)
            self.verified.add(iid)
        value = torch.load(str(path), map_location="cpu", weights_only=False)
        if iid not in self.metadata:
            self.metadata[iid] = torch.load(str(meta_path), map_location="cpu", weights_only=False)
        meta = self.metadata[iid]
        if value["protocol_sha256"] != self.reg or meta["protocol_sha256"] != self.reg or value["image_id"] != iid:
            raise RuntimeError("Wrong cache registration")
        return value["tokens"].cuda(), tensor_meta(meta, "cuda")


def audit(vision, records, ids, stage, reg):
    candidate = None
    for iid in ids:
        tokens, meta = records.get(iid)
        if meta["protected"].any():
            candidate = iid, tokens, meta
            break
    if candidate is None:
        raise RuntimeError("No fit-image protected relation for gradient audit")
    iid, tokens, meta = candidate
    torch.manual_seed(SPEC["seed"])
    model = VisualResidual(vision, True).cuda().eval()
    with torch.no_grad():
        if not torch.equal(model(tokens, meta["native"]), meta["native"]):
            raise RuntimeError("Initial residual is not exactly native")
    # Zero readout blocks feature gradients initially; exercise the actual warm-up.
    initial = {k:v.detach().clone() for k,v in model.layers.state_dict().items()}
    opt = torch.optim.AdamW([dict(params=model.readout.parameters(), lr=SPEC["head_lr"]),
                            dict(params=model.layers.parameters(), lr=SPEC["tail_lr"])], weight_decay=SPEC["weight_decay"])
    first_norm, second_norm = None, None
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        objective(model(tokens, meta["native"]), meta, PRIMARY)["loss"].backward()
        norm = float(torch.sqrt(sum(x.grad.square().sum() for x in model.layers.parameters() if x.grad is not None)))
        if step == 0:
            first_norm = norm
        else:
            second_norm = norm
        nn.utils.clip_grad_norm_(model.parameters(), SPEC["clip"])
        opt.step()
    if not second_norm > 0:
        raise RuntimeError("No actual visual-encoder gradient after readout warm-up")
    changes = {str(i):max(float((v-initial[k]).abs().max()) for k,v in model.layers.state_dict().items() if k.startswith(str(i)+".")) for i in range(2)}
    if not all(v > 0 for v in changes.values()):
        raise RuntimeError("One of the two visual blocks did not update")
    with torch.no_grad():
        model.readout.weight.normal_(0, .02)
        model.readout.bias[1:] -= .5
    losses = objective(model(tokens, meta["native"]), meta, PRIMARY)
    norms = {}
    for key in ["object_ce", "native_kl", "relation_protection"]:
        norms[key] = {}
        for i, block in enumerate(model.layers):
            g = torch.autograd.grad(losses[key], tuple(block.parameters()), retain_graph=True)
            norms[key][str(i)] = float(torch.sqrt(sum(x.square().sum() for x in g)))
        if not all(v > 0 and torch.isfinite(torch.tensor(v)) for v in norms[key].values()):
            raise RuntimeError("A promised loss failed to reach each visual block")
    if any(x.requires_grad or x.grad is not None for x in list(model.pool.parameters())+list(model.norm.parameters())+list(vision.parameters())):
        raise RuntimeError("Frozen prefix/pool/norm unexpectedly trained")
    saved = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    temp = CACHE/stage.relative_to(OUT)/"gradient_roundtrip.pt"
    save(temp, saved)
    clone = VisualResidual(vision, True).cuda().eval()
    clone.load_state_dict(torch.load(str(temp), map_location="cpu", weights_only=False), strict=True)
    with torch.no_grad():
        before, after = model(tokens, meta["native"]), clone(tokens, meta["native"])
        if not torch.equal(before, after):
            raise RuntimeError("Serialized model changed output")
    value = dict(protocol_sha256=reg, image_id=iid, fit_image_only=True,
                 first_zero_head_tail_gradient=first_norm, after_warmup_tail_gradient=second_norm,
                 actual_tail_block_changes=changes, loss_to_each_tail_gradient_norm=norms,
                 synthetic_probe_discarded=True, frozen_prefix_pool_and_norm=True,
                 state_roundtrip_exact=True, all_checks_passed=True)
    atomic_json(stage/"gradient_audit.json", value)
    return value


def emit(p, reg, lane, arm, epoch, role, model, records, checkpoint_path):
    stage = OUT/lane/arm/("%s_epoch%d" % (role, epoch))
    path = CACHE/lane/arm/("%s_epoch%d.pt" % (role, epoch))
    if checked_done(stage, reg):
        done = read(stage/"summary.json")
        if done["checkpoint_sha256"] != sha256(checkpoint_path) or done["prediction_sha256"] != sha256(path):
            raise RuntimeError("Completed prediction or checkpoint drift")
        return done
    model.eval()
    logits = {}
    start = time.monotonic()
    with torch.no_grad():
        for n, iid in enumerate(p["split"][role], 1):
            tokens, meta = records.get(iid)
            updated = model(tokens, meta["native"])
            if not torch.isfinite(updated).all():
                raise RuntimeError("Nonfinite exported logits")
            if epoch == 0 and not torch.equal(updated, meta["native"]):
                raise RuntimeError("Epoch0 drift")
            logits[iid] = updated.cpu()
            if n % 20 == 0 or n == len(p["split"][role]):
                progress(stage, "export_updated_object_logits", n, len(p["split"][role]), start)
    save(path, dict(protocol_sha256=reg, image_ids=p["split"][role], role=role,
                   arm=arm, epoch=epoch, checkpoint_sha256=sha256(checkpoint_path), logits=logits))
    command = [os.environ["SGG_NATIVE_PYTHON"], str(HERE/"vx_native.py"), "--stage", "evaluate", "--prediction_file", str(path)]
    if p["smoke"]:
        command.append("--smoke")
    subprocess.run(command, check=True, timeout=3600)
    if not checked_done(stage, reg):
        raise RuntimeError("Official postprocessing incomplete")
    return read(stage/"summary.json")


def train(p, reg, lane, arm):
    stage = OUT/lane/arm/"train"
    if checked_done(stage, reg):
        return
    vision, _ = load_vision(p)
    records = Records(p, reg, lane)
    audit(vision, records, p["split"]["fit"], stage, reg)
    torch.manual_seed(SPEC["seed"])
    model = VisualResidual(vision, arm != "frozen_head").cuda()
    original = {k:v.detach().cpu().clone() for k,v in model.layers.state_dict().items()}
    groups = [dict(params=model.readout.parameters(), lr=SPEC["head_lr"])]
    if arm != "frozen_head":
        groups.append(dict(params=model.layers.parameters(), lr=SPEC["tail_lr"]))
    opt = torch.optim.AdamW(groups, weight_decay=SPEC["weight_decay"])
    # Releasing the untrained prefix reduces memory; the saved cache replaces only frozen computation.
    del vision
    generator = torch.Generator().manual_seed(SPEC["seed"])
    candidates, history = [], []
    path_root = WEIGHTS/lane/arm
    resume = path_root/"resume.pt"
    first_epoch, offset, order, sums = 1, 0, None, None
    limit = 2 if p["smoke"] else SPEC["epochs"]
    if resume.exists():
        old = torch.load(str(resume), map_location="cpu", weights_only=False)
        if old["protocol_sha256"] != reg:
            raise RuntimeError("Resume code/protocol changed")
        model.load_state_dict(old["model"], strict=True)
        opt.load_state_dict(old["optimizer"])
        generator.set_state(old["generator"])
        first_epoch, offset, order = old["epoch"], old["offset"], old["order"]
        candidates, history, sums = old["candidates"], old["history"], old["sums"]
    else:
        zero = path_root/"epoch0.pt"
        if not zero.exists():
            save(zero, dict(protocol_sha256=reg, epoch=0, arm=arm, state=model.state_dict()))
        else:
            saved_zero = torch.load(str(zero), map_location="cpu", weights_only=False)
            if saved_zero["protocol_sha256"] != reg or any(not torch.equal(v.cpu(), saved_zero["state"][k]) for k,v in model.state_dict().items()):
                raise RuntimeError("Epoch zero checkpoint drift")
        result = emit(p, reg, lane, arm, 0, "inner_validation", model, records, zero)
        m = result["metrics"]
        candidates.append(dict(epoch=0, object=m["post_nms_object_top1"], R50=m["R"]["50"], mR50=m["mR"]["50"]))
    start = time.monotonic()
    fit_ids = p["split"]["fit"]

    def snapshot(epoch, at, sequence, totals):
        save(resume, dict(protocol_sha256=reg, model=model.state_dict(), optimizer=opt.state_dict(),
                          generator=generator.get_state(), epoch=epoch, offset=at, order=sequence,
                          candidates=candidates, history=history, sums=totals))

    for epoch in range(first_epoch, limit+1):
        epoch_start = time.monotonic()
        if order is None:
            order = torch.randperm(len(fit_ids), generator=generator).tolist()
            offset = 0
            sums = {k:0. for k in ["loss", "object_ce", "native_kl", "relation_protection"]}
        model.train()
        for index in range(offset, len(order), SPEC["accumulation_images"]):
            batch = order[index:index+SPEC["accumulation_images"]]
            opt.zero_grad(set_to_none=True)
            for j in batch:
                tokens, meta = records.get(fit_ids[j])
                values = objective(model(tokens, meta["native"]), meta, arm)
                if not all(torch.isfinite(v) for v in values.values()):
                    raise RuntimeError("Nonfinite training objective")
                (values["loss"]/len(batch)).backward()
                for k in sums:
                    sums[k] += float(values[k].detach())
            norm = nn.utils.clip_grad_norm_(model.parameters(), SPEC["clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            opt.step()
            at = index+len(batch)
            if at % 64 == 0 or at == len(order) or STOP:
                snapshot(epoch, at, order, sums)
                progress(stage, "train_visual_regions_epoch%d" % epoch, at, len(order), epoch_start,
                         epoch=epoch, max_epochs=limit, losses={k:v/at for k,v in sums.items()})
            if STOP:
                raise RuntimeError("Training stopped at optimizer boundary; resume checkpoint retained")
        model.eval()
        changes = {str(i):max(float((v.detach().cpu()-original[k]).abs().max()) for k,v in model.layers.state_dict().items() if k.startswith(str(i)+".")) for i in range(2)}
        if arm == "frozen_head" and any(changes.values()):
            raise RuntimeError("Frozen-head control changed visual blocks")
        if arm != "frozen_head" and not all(v > 0 for v in changes.values()):
            raise RuntimeError("Adaptive arm failed to update both visual blocks")
        epoch_path = path_root/("epoch%d.pt" % epoch)
        if not epoch_path.exists():
            save(epoch_path, dict(protocol_sha256=reg, epoch=epoch, arm=arm,
                                 state=model.state_dict(), tail_changes=changes))
        else:
            saved_epoch = torch.load(str(epoch_path), map_location="cpu", weights_only=False)
            if saved_epoch["protocol_sha256"] != reg or any(not torch.equal(v.cpu(), saved_epoch["state"][k]) for k,v in model.state_dict().items()):
                raise RuntimeError("Existing epoch checkpoint drift")
        result = emit(p, reg, lane, arm, epoch, "inner_validation", model, records, epoch_path)
        m = result["metrics"]
        candidates.append(dict(epoch=epoch, object=m["post_nms_object_top1"], R50=m["R"]["50"], mR50=m["mR"]["50"]))
        history.append(dict(epoch=epoch, losses={k:v/len(order) for k,v in sums.items()}, tail_changes=changes))
        atomic_json(stage/"candidates.json", dict(protocol_sha256=reg, candidates=candidates, history=history))
        order, offset = None, 0
        snapshot(epoch+1, 0, None, None)
    selected = choose_epoch(candidates)
    selected_path = path_root/("epoch%d.pt" % selected["selected_epoch"])
    finish(stage, reg, len(fit_ids), arm=arm, candidates=candidates, history=history,
           checkpoint=str(selected_path), checkpoint_sha256=sha256(selected_path),
           gradient_checks_passed=True, **selected)


def held(p, reg, lane, arm):
    for a in ARMS[1:]:
        if not checked_done(OUT/lane/a/"train", reg):
            raise RuntimeError("All selections must finish before held predictions")
    done = read(OUT/lane/arm/"train/summary.json")
    if sha256(done["checkpoint"]) != done["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint drift")
    epoch = done["selected_epoch"]
    vision, _ = load_vision(p)
    model = VisualResidual(vision, arm != "frozen_head").cuda().eval()
    value = torch.load(done["checkpoint"], map_location="cpu", weights_only=False)
    if value["protocol_sha256"] != reg or value["arm"] != arm or value["epoch"] != epoch:
        raise RuntimeError("Selected checkpoint metadata mismatch")
    model.load_state_dict(value["state"], strict=True)
    del vision
    emit(p, reg, lane, arm, epoch, "held", model, Records(p, reg, lane), Path(done["checkpoint"]))
    finish(OUT/lane/arm/"held", reg, len(p["split"]["held"]), epoch=epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract", "train", "held"], required=True)
    parser.add_argument("--shard", type=int, choices=[0, 1])
    parser.add_argument("--arm", choices=ARMS[1:])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(2)
    torch.manual_seed(SPEC["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_stop)
    p, reg, lane = load(args.smoke)
    if args.stage == "extract":
        if args.shard is None:
            raise ValueError("Shard required")
        extract(p, reg, lane, args.shard)
    elif args.stage == "train":
        train(p, reg, lane, args.arm)
    else:
        held(p, reg, lane, args.arm)


if __name__ == "__main__":
    main()
