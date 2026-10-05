"""Frozen crop evidence and V-K scores; no labels enter repair inference."""
import argparse
import time

import torch
from torch.nn import functional as F
from PIL import Image

from common import ROOT, atomic_json, ensure_storage, sha256
from independent_identity_expert import crop_tensor, REPO, WEIGHT, save
from vk_gate_math import repair, validate_fixture
from vk_gate_protocol import CHECKPOINT, FIXTURE, PRIMARY_SHA, SPEC, read, verify, paths, eligible, ids


def run(args):
    eligible(args.task, args.smoke); registration = verify(full=True)
    out, cache = paths(args.task, args.smoke); folder = out / "features"
    export = read(out / "export/summary.json")
    if export["protocol_sha256"] != registration: raise RuntimeError("Incomplete native export")
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    fixture = torch.load(str(FIXTURE), map_location="cpu", weights_only=True)
    parity = validate_fixture(fixture, state)
    encoder = torch.hub.load(str(REPO), "dinov2_vitb14", source="local", pretrained=False)
    encoder.load_state_dict(torch.load(str(WEIGHT), map_location="cpu", weights_only=True), strict=True)
    encoder.cuda().eval().requires_grad_(False)
    selected = ids(args.smoke); started = time.monotonic(); crop_checks = []
    atomic_json(folder / "progress.json", dict(stage="frozen_crop_evidence", images=0, total=len(selected)))
    for n, iid in enumerate(selected, 1):
        if time.time() >= args.deadline: raise TimeoutError("V-K crop stage budget exhausted")
        item = torch.load(str(cache / "export" / (iid + ".pt")), map_location="cpu", weights_only=True)
        if item["image_id"] != iid or item["protocol_sha256"] != registration:
            raise RuntimeError("Native proposal order/provenance mismatch")
        update_path = cache / "updates" / (iid + ".pt")
        if update_path.exists() and not args.smoke:
            update = torch.load(str(update_path), map_location="cpu", weights_only=True)
            if (update["protocol_sha256"] != registration or update["checkpoint_sha256"] != PRIMARY_SHA
                    or update["image_id"] != iid or not torch.equal(update["native_logits"], item["baseline"])):
                raise RuntimeError("Reused V-K scores differ")
        else:
            with Image.open(item["image"]) as source:
                image = source.convert("RGB")
            if tuple(image.size) != tuple(item["image_size"]): raise RuntimeError("Raw image coordinate system changed")
            with torch.inference_mode():
                parts = []
                for chunk in item["crop_boxes"].split(SPEC["crop_batch"]):
                    crops = torch.stack([crop_tensor(image, box.tolist()) for box in chunk]).cuda()
                    parts.append(encoder(crops).float().cpu())
                features = F.normalize(torch.cat(parts), dim=-1)
                result = repair(features, item["baseline"], state)
            if args.smoke and args.task == "sgcls":
                original = torch.load(str(ROOT / "cache/VH_independent_identity/development" / (iid + ".pt")),
                    map_location="cpu", weights_only=True)["features"]
                if not torch.allclose(features, original, atol=2e-5, rtol=2e-4):
                    raise RuntimeError("Original supplied-box crop feature replay failed")
                crop_checks.append(dict(image_id=iid, max_error=float((features - original).abs().max())))
            save(cache / "features" / (iid + ".pt"), dict(image_id=iid, features=features,
                protocol_sha256=registration, crop_boxes=item["crop_boxes"]))
            save(update_path, dict(image_id=iid, logits=result["logits"], native_logits=item["baseline"],
                boxes=item["boxes"], size=item["size"], selected=result["selected"],
                repair_probability=result["repair_probability"], protocol_sha256=registration,
                checkpoint_sha256=PRIMARY_SHA))
        if n % 10 == 0 or n == len(selected):
            seconds = time.monotonic() - started
            value = dict(stage="frozen_crop_evidence", task=args.task, images=n, total=len(selected),
                seconds=seconds, eta_seconds=seconds / n * (len(selected) - n))
            atomic_json(folder / "progress.json", value); print(__import__("json").dumps(value), flush=True)
    if sha256(CHECKPOINT) != PRIMARY_SHA: raise RuntimeError("Frozen V-K weights changed")
    atomic_json(folder / "summary.json", dict(status="complete", protocol_sha256=registration, images=len(selected),
        checkpoint_sha256=PRIMARY_SHA, runtime_parity=parity, original_crop_checks=crop_checks,
        labels_used_in_repair=False, fitting_performed=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=SPEC["tasks"], required=True)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args(); ensure_storage(); torch.set_num_threads(2)
    torch.manual_seed(17); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    run(a)
