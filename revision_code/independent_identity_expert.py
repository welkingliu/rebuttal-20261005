"""V-H: finite frozen-DINO crop-expert feasibility, without gate or test access."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, OLD, atomic_json, ensure_storage, output_path, sha256

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/VH_independent_identity"
CACHE = ROOT / "cache/VH_independent_identity"
WEIGHT = OLD / "checkpoints/foundation/dinov2/dinov2_vitb14_pretrain.pth"
REPO = OLD / "external/foundation_repos/dinov2"
SPEC = dict(version="vh_dinov2_independent_crop_expert_v1", task="sgcls", seed=17,
    train_images=5000, development_images=500,
    model="local DINOv2 ViT-B/14 frozen; normalized CLS feature; no additional model downloads",
    support="GT bounding-box crops; supplied support, not autonomous detection or segmentation",
    crop="inclusive GT box, clipped to image; centered square RGB(124,116,104) pad; bicubic224; ImageNet normalization",
    feature_batch=8, precision="fp32 encoder", probe="linear 768-to150 foreground classifier",
    optimizer="AdamW", learning_rate=.003, weight_decay=.01, batch_size=256,
    max_epochs=100, min_epochs=10, patience=10,
    checkpoint_selection="maximum development object Top1, then minimum NLL; no gate/test selection",
    candidates=[dict(name="selective_alpha0.25", alpha=.25, threshold=.7),
                dict(name="selective_alpha0.50", alpha=.5, threshold=.7)],
    expert_confidence_min=.5,
    correction="only low-confidence native predictions with confident disagreeing expert; keep background mass, not old foreground scores",
    requirements=dict(identity_gain_min=.005, correction_precision_min=.6, macro_accuracy_not_lower=True),
    selection="among two eligible fixed candidates: highest Top1, then fewer damaged objects, then alpha0.25",
    stop="development feasibility only; no automatic model/parameter search, fresh gate or final test",
    next_stage="if eligible, separately register native NMS/ranking and unchanged two-task identity/relation gates",
    interpretation="new auxiliary visual evidence, not another complete SGG model or evidence of open-vocabulary SGG",
    wall_budget_hours=12)


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError("Existing V-H protocol differs; preserve it")
    if not path.exists():
        atomic_json(path, value)


def save(path, value):
    path = output_path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, str(temp)); temp.replace(path)


def budget(deadline):
    if time.time() >= deadline:
        raise TimeoutError("V-H feasibility budget exhausted")


def update(stage, n, total, start, **extra):
    seconds = time.monotonic() - start
    record = dict(detail=stage, images=n, total=total, seconds=seconds,
                  eta_seconds=seconds / max(n, 1) * (total - n), **extra)
    atomic_json(OUT / "progress.json", record); print(json.dumps(record), flush=True)


def export():
    from vf_protocol import verify
    from repro_experiment import config, dataset
    from ve_experiment import lookup
    source = verify()
    splits = read(ROOT / "results/VF/sgcls/splits.json")
    cfg = config("sgcls", OUT / "export")
    annotations = {}
    for split, native in [("train", "train"), ("development", "val")]:
        ds = dataset(cfg, native); mapping = lookup(ds); rows = []
        for iid in splits[split]:
            index = mapping[iid]; gt = ds.get_groundtruth(index, evaluation=True)
            cached = torch.load(str(ROOT / "cache/VF/sgcls" / split / (iid + ".pt")), map_location="cpu")
            labels = gt.get_field("labels").cpu().long()
            if not torch.equal(labels, cached["targets"]) or cached["registration_sha256"] != source:
                raise RuntimeError("Independent crop object order differs from native SGG")
            image = Path(ds.filenames[index]).resolve()
            if not image.is_file():
                raise RuntimeError("Missing raw image: " + str(image))
            rows.append(dict(image_id=iid, image=str(image), boxes=gt.bbox.cpu().tolist(), labels=labels.tolist(), size=list(gt.size)))
        annotations[split] = rows
    if set(splits["train"]) & set(splits["development"]):
        raise RuntimeError("Train/development overlap")
    value = dict(spec=SPEC, source_vf_registration=source,
        sources={name: sha256(HERE / name) for name in ["independent_identity_expert.py", "run_independent_identity_expert.sh"]},
        encoder_sha256=sha256(WEIGHT), encoder_sources={str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "dinov2").rglob("*.py"))},
        annotations_sha256=hashlib.sha256(json.dumps(annotations, sort_keys=True).encode()).hexdigest(),
        fresh_gate_not_accessed=True, test_not_accessed=True)
    immutable(OUT / "protocol.json", value)
    immutable(OUT / "annotations.json", annotations)
    atomic_json(OUT / "export/summary.json", dict(status="complete", train_images=len(annotations["train"]),
        development_images=len(annotations["development"]), protocol_sha256=sha256(OUT / "protocol.json")))
    print("[EXPORTED] aligned native GT boxes, classes and image IDs", flush=True)


def crop_tensor(image, box):
    from PIL import Image
    from torchvision.transforms import functional as TF
    x0, y0, x1, y1 = box
    bounds = (max(0, math.floor(x0)), max(0, math.floor(y0)), min(image.width, math.ceil(x1 + 1)), min(image.height, math.ceil(y1 + 1)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise RuntimeError("Empty supplied-support crop")
    crop = image.crop(bounds)
    edge = max(crop.size)
    square = Image.new("RGB", (edge, edge), (124, 116, 104))
    square.paste(crop, ((edge - crop.width) // 2, (edge - crop.height) // 2))
    square = square.resize((224, 224), Image.Resampling.BICUBIC)
    return TF.normalize(TF.to_tensor(square), [.485, .456, .406], [.229, .224, .225])


def extract(annotations, deadline):
    from PIL import Image
    protocol = read(OUT / "protocol.json")
    if sha256(WEIGHT) != protocol["encoder_sha256"]:
        raise RuntimeError("Frozen encoder changed")
    model = torch.hub.load(str(REPO), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(str(WEIGHT), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True); model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = sha256(OUT / "protocol.json")
    start, completed = time.monotonic(), 0
    for split in ["development", "train"]:
        for item in annotations[split]:
            budget(deadline)
            path = CACHE / split / (item["image_id"] + ".pt")
            if path.exists():
                if torch.load(str(path), map_location="cpu", weights_only=True)["protocol_sha256"] != digest:
                    raise RuntimeError("Incompatible crop feature cache")
            else:
                with Image.open(item["image"]) as source:
                    image = source.convert("RGB")
                if list(image.size) != item["size"]:
                    raise RuntimeError("Image/GT geometry mismatch: " + item["image_id"])
                crops = [crop_tensor(image, box) for box in item["boxes"]]
                features = []
                with torch.inference_mode():
                    for j in range(0, len(crops), SPEC["feature_batch"]):
                        batch = torch.stack(crops[j:j + SPEC["feature_batch"]]).cuda()
                        features.append(model(batch).float().cpu())
                x = F.normalize(torch.cat(features), dim=-1)
                if x.shape != (len(item["labels"]), 768) or not torch.isfinite(x).all():
                    raise RuntimeError("Invalid independent visual feature")
                save(path, dict(features=x, labels=torch.tensor(item["labels"]), image_id=item["image_id"], protocol_sha256=digest))
            completed += 1
            if completed % 25 == 0:
                update("frozen DINOv2 GT-crop features: " + split, completed, 5500, start)
    del model; torch.cuda.empty_cache()
    atomic_json(OUT / "feature_summary.json", dict(status="complete", images=completed, protocol_sha256=digest,
        seconds=time.monotonic() - start, encoder_sha256=protocol["encoder_sha256"]))


def load_split(items, split):
    data = [torch.load(str(CACHE / split / (i["image_id"] + ".pt")), map_location="cpu", weights_only=True) for i in items]
    x = torch.cat([i["features"] for i in data]).cuda()
    y = torch.cat([i["labels"] for i in data]).cuda() - 1
    if not bool(((y >= 0) & (y < 150)).all()):
        raise RuntimeError("Wrong foreground vocabulary")
    return x, y


def fit(annotations, deadline):
    x, y = load_split(annotations["train"], "train")
    dx, dy = load_split(annotations["development"], "development")
    torch.manual_seed(17)
    head = nn.Linear(768, 150).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"])
    best, stale, history = None, 0, []
    path = ROOT / "checkpoints/VH_independent_identity/linear_seed17.pth"
    fit_started = time.monotonic()
    for epoch in range(1, SPEC["max_epochs"] + 1):
        budget(deadline); start = time.monotonic(); head.train()
        generator = torch.Generator(device="cuda").manual_seed(17 + epoch)
        order = torch.randperm(len(y), generator=generator, device="cuda")
        total = 0.
        for index in order.split(SPEC["batch_size"]):
            budget(deadline); optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[index]), y[index])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite auxiliary probe loss")
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(index)
        head.eval()
        with torch.no_grad():
            logits = head(dx)
            top1 = float((logits.argmax(-1) == dy).float().mean())
            nll = float(F.cross_entropy(logits, dy))
        row = dict(epoch=epoch, train_nll=total / len(y), development_top1=top1, development_nll=nll)
        history.append(row); print(json.dumps(row), flush=True)
        score = (top1, -nll)
        if best is None or score > best:
            best, stale = score, 0
            save(path, dict(head=head.state_dict(), epoch=epoch, development=row, protocol_sha256=sha256(OUT / "protocol.json")))
        else:
            stale += 1
        atomic_json(OUT / "training/history.json", history)
        update("independent linear probe; ETA to epoch cap, early stopping may finish sooner", epoch, SPEC["max_epochs"], fit_started, selected_top1=best[0])
        if epoch >= SPEC["min_epochs"] and stale >= SPEC["patience"]:
            break
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    head.load_state_dict(payload["head"]); head.eval()
    atomic_json(OUT / "training/summary.json", dict(status="complete", selected=payload["development"],
        history=history, checkpoint_sha256=sha256(path), train_objects=len(y), development_objects=len(dy)))
    return head


def combine(logits, expert, alpha):
    prob = logits.softmax(-1); mass = prob[:, 1:].sum(-1, keepdim=True)
    confidence, before = (prob[:, 1:] / mass.clamp_min(1e-12)).max(-1)
    econf, after = expert.max(-1)
    mask = (confidence < .7) & (econf >= .5) & (before != after)
    changed = prob.clone(); changed[:, 1:] = (1 - alpha) * prob[:, 1:] + alpha * mass * expert
    return torch.where(mask[:, None], changed, prob), mask


def summary(rows):
    sums = {k: sum(i[k] for i in rows) for k in ["objects", "correct", "baseline_correct", "repaired", "damaged", "eligible", "label_flips"]}
    counts = np.sum([i["class_count"] for i in rows], axis=0)
    correct = np.sum([i["class_correct"] for i in rows], axis=0)
    positive = counts > 0
    sums.update(top1=sums["correct"] / sums["objects"], delta=(sums["correct"] - sums["baseline_correct"]) / sums["objects"],
        macro_accuracy=float((correct[positive] / counts[positive]).mean()),
        correction_precision=sums["repaired"] / (sums["repaired"] + sums["damaged"]) if sums["repaired"] + sums["damaged"] else None)
    return sums


def evaluate(head, annotations):
    names = ["native", "expert_only"] + [c["name"] for c in SPEC["candidates"]]
    rows = {name: [] for name in names}
    with torch.no_grad():
        for item in annotations["development"]:
            iid = item["image_id"]
            feature = torch.load(str(CACHE / "development" / (iid + ".pt")), map_location="cpu", weights_only=True)
            baseline = torch.load(str(ROOT / "cache/VF/sgcls/development" / (iid + ".pt")), map_location="cpu", weights_only=True)
            y, logits = feature["labels"].cuda(), baseline["baseline"].cuda()
            if not torch.equal(feature["labels"], baseline["targets"]):
                raise RuntimeError("Auxiliary/native label alignment changed")
            expert = head(feature["features"].cuda()).softmax(-1)
            original = logits.softmax(-1)
            exp = torch.cat([original[:, :1], (1 - original[:, :1]) * expert], -1)
            conditions = {"native": (original, torch.zeros(len(y), dtype=torch.bool, device="cuda")),
                          "expert_only": (exp, torch.ones(len(y), dtype=torch.bool, device="cuda"))}
            for c in SPEC["candidates"]:
                conditions[c["name"]] = combine(logits, expert, c["alpha"])
            old = logits[:, 1:].argmax(-1) + 1 == y
            for name, (p, mask) in conditions.items():
                new_label = p[:, 1:].argmax(-1) + 1; correct = new_label == y
                rows[name].append(dict(image_id=iid, objects=len(y), correct=int(correct.sum()), baseline_correct=int(old.sum()),
                    repaired=int((~old & correct).sum()), damaged=int((old & ~correct).sum()), eligible=int(mask.sum()),
                    label_flips=int((new_label != logits[:, 1:].argmax(-1) + 1).sum()),
                    class_count=torch.bincount(y, minlength=151)[1:].cpu().tolist(),
                    class_correct=torch.bincount(y[correct], minlength=151)[1:].cpu().tolist()))
    results = {name: summary(r) for name, r in rows.items()}
    eligible, checks = [], {}
    for c in SPEC["candidates"]:
        result = results[c["name"]]
        checks[c["name"]] = dict(identity_gain=result["delta"] >= .005,
            correction_precision=result["correction_precision"] is not None and result["correction_precision"] >= .6,
            macro_accuracy=result["macro_accuracy"] >= results["native"]["macro_accuracy"])
        if all(checks[c["name"]].values()):
            eligible.append(c)
    selected = max(eligible, key=lambda c: (results[c["name"]]["top1"], -results[c["name"]]["damaged"], -c["alpha"])) if eligible else None
    atomic_json(OUT / "development_images.json", rows)
    atomic_json(OUT / "summary.json", dict(status="complete", development_eligible=bool(selected), selected=selected,
        results=results, checks=checks, images=500, fresh_gate_evaluated=False, test_evaluated=False,
        full_system_evaluated=False, protocol_sha256=sha256(OUT / "protocol.json"),
        conclusion="Feasibility only; original SGCls/SGDet joint gate still required"))
    print(json.dumps(dict(development_eligible=bool(selected), selected=selected, results=results)), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["export", "run"])
    args = parser.parse_args(); ensure_storage(); torch.set_num_threads(2)
    if args.stage == "export":
        export(); return
    state_path = ROOT / "status/VH_independent_identity.json"
    status = dict(status="waiting_gpu", gpu=[1], pid=os.getpid(), command=[str(HERE / "independent_identity_expert.py"), "run"],
        completion=str(OUT / "summary.json"), progress_file=str(OUT / "progress.json"), log=str(ROOT / "logs/VH_independent_identity.log"))
    atomic_json(state_path, status)
    lock = (ROOT / "status/gpu1.resource.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("Use physical GPU1")
        if subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            raise RuntimeError("GPU1 occupied")
        manifest = read(OUT / "protocol.json")
        if manifest["spec"] != SPEC or any(sha256(HERE / k) != v for k, v in manifest["sources"].items()):
            raise RuntimeError("Registered V-H code changed")
        atomic_json(state_path, dict(status, status="running"))
        annotations = read(OUT / "annotations.json")
        if hashlib.sha256(json.dumps(annotations, sort_keys=True).encode()).hexdigest() != manifest["annotations_sha256"]:
            raise RuntimeError("Crop annotation manifest changed")
        deadline = time.time() + SPEC["wall_budget_hours"] * 3600
        extract(annotations, deadline)
        head = fit(annotations, deadline)
        evaluate(head, annotations)
        atomic_json(state_path, dict(status, status="complete"))
    except Exception as exc:
        atomic_json(state_path, dict(status, status="failed", reason=str(exc))); raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
