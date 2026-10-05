"""Replay original DINO crop features on three old development images only."""
import json

from PIL import Image
import torch
from torch.nn import functional as F

from common import ROOT, atomic_json, ensure_storage, sha256
from independent_identity_expert import crop_tensor, REPO, WEIGHT
from vk_gate_protocol import read, MANIFEST, verify


def main():
    ensure_storage(); registration = verify(full=True); torch.set_num_threads(2)
    manifest = read(MANIFEST); selected = manifest["smoke_ids"]
    assert not set(selected) & set(manifest["gate_ids"])
    items = {r["image_id"]: r for r in read(ROOT / "results/VH_independent_identity/annotations.json")["development"]}
    initial = dict(benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    encoder = torch.hub.load(str(REPO), "dinov2_vitb14", source="local", pretrained=False)
    encoder.load_state_dict(torch.load(str(WEIGHT), map_location="cpu", weights_only=True), strict=True)
    encoder.cuda().eval().requires_grad_(False)
    rows = []
    for deterministic in [True, False]:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        for iid in selected:
            item = items[iid]
            with Image.open(item["image"]) as source:
                image = source.convert("RGB")
            crops = [crop_tensor(image, box) for box in item["boxes"]]
            with torch.inference_mode():
                parts = [encoder(torch.stack(crops[j:j+8]).cuda()).float().cpu() for j in range(0, len(crops), 8)]
                features = F.normalize(torch.cat(parts), dim=-1)
            path = ROOT / "cache/VH_independent_identity/development" / (iid + ".pt")
            original = torch.load(str(path), map_location="cpu", weights_only=True)["features"]
            row = dict(image_id=iid, cudnn_deterministic=deterministic,
                reference_sha256=sha256(path), reference_path=str(path),
                passed=torch.allclose(features, original, atol=2e-5, rtol=2e-4),
                max_error=float((features-original).abs().max()))
            rows.append(row); print(json.dumps(row), flush=True)
    atomic_json(ROOT / "results/VK_runtime_audit_20261003/crops/summary.json",
        dict(status="complete", original_registration_sha256=registration,
             encoder_sha256=sha256(WEIGHT), initial_backend=initial,
             gate_evaluated=False, smoke_ids=selected, comparisons=rows))


if __name__ == "__main__":
    main()
