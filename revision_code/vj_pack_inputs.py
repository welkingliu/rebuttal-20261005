"""Read-only input export for the Mac V-J pilot; no native model/GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from common import ROOT, atomic_json, ensure_storage, output_path, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ensure_storage()
    torch.set_num_threads(2)
    out = output_path(args.output)
    if out.exists():
        raise RuntimeError("Preserve existing transfer bundle")
    start = time.monotonic()
    vi = ROOT / "manifests/VI_protocol.json"
    vh = ROOT / "results/VH_independent_identity/protocol.json"
    vf = ROOT / "manifests/VF_protocol.json"
    split_path = ROOT / "results/VI_selective_fusion/splits.json"
    splits = read(split_path)
    previous = set()
    split_sources = {}
    for task in ["sgcls", "sgdet"]:
        paths = [ROOT / "cache" / name / task / "protocol.json" for name in ["VB", "VC"]]
        paths += [ROOT / "results/VE" / task / "splits.json"]
        for path in paths:
            value = read(path)
            previous.update(value["development"] + value["gate"])
            split_sources[str(path)] = sha256(path)
    reserved = splits["gate"]
    if len(reserved) != 1000 or len(set(reserved)) != 1000 or set(reserved) & previous:
        raise RuntimeError("Reserved gate overlaps earlier adaptation evidence")
    if set(splits["train"]) & set(splits["development"] + reserved):
        raise RuntimeError("Split leakage")
    for name in ["VF", "VI_selective_fusion"]:
        for task in ["sgcls", "sgdet"]:
            gate = ROOT / "results" / name / task / "gate"
            if gate.exists() and any(gate.rglob("*.json")):
                raise RuntimeError("Reserved adaptation gate already consumed")
    inputs = [vi, vh, vf, split_path]
    data = {}
    for split in ["train", "development"]:
        xs, ys, bs, offsets = [], [], [], [0]
        source_digest = hashlib.sha256()
        for n, iid in enumerate(splits[split], 1):
            a = ROOT / "cache/VH_independent_identity" / split / (iid + ".pt")
            b = ROOT / "cache/VF/sgcls" / split / (iid + ".pt")
            f = torch.load(str(a), map_location="cpu", weights_only=True)
            p = torch.load(str(b), map_location="cpu", weights_only=True)
            if (f["protocol_sha256"] != sha256(vh) or p["registration_sha256"] != sha256(vf)
                    or f["image_id"] != iid or p["image_id"] != iid
                    or not torch.equal(f["labels"], p["targets"])):
                raise RuntimeError("Feature, native-score or label provenance mismatch: " + iid)
            xs.append(f["features"]); ys.append(f["labels"]); bs.append(p["baseline"])
            offsets.append(offsets[-1] + len(f["labels"]))
            source_digest.update((iid + ":" + sha256(a) + ":" + sha256(b) + "\n").encode())
            if n % 500 == 0:
                print(json.dumps(dict(stage="pack_" + split, images=n, total=len(splits[split]))), flush=True)
        data[split] = dict(features=torch.cat(xs), labels=torch.cat(ys), baseline=torch.cat(bs),
                           image_ids=splits[split], offsets=offsets,
                           ordered_input_digest=source_digest.hexdigest())
    oof_path = ROOT / "cache/VI_selective_fusion/oof.pt"
    selected = ROOT / "checkpoints/VI_selective_fusion/selected.pth"
    oof = torch.load(str(oof_path), map_location="cpu", weights_only=True)
    ckpt = torch.load(str(selected), map_location="cpu", weights_only=True)
    groups = [iid for iid, a, b in zip(splits["train"], data["train"]["offsets"][:-1], data["train"]["offsets"][1:])
              for _ in range(b - a)]
    if (oof["protocol_sha256"] != sha256(vi) or ckpt["protocol_sha256"] != sha256(vi)
            or groups != oof["image_ids"] or not torch.equal(oof["labels"], data["train"]["labels"])):
        raise RuntimeError("OOF alignment failure")
    inputs += [oof_path, selected]
    folds = []
    for fold in range(5):
        path = ROOT / "checkpoints/VI_selective_fusion" / ("fold%d.pth" % fold)
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if payload["protocol_sha256"] != sha256(vi):
            raise RuntimeError("Wrong auxiliary fold source")
        folds.append(payload["head"]); inputs.append(path)
    provenance = dict(source_root=str(ROOT), input_hashes={str(p): sha256(p) for p in inputs},
        earlier_split_hashes=split_sources, reserved_gate_ids=reserved,
        reserved_gate_unused_for_adapter_selection=True,
        caveat="Native reference training/audits already used VG validation; this is adaptation-held-out, not globally unseen data",
        exporter_sha256=sha256(Path(__file__)), cuda_used=False, seconds=time.monotonic() - start)
    bundle = dict(data=data, oof=oof["expert"], fold_heads=folds, selected=ckpt,
                  folds=read(vi)["folds"], provenance=provenance)
    temporary = out.with_suffix(".tmp")
    torch.save(bundle, str(temporary)); temporary.replace(out)
    atomic_json(out.with_suffix(".json"), dict(**provenance, bundle_sha256=sha256(out), bytes=out.stat().st_size))
    print(json.dumps(dict(status="complete", bundle=str(out), bytes=out.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()
