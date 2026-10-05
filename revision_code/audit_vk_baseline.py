"""Read-only baseline comparisons on previously used smoke images, not gate data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from common import ROOT, atomic_json, ensure_storage, sha256
from native_runtime import infer
from repro_experiment import dataset
from ve_experiment import lookup
from ve_protocol import baseline
from vi_native import setup
from vk_gate_protocol import MANIFEST, read, verify


def compare(values, reference):
    result = {}
    for key, value in values.items():
        expected = reference[key]
        shape = value.shape == expected.shape
        integer = np.issubdtype(value.dtype, np.integer)
        ok = shape and (np.array_equal(value, expected) if integer else
                        np.allclose(value, expected, atol=1e-5, rtol=1e-5))
        result[key] = dict(shape_match=shape, passed=bool(ok),
            max_error=float(np.max(np.abs(value - expected))) if shape and value.size else None,
            changed=int(np.count_nonzero(value != expected)) if shape else None)
    return result


def differences(a, b, prefix=""):
    result = {}
    for key in sorted(set(a) | set(b)):
        name = prefix + str(key)
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            result.update(differences(a[key], b[key], name + "."))
        elif a.get(key) != b.get(key):
            result[name] = dict(reference=a.get(key), current=b.get(key))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["sgcls", "sgdet"], default="sgcls")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ensure_storage()
    registration = verify(full=True)
    manifest = read(MANIFEST)
    torch.set_num_threads(2)
    initial = dict(benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    model, cfg, transform, provenance, patch = setup(args.task, args.output)
    patch.enabled = False
    reference_provenance = read(baseline(args.task) / "model_provenance.json")
    assert provenance["checkpoint_sha256"] == reference_provenance["checkpoint_sha256"]
    assert provenance["checkpoint_sha256"] == manifest["baseline_assets"][args.task]["sha256"]
    ds = dataset(cfg, "val")
    mapping = lookup(ds)
    selected = manifest["smoke_ids"]
    assert len(selected) == 3 and not set(selected) & set(manifest["gate_ids"])
    folder = (ROOT / "results/R14_pair_cap/sgdet/predictions/reference_all_pairs" if args.task == "sgdet"
              else ROOT / "results/R9_reproduction/sgcls/eval_corrected/val/predictions")
    from export_pysgg_vg_task import convert_prediction
    records = []
    # Both backends are compared against the same immutable old caches.
    for deterministic in [True, False]:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        for iid in selected:
            torch.manual_seed(17)
            pred, _ = infer(model, cfg, transform, ds, mapping[iid])
            values = convert_prediction(pred.to("cpu"))
            path = folder / (iid + ".npz")
            with np.load(path, allow_pickle=False) as reference:
                errors = compare(values, reference)
                dense = None
                if "dense_postprocessor_object_logits" in reference:
                    dense = compare(dict(logits=patch.capture["baseline"].cpu().numpy()),
                                    dict(logits=reference["dense_postprocessor_object_logits"]))
            row = dict(image_id=iid, cudnn_deterministic=deterministic,
                       reference_path=str(path), reference_sha256=sha256(path),
                       passed=all(x["passed"] for x in errors.values()), errors=errors,
                       dense_logit_comparison=dense)
            records.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(args.output / "comparisons.json", records)
    patch.close()
    result = dict(status="complete", task=args.task, original_registration_sha256=registration,
        checkpoint_sha256=provenance["checkpoint_sha256"], initial_backend=initial,
        config_differences=differences(yaml.safe_load(reference_provenance["config"]),
                                       yaml.safe_load(provenance["config"])),
        statistics_match=reference_provenance["statistics_sha256"] == provenance["statistics_sha256"],
        gate_evaluated=False, smoke_ids=selected, comparisons=records)
    atomic_json(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    main()
