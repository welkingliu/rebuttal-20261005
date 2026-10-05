"""Compare the reconstructed zero-shot denominator to the released reference set."""
import json
from common import ROOT, OLD, atomic_json, ensure_storage, sha256
from native_runtime import configure, dataset


def triples(ds):
    for labels, relations in zip(ds.gt_classes, ds.relationships):
        for s, o, p in relations:
            yield int(labels[s]), int(labels[o]), int(p)


def main():
    import torch
    ensure_storage()
    out = ROOT / "results/R14_reproduction_audit"
    cfg = configure("transformer", "sgdet", out / "zero_runtime")
    full = set(triples(dataset(cfg, "train", all_training=True)))
    actual_train = set(triples(dataset(cfg, "train")))
    test = list(triples(dataset(cfg, "test")))
    path = OLD / "external/official_repos/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch"
    ref = torch.load(str(path), map_location="cpu")
    ref = {tuple(map(int, row)) for row in ref.tolist()}
    reconstructed = set(test) - full
    heldout = set(test) - actual_train
    rows_differ = sum((t in ref) != (t in reconstructed) for t in test)
    record = dict(status="complete", released_set_sha256=sha256(path),
                  coordinate_order=["subject_class", "object_class", "predicate"],
                  official_unique_triplets=len(ref), reconstructed_unique=len(reconstructed),
                  only_official=sorted(ref - reconstructed), only_reconstructed=sorted(reconstructed - ref),
                  test_relation_rows=len(test), official_zero_shot_rows=sum(t in ref for t in test),
                  reconstructed_zero_shot_rows=sum(t in reconstructed for t in test),
                  different_test_denominator_rows=rows_differ, denominator_matches_reference=rows_differ == 0,
                  true_train_only_zero_shot_rows=sum(t in heldout for t in test),
                  note="Train-only novelty and reference-release novelty are distinct contracts; do not silently interchange them")
    atomic_json(out / "zeroshot.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ["only_official", "only_reconstructed"]}), flush=True)


if __name__ == "__main__":
    main()
