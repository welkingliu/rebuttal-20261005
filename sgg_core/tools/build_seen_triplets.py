"""Build dataset-specific training triplet sets for zero-shot SGG recall."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from sgg_core.data.data_utils import build_vg_test_loader
from sgg_core.data.gqa_psg_data_utils import build_gqa_loader, build_psg_loader
from sgg_core.data.oi_data_utils import build_oi_loader
from sgg_core.data.vrd_data_utils import build_vrd_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Build seen SGG triplet manifests")
    parser.add_argument("--datasets", nargs="+", default=["vg", "oi", "psg", "gqa", "vrd"])
    parser.add_argument("--vg_root")
    parser.add_argument("--oi_root")
    parser.add_argument("--psg_train_ann")
    parser.add_argument("--psg_eval_ann")
    parser.add_argument("--gqa_train_scene_graph")
    parser.add_argument("--vrd_root")
    parser.add_argument("--max_images", type=int, default=1000000000)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--merge_existing", action="store_true",
        help="Preserve datasets not requested when the output already exists.",
    )
    return parser.parse_args()


def _loaders(args):
    loaders = {}
    if "vg" in args.datasets:
        loaders["vg"] = build_vg_test_loader(
            args.vg_root, args.max_images, split=0,
            include_proxy_features=False, include_raw_images=False,
        )
    if "oi" in args.datasets:
        loaders["oi"] = build_oi_loader(
            args.oi_root, "train", args.max_images,
            include_proxy_features=False, include_raw_images=False,
        )
    if "psg" in args.datasets:
        loaders["psg"] = build_psg_loader(
            args.psg_train_ann, args.max_images,
            exclude_annotation_path=args.psg_eval_ann,
            include_proxy_features=False, include_raw_images=False,
        )
    if "gqa" in args.datasets:
        loaders["gqa"] = build_gqa_loader(
            args.gqa_train_scene_graph, args.max_images,
            vocabulary_path=args.gqa_train_scene_graph,
            include_proxy_features=False, include_raw_images=False,
        )
    if "vrd" in args.datasets:
        loaders["vrd"] = build_vrd_loader(
            args.vrd_root, "train", args.max_images,
            include_proxy_features=False, include_raw_images=False,
        )
    return loaders


def _triplets(loader):
    seen = set()
    images = 0
    relations = 0
    object_support = Counter()
    for batch in loader:
        labels = batch["entity_labels"].long()
        object_support.update(int(value) for value in labels.tolist() if int(value) > 0)
        pairs = batch["rel_pairs"].long()
        predicates = batch["rel_labels"].long()
        images += 1
        for pair, predicate in zip(pairs, predicates):
            subject = int(pair[0])
            obj = int(pair[1])
            predicate = int(predicate)
            if (
                predicate > 0
                and 0 <= subject < labels.numel()
                and 0 <= obj < labels.numel()
            ):
                seen.add((int(labels[subject]), predicate, int(labels[obj])))
                relations += 1
    return sorted(seen), images, relations, object_support


def main():
    args = parse_args()
    output = Path(args.output)
    if args.merge_existing and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Seen-triplet manifest must be a mapping: {output}")
        payload.setdefault("_metadata", {})
    else:
        payload = {"_metadata": {}}
    for dataset_name, loader in _loaders(args).items():
        triplets, images, relations, object_support = _triplets(loader)
        payload[dataset_name] = [list(triplet) for triplet in triplets]
        payload["_metadata"][dataset_name] = {
            "ontology_id": getattr(loader.dataset, "ontology_id", None),
            "num_images": images,
            "num_relations": relations,
            "num_unique_triplets": len(triplets),
            "object_class_support": {
                str(key): value for key, value in sorted(object_support.items())
            },
        }
        print(f"{dataset_name}: images={images} unique_triplets={len(triplets)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
