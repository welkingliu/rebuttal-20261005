"""Recompute the VG seen-triplet manifest from all native training annotations."""
import json

from common import OLD, ROOT, ensure_storage, atomic_json, sha256
from native_runtime import configure, dataset


def triples(ds):
    for labels,relations in zip(ds.gt_classes,ds.relationships):
        for subject,obj,predicate in relations:
            yield int(labels[subject]),int(predicate),int(labels[obj])


def main():
    ensure_storage()
    out=ROOT/"results/R0/zero_shot"
    cfg=configure("transformer","sgcls",out)
    train=dataset(cfg,"train",all_training=True)
    seen=set(triples(train))
    source=OLD/"artifacts/manifests/seen_triplets_full.json"
    reference=json.loads(source.read_text())
    expected={tuple(x) for x in reference["vg"]}
    test=dataset(cfg,"test")
    rows=list(triples(test))
    record=dict(status="complete",source_manifest=str(source),source_sha256=sha256(source),
        training_images=len(train),training_relation_rows=sum(len(r) for r in train.relationships),
        distinct_training_triplets=len(seen),declared_training_images=reference["_metadata"]["vg"]["num_images"],
        missing_in_manifest=len(seen-expected),extra_in_manifest=len(expected-seen),
        manifest_matches_native_full_training=seen==expected,test_images=len(test),test_relation_rows=len(rows),
        zero_shot_test_relation_rows=sum(t not in seen for t in rows),
        note="Full training annotations, not the 5000-image probe/mitigation training subset; repeated test annotations retain their own denominator rows.")
    atomic_json(out/"summary.json",record)
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    main()
