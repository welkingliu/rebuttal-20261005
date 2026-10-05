"""All-image eligibility audit; unmapped annotations are not prediction errors."""
from collections import Counter
import json

from common import OLD, ROOT, atomic_json, ensure_storage, sha256
from extension_protocol import verify


def normalize(value):
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def audit(graphs, objects, predicates):
    counts = Counter()
    classes, relations, eligible = Counter(), Counter(), Counter()
    rows = []
    for iid, names, triples in graphs:
        names = {str(k): normalize(v) for k, v in names.items()}
        classes.update(names.values())
        c = Counter(images=1, objects=len(names), mapped_objects=sum(n in objects for n in names.values()))
        for s, p, o in triples:
            p = normalize(p)
            relations[p] += 1
            endpoint = names[str(s)] in objects and names[str(o)] in objects
            predicate = p in predicates
            c["relations"] += 1
            c["endpoints_mapped"] += int(endpoint)
            c["predicate_mapped"] += int(predicate)
            c["eligible_relations"] += int(endpoint and predicate)
            c[("endpoint_ok" if endpoint else "endpoint_unmapped") + "/" +
              ("predicate_ok" if predicate else "predicate_unmapped")] += 1
            if endpoint and predicate:
                eligible[p] += 1
        c["images_with_eligible_relations"] = int(c["eligible_relations"] > 0)
        counts.update(c)
        rows.append(dict(image_id=str(iid), **c))
    return dict(counts=dict(counts), per_image=rows,
                relation_instance_coverage=counts["eligible_relations"] / max(counts["relations"], 1),
                object_instance_coverage=counts["mapped_objects"] / max(counts["objects"], 1),
                object_class_coverage=sum(x in objects for x in classes) / max(len(classes), 1),
                predicate_class_coverage=sum(x in predicates for x in relations) / max(len(relations), 1),
                predicate_support={p: dict(total=n, eligible=eligible[p], mapped=p in predicates)
                                   for p, n in relations.most_common()},
                unmapped_object_support={p: n for p, n in classes.most_common() if p not in objects},
                estimand="annotation-level exact-ontology eligibility; no model performance inferred")


def main():
    ensure_storage(); verify()
    dictionary = OLD / "data/vg/v1.4/VG-SGG-dicts.json"
    vocab = json.loads(dictionary.read_text())
    objects = {normalize(x) for x in vocab["label_to_idx"]}
    predicates = {normalize(x) for x in vocab["predicate_to_idx"]}
    gp = OLD / "data/gqa/val_sceneGraphs.json"
    gqa = json.loads(gp.read_text())
    def gqa_graphs():
        for iid, graph in gqa.items():
            nodes = graph["objects"]
            yield iid, {k: v["name"] for k, v in nodes.items()}, [
                (k, r["name"], str(r["object"])) for k, v in nodes.items() for r in v["relations"]]
    out = ROOT / "results/R7_coverage"
    result = audit(gqa_graphs(), objects, predicates)
    result["source_sha256"] = sha256(gp)
    atomic_json(out / "gqa.json", result)
    vp = OLD / "data/vrd/json_dataset"
    vo = json.loads((vp / "objects.json").read_text())
    vr = json.loads((vp / "predicates.json").read_text())
    annotations = json.loads((vp / "annotations_test.json").read_text())
    def vrd_graphs():
        for iid, entries in annotations.items():
            nodes, lookup, triples = {}, {}, []
            for r in entries:
                ids = []
                for role in ("subject", "object"):
                    obj = r[role]
                    key = (obj["category"], tuple(obj["bbox"]))
                    if key not in lookup:
                        lookup[key] = str(len(nodes)); nodes[lookup[key]] = vo[obj["category"]]
                    ids.append(lookup[key])
                triples.append((ids[0], vr[r["predicate"]], ids[1]))
            yield iid, nodes, triples
    result = audit(vrd_graphs(), objects, predicates)
    result["source_sha256"] = {p.name: sha256(p) for p in [vp / "annotations_test.json", vp / "objects.json", vp / "predicates.json"]}
    atomic_json(out / "vrd.json", result)
    atomic_json(out / "summary.json", dict(status="complete", ontology_sha256=sha256(dictionary),
                reports=[str(out / "gqa.json"), str(out / "vrd.json")],
                caveat="Coverage is not repaired by increasing sample size. Do not extrapolate eligible-subset accuracy to the full dataset."))
    print("[COMPLETE] R7 external coverage", flush=True)


if __name__ == "__main__":
    main()
