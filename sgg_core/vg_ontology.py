"""Canonical VG-150 ontology loading and exact-order validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


BACKGROUND = "__background__"


def _ordered(mapping: dict, kind: str) -> tuple[str, ...]:
    indexed = {str(name): int(index) for name, index in mapping.items()}
    indexed.setdefault(BACKGROUND, 0)
    inverse = {}
    for name, index in indexed.items():
        if index in inverse:
            raise ValueError(
                f"Duplicate {kind} index {index}: {inverse[index]!r}, {name!r}"
            )
        inverse[index] = name
    expected = set(range(max(inverse) + 1))
    if set(inverse) != expected:
        missing = sorted(expected - set(inverse))
        raise ValueError(f"Non-contiguous {kind} ontology; missing indices {missing}")
    return tuple(inverse[index] for index in range(len(inverse)))


@dataclass(frozen=True)
class VG150Ontology:
    object_classes: tuple[str, ...]
    predicate_classes: tuple[str, ...]
    ontology_id: str
    source: str


def load_vg150_ontology(path) -> VG150Ontology:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    objects = _ordered(payload["label_to_idx"], "object")
    predicates = _ordered(payload["predicate_to_idx"], "predicate")
    if len(objects) != 151 or len(predicates) != 51:
        raise ValueError(
            "VG-150 requires 151 object entries and 51 predicate entries "
            f"including background, got {len(objects)} and {len(predicates)}"
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return VG150Ontology(objects, predicates, f"vg150:{digest}", str(source))


def assert_vg150_alignment(reference, candidate) -> dict:
    expected = load_vg150_ontology(reference)
    observed = load_vg150_ontology(candidate)
    failures = []
    for kind, left, right in (
        ("object", expected.object_classes, observed.object_classes),
        ("predicate", expected.predicate_classes, observed.predicate_classes),
    ):
        for index, (expected_name, observed_name) in enumerate(zip(left, right)):
            if expected_name != observed_name:
                failures.append({
                    "kind": kind,
                    "index": index,
                    "expected": expected_name,
                    "observed": observed_name,
                })
                break
    if failures:
        raise ValueError(f"VG-150 ontology order mismatch: {failures}")
    return {
        "status": "aligned",
        "ontology_id": expected.ontology_id,
        "reference": expected.source,
        "candidate": observed.source,
        "object_classes": len(expected.object_classes) - 1,
        "predicate_classes": len(expected.predicate_classes) - 1,
    }
