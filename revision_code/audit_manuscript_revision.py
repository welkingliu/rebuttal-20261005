"""Check revision structure, selected numeric claims, and source preservation."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "rebuttal/manuscript_revision_20261003"
SOURCE = Path("/path/to/local_user/Desktop/0507/project/kdd_sgg_core_experiments/tex/kdd2027_submission/main.tex")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def active_text(path):
    return re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())


def check_structure(text):
    stack = []
    for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if action == "begin":
            stack.append(environment)
        else:
            assert stack and stack.pop() == environment, environment
    assert not stack, stack
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    assert not (set(refs) - set(labels)), "Undefined references"
    depth = 0
    for brace in re.findall(r"(?<!\\)[{}]", text):
        depth += 1 if brace == "{" else -1
        assert depth >= 0, "Unbalanced braces"
    assert depth == 0, "Unbalanced braces"


def main():
    manuscript = active_text(REVISION / "main.tex")
    response_path = ROOT / "rebuttal/response_corrections_20261003.tex"
    check_structure(manuscript)
    check_structure(active_text(response_path))
    assert digest(SOURCE) == "fb4fa59a2973a251f09b7f0b8bbe6b0e6369305e28e262de82cfe646cc2044fe"
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript)
    assert all((REVISION / "figures" / name).is_file() for name in figures)
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", (REVISION / "reference.bib").read_text()))
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", manuscript) for key in group.split(",")}
    assert citations <= bib_keys, sorted(citations - bib_keys)
    abstract = manuscript.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    assert len(abstract.split()) <= 250

    evidence = {
        "R17": "output/rebuttal_results_20261002/R17_original_v/summary.json",
        "R13": "output/rebuttal_results_20261001/R13_gqa_disjoint/summary.json",
        "SGCls": "output/VK_native_results_20261003/sgcls/decision.json",
        "SGDet": "output/VK_native_results_20261003/sgdet/decision.json",
        "exposure": "output/evidence_completion_20261003/R21_release_execution/validation_exposure.json",
    }
    r17 = read_json(evidence["R17"])
    assert r17["records"] == 384
    assert r17["max_gradient_equivalence_error"] == 0
    assert r17["max_history_reconstruction_error"] < 1e-8
    gqa = read_json(evidence["R13"])
    assert gqa["images"] == 1674
    coverage = gqa["coverage"]
    assert coverage["retained_relations"] == 4676 and coverage["source_relations"] == 149588
    assert f'{100 * coverage["retained_relations"] / coverage["source_relations"]:.3f}' == "3.126"
    assert f'{100 * coverage["retained_objects"] / coverage["source_objects"]:.3f}' == "13.155"
    decisions = {task: read_json(evidence[task]) for task in ("SGCls", "SGDet")}
    assert decisions["SGCls"]["accepted"] and not decisions["SGDet"]["accepted"]
    for task, expected in (("SGCls", "0.943"), ("SGDet", "0.213")):
        assert decisions[task]["images"] == 1000
        assert f'{100 * decisions[task]["delta"]["object"]:.3f}' == expected
    exposure = read_json(evidence["exposure"])
    assert exposure["known_repair_exposed"] == 4782
    assert len(exposure["outside_indexed_repair_protocols"]) == 218
    assert not exposure["new_independent_gate_certified"]

    report = {
        "status": "passed_static_and_selected_evidence_checks",
        "source_preserved": {str(SOURCE): digest(SOURCE)},
        "revised_files": {str(p): digest(p) for p in (REVISION / "main.tex", response_path)},
        "evidence": {key: {"path": value, "sha256": digest(ROOT / value)} for key, value in evidence.items()},
        "abstract_whitespace_word_count": len(abstract.split()),
        "figure_references_present": figures,
        "scope": "Static structure and selected evidence assertions, not a TeX compilation or a full scientific re-audit.",
        "remaining": [
            "Verify rendered page budget, tables, and figure labels after compilation.",
            "Embedded historical overview/system labels are unchanged; captions explain the correction.",
            "No Overleaf, OpenReview, public repository, or server experiment changes were made.",
        ],
    }
    path = ROOT / "rebuttal/manuscript_revision_audit_20261003.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "abstract_words": len(abstract.split()), "report": str(path)}))


if __name__ == "__main__":
    main()
