"""Evidence inventory for diagnostic topics, with explicit claim boundaries."""
import argparse
import json
import time

from common import ROOT, atomic_json, ensure_storage, sha256
from evidence_completion import read


ITEMS = [
    ("E01", "Distinct benchmark utility", ["VK_native_confirmation/summary.json", "R19_vk_fixed_order/sgdet/summary.json"],
     "Diagnosis-to-decision and model-dependent mechanisms; no demonstrated robot/agent task improvement."),
    ("E02", "Emerging open-vocabulary systems", ["R18_ovsgtr/formal/summary.json", "R18_ovsgtr/formal/identity_confidence_intervals.json"],
     "One direct open-vocabulary vision-language SGG case; not general MLLM coverage or a full official reproduction."),
    ("E03", "Intervention confounding", ["R12_sgdet_identity/tde_motifs/test/summary.json", "R15_transformer_identity/transformer/test/summary.json", "R20_plain_motifs_native_bridge/identity/summary.json"],
     "Fixed-evidence semantic/frequency routes; retain adverse and model-specific effects; no universal identity-error causality."),
    ("E04", "Practical mitigation", ["VK_native_confirmation/sgcls/decision.json", "VK_native_confirmation/sgdet/decision.json", "R19_vk_fixed_order/sgdet/summary.json", "R19_proposal_sensitivity/sgdet/summary.json"],
     "SGCls passes, SGDet does not; decomposition and sensitivity are post-hoc, not a new successful joint repair."),
    ("E05", "Identity-specific causal interpretation", ["R12_sgdet_identity/tde_motifs/test/summary.json", "R15_transformer_identity/transformer/test/summary.json", "R20_plain_motifs_native_bridge/identity/summary.json"],
     "Internal-channel interventions under frozen inputs; oracle labels only diagnose channels, never define a deployable method."),
    ("E06", "Probe and oracle-support limitations", ["probe_convergence/summary.json", "R19_proposal_sensitivity/sgcls/summary.json"],
     "Decoder/convergence sensitivity and complete-system evidence are separate. No claim of autonomous segmentation or information absence; human ambiguity audit absent."),
    ("E07", "Failed original live reproductions", ["R16_plain_motifs/predcls/formal/summary.json", "R16_plain_motifs/sgcls/formal/summary.json", "R16_plain_motifs/sgdet/formal/summary.json", "R14_reference_sgdet_test/summary.json", "R20_plain_motifs_native_bridge/identity/summary.json"],
     "Corrected task references and new plain-Motifs bridge; do not reuse old failing checkpoint effects or call training bitwise identical."),
    ("E08", "Actual original V objective", ["R17_original_v/summary.json"],
     "Original effective update is a 22,952-parameter object readout: scaled weighted CE plus Brier. Inactive relation terms must be corrected in text; later experiments do not retroactively fix it."),
    ("E09", "Low external coverage", ["R13_gqa_disjoint/summary.json"],
     "Disjoint exact-shared-label transfer with source denominators; low retained relation coverage persists. Not native full-ontology OOD evaluation."),
    ("E10", "Semantic versus visual removal", ["R20_plain_motifs_native_bridge/identity/summary.json", "R20_plain_motifs_native_bridge/visual/summary.json", "R15_transformer_identity/transformer/test/summary.json"],
     "Report identity-channel and spatial-image controls separately; annotation-relative unrelated objects are not guaranteed semantically irrelevant."),
    ("E11", "GCN-depth scope", ["probe_convergence/summary.json"],
     "Answered by narrowing I-B to its fixed-feature GCN-depth component protocol; these files are context, not evidence for other reasoning architectures."),
    ("E12", "Live architecture breadth", ["R6_sgtr_live/summary.json", "R18_ovsgtr/formal/summary.json", "R20_plain_motifs_native_bridge/visual/summary.json"],
     "Per-model capability coverage, not every model supporting every perturbation or training interface."),
    ("E13", "Calibration, ranking and reference alignment", ["R3/sgdet/summary.json", "R19_vk_fixed_order/sgdet/summary.json", "R19_proposal_sensitivity/sgdet/summary.json", "R20_capture_audit/summary.json"],
     "Frozen factorial outputs identify interacting channels, not an additive or universal failure explanation; original joint acceptance remains failed."),
]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--wait",action="store_true");a=p.parse_args()
    ensure_storage();out=ROOT/"results/R22_response_evidence";start=time.monotonic()
    if a.wait:
        while time.monotonic()-start<7200:
            queue=ROOT/"status/R20N_queue.json"
            if queue.exists() and read(queue).get("status") in ("complete","failed","blocked"):break
            atomic_json(out/"progress.json",dict(stage="waiting_corrected_motifs_bridge",seconds=time.monotonic()-start))
            time.sleep(15)
    rows=[]
    for item,title,paths,boundary in ITEMS:
        evidence=[]
        for name in paths:
            path=ROOT/"results"/name
            record=dict(path=str(path),present=path.is_file())
            if path.is_file():
                value=read(path)
                record.update(sha256=sha256(path),status=value.get("status"),accepted=value.get("accepted"),
                              images=value.get("images"),task=value.get("task"))
            evidence.append(record)
        rows.append(dict(evidence_item=item,topic=title,evidence=evidence,
            listed_files_present=all(e["present"] for e in evidence),required_claim_boundary=boundary,
            evidence_presence_is_not_unqualified_resolution=True))
    report=dict(status="complete",evidence_topics=len(rows),rows=rows,
        issues_with_all_listed_files=sum(r["listed_files_present"] for r in rows),
        unresolved=["No successful original two-task V-K mitigation", "No broad MLLM evaluation", "Low native external-ontology coverage remains",
                    "Original III paired raw records not recovered", "Public README/manifest mismatch unresolved", "No certified fresh 1000-image repair gate"],
        release_checks=[str(ROOT/"results/R21_release_execution/server_verified/summary.json")],
        next_action="Check evidence completeness and claim scope; file presence alone does not validate a conclusion",
        new_experiments_without_retraining=True)
    atomic_json(out/"summary.json",report)
    print(json.dumps(dict(status="complete",items=len(rows),listed_files_present=report["issues_with_all_listed_files"])),flush=True)


if __name__=="__main__":main()
