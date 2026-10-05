"""Independent native confirmation of the already frozen V-K primary candidate."""
import json
from pathlib import Path

from common import ROOT, OLD, atomic_json, sha256, ensure_storage

HERE = Path(__file__).resolve().parent
PARENT = ROOT / "results/VK_mac_calibration_shift"
OUT = ROOT / "results/VK_native_confirmation"
CACHE = ROOT / "cache/VK_native_confirmation"
MANIFEST = ROOT / "manifests/VK_native_protocol.json"
CHECKPOINT = PARENT / "selected.pth"
FIXTURE = ROOT / "transfers/VK_native/router_parity.pt"
PRIMARY_SHA = "573d4fe98e2003b778aeec0edf910e0e64dda9881ca635d862d57058eddf1bd1"
ORIGINAL_REGISTRATION = "7ae9af0dc87c256649ffa0463212c9d4360d6021a5ed4a5e1d0cedfe4a6a4328"
REPAIR_RECORD = OUT / "runtime_repair_20261003/repair_record.json"
SPEC = dict(version="vk_native_confirmation_v1_runtime_fix1", seed=17, tasks=["sgcls", "sgdet"],
    gate_images=1000, smoke_images=3, image_batch=1, roi_chunk=256, crop_batch=8,
    precision="fp32; no AMP", encoder="frozen DINOv2-B normalized CLS, same VH crop recipe",
    update="frozen VK primary,14 features,alpha0.5,selection>=0.6; no fitting or task-specific tuning",
    support="SGCls supplied GT boxes; SGDet native detected proposal boxes",
    endpoint="post-NMS identity on matched native proposals, plus native image-ranked R/mR",
    relation="predicate/context logits fixed; actual object scores, class-box NMS and triplet ranking rerun",
    stop="SGCls rejection stops SGDet; any runtime/parity failure stops; no auto threshold/seed/test expansion",
    scope="held out from adapter fitting/selection, not from historical native validation audits",
    execution_hours=12, waiting_hours=72, gpu_preference=[1, 0],
    native_backend=dict(cudnn_benchmark=False, cudnn_deterministic=False,
                        cudnn_allow_tf32=True, matmul_allow_tf32=True),
    queue="exclusive first idle GPU after its current R16 queue ends; never interrupt or share GPU")
SOURCES = ["vk_gate_protocol.py", "vk_gate_math.py", "vk_gate_native.py", "vk_gate_features.py",
    "vk_gate_queue.py", "vj_mac.py", "vi_native.py", "vf_native.py", "ve_experiment.py", "ve_native.py",
    "ve_protocol.py", "native_runtime.py", "repro_experiment.py", "repro_protocol.py", "common.py",
    "vb_native.py", "vb_protocol.py", "vc_protocol.py", "reproduction_pair_audit.py",
    "independent_identity_expert.py", "storage_runtime.sh", "test_vk_gate.py",
    "prepare_vk_runtime_fixture.py", "verify_vk_runtime.py", "VK_NATIVE_PLAN.md",
    "audit_vk_baseline.py", "audit_vk_crop_backend.py"]


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise RuntimeError("Frozen V-K native record changed: " + str(path))
    if not path.exists(): atomic_json(path, value)


def runtime_repair_inputs():
    record = read(REPAIR_RECORD)
    original_path = Path(record["original_manifest"])
    if (record["original_manifest_sha256"] != ORIGINAL_REGISTRATION
            or sha256(original_path) != ORIGINAL_REGISTRATION or record["gate_evaluated"]):
        raise RuntimeError("Missing original V-K runtime-failure provenance")
    original = read(original_path)
    if original["checkpoint_sha256"] != PRIMARY_SHA:
        raise RuntimeError("Runtime repair changed the selected candidate")
    expected = dict(original["spec"], version=SPEC["version"], native_backend=SPEC["native_backend"])
    if expected != SPEC:
        raise RuntimeError("Runtime repair changed the scientific protocol")
    files = [REPAIR_RECORD, original_path]
    for task in SPEC["tasks"] + ["crops"]:
        path = Path(record["audits"][task]); audit = read(path)
        if (audit["status"] != "complete" or audit["gate_evaluated"]
                or audit["original_registration_sha256"] != ORIGINAL_REGISTRATION
                or audit["smoke_ids"] != original["smoke_ids"]):
            raise RuntimeError("Invalid old-development runtime audit: " + task)
        if task != "crops":
            if (audit["checkpoint_sha256"] != original["baseline_assets"][task]["sha256"]
                    or not audit["statistics_match"] or set(audit["config_differences"]) - {"OUTPUT_DIR"}):
                raise RuntimeError("Native checkpoint/configuration changed")
            rows = [r for r in audit["comparisons"] if r["cudnn_deterministic"] is False]
            if (len(rows) != 3 or {r["image_id"] for r in rows} != set(original["smoke_ids"])
                    or not all(r["passed"] and all(v["max_error"] == 0 for v in r["errors"].values()) for r in rows)):
                raise RuntimeError("Native reference-cache parity not restored")
        elif not all(r["passed"] and r["max_error"] == 0 for r in audit["comparisons"]):
            raise RuntimeError("Frozen crop evidence changed")
        files.append(path)
    return original, files


def register():
    ensure_storage()
    original, repair_inputs = runtime_repair_inputs()
    parent, summary = read(PARENT / "protocol.json"), read(PARENT / "summary.json")
    readiness = read(PARENT / "native_confirmation_readiness.json")
    if (sha256(CHECKPOINT) != PRIMARY_SHA or summary["full_checkpoint_sha256"] != PRIMARY_SHA
            or sha256(PARENT / "protocol.json") != summary["protocol_sha256"]
            or not summary["screening_eligible"] or summary["heldout_confirmation_evaluated"]):
        raise RuntimeError("Frozen V-K candidate not eligible")
    if sha256(HERE / "vj_mac.py") != parent["helper_sha256"]:
        raise RuntimeError("Frozen V-J math changed")
    split = read(ROOT / "results/VI_selective_fusion/splits.json")
    gate = parent["provenance"]["reserved_gate_ids"]
    if (len(gate) != 1000 or len(set(gate)) != 1000 or set(gate) != set(split["gate"])
            or set(gate) & set(split["train"] + split["development"])):
        raise RuntimeError("Reserved adaptation gate changed")
    inputs = repair_inputs + [CHECKPOINT, FIXTURE, PARENT / "protocol.json", PARENT / "summary.json",
        ROOT / "results/VI_selective_fusion/splits.json", ROOT / "results/VH_independent_identity/protocol.json",
        OLD / "data/vg/v1.4/VG-SGG-dicts.json", OLD / "data/vg/v1.4/VG-SGG.h5"]
    for name in ["native_cpu", "modern_cpu"]:
        path = OUT / "preflight" / (name + ".json")
        result = read(path)
        if (result.get("status") != "complete" or result.get("cuda_initialized")
                or result["checkpoint_sha256"] != PRIMARY_SHA or result["fixture_sha256"] != sha256(FIXTURE)):
            raise RuntimeError("Cross-runtime CPU preflight failed")
        inputs.append(path)
    for task in SPEC["tasks"]:
        for path in [ROOT / "cache" / n / task / "protocol.json" for n in ["VB", "VC"]] + [ROOT / "results/VE" / task / "splits.json"]:
            previous = read(path)
            if set(gate) & set(previous["development"] + previous["gate"]):
                raise RuntimeError("Reserved images used by a previous adapter")
            inputs.append(path)
        for name in ["VF", "VI_selective_fusion"]:
            if any((ROOT / "results" / name / task / "gate").rglob("*.json")):
                raise RuntimeError("Prior reserved gate has been consumed")
    inputs += [Path(v["path"]) for v in readiness["assets"].values()]
    for value in readiness["assets"].values():
        if sha256(Path(value["path"])) != value["sha256"]: raise RuntimeError("Readiness asset changed")
    vh = read(ROOT / "results/VH_independent_identity/protocol.json")
    repo = OLD / "external/foundation_repos/dinov2"
    inputs += [repo / name for name in vh["encoder_sources"]] + [repo / "hubconf.py"]
    # Freeze the external inference implementation as well as the wrapper.
    inputs += sorted((OLD / "external/official_repos/PySGG/pysgg").rglob("*.py"))
    from native_runtime import assets
    for task in SPEC["tasks"]:
        assets_ = assets("transformer", task)
        inputs += [assets_[3]]
        if task == "sgcls": inputs += [assets_[1].parent / "config.yml"]
    inputs += [assets("transformer", "sgdet")[1].parent / "VG_stanford_filtered_with_attribute_train_statistics.cache"]
    from vc_protocol import SPEC as reference
    gate_spec = parent["spec"]["gate"]
    for key in ["object_gain_min", "object_bootstrap_lower_min", "relation_noninferiority_margin",
                "bootstrap_samples", "lower_quantile", "metrics", "minimum_positive_objects", "requires_both_tasks"]:
        if gate_spec[key] != reference["gate"][key]: raise RuntimeError("Acceptance gate changed")
    value = dict(spec=SPEC, gate=gate_spec, parent_protocol_sha256=summary["protocol_sha256"],
        checkpoint_sha256=PRIMARY_SHA, gate_ids=gate, smoke_ids=split["development"][:3],
        sources={n: sha256(HERE / n) for n in SOURCES},
        inputs={str(p): sha256(p) for p in sorted(set(inputs))},
        baseline_assets=readiness["assets"], supersedes_registration_sha256=ORIGINAL_REGISTRATION)
    for field in ["gate", "parent_protocol_sha256", "checkpoint_sha256", "gate_ids", "smoke_ids", "baseline_assets"]:
        if value[field] != original[field]:
            raise RuntimeError("Runtime repair changed a frozen experiment field: " + field)
    if MANIFEST.exists() and sha256(MANIFEST) == ORIGINAL_REGISTRATION:
        # The failed attempt has been archived before replacing its registration.
        for task in SPEC["tasks"]:
            if any((OUT / task / "gate").rglob("*")) or any((CACHE / task / "gate").rglob("*")) or (OUT / task / "decision.json").exists():
                raise RuntimeError("Cannot amend a protocol after gate execution")
        atomic_json(MANIFEST, value)
        return sha256(MANIFEST)
    immutable(MANIFEST, value)
    return sha256(MANIFEST)


def verify(full=False):
    value = read(MANIFEST)
    if value["spec"] != SPEC or value["checkpoint_sha256"] != PRIMARY_SHA:
        raise RuntimeError("V-K native protocol changed")
    for name, digest in value["sources"].items():
        if sha256(HERE / name) != digest: raise RuntimeError("V-K native source changed: " + name)
    if full:
        for path, digest in value["inputs"].items():
            if sha256(Path(path)) != digest: raise RuntimeError("V-K native input changed: " + path)
    return sha256(MANIFEST)


def paths(task, smoke=False):
    return OUT / task / ("smoke" if smoke else "gate"), CACHE / task / ("smoke" if smoke else "gate")


def eligible(task, smoke=False):
    verify()
    if task == "sgdet" and not read(OUT / "sgcls/decision.json")["accepted"]:
        raise RuntimeError("SGCls joint gate did not permit SGDet")
    if not smoke and read(OUT / task / "smoke/selective/summary.json").get("status") != "complete":
        raise RuntimeError("Native crop/router/postprocessor smoke has not passed")


def ids(smoke=False):
    value = read(MANIFEST)
    return value["smoke_ids"] if smoke else value["gate_ids"]
