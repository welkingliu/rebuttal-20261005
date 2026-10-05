"""Resolve inputs and output destinations against actual mounts before R9."""
import json
import os
from pathlib import Path
import subprocess

from common import OLD, ROOT, atomic_json, ensure_storage
from native_runtime import assets
from repro_protocol import RUN

EXPECTED = "401091e4-9c20-4043-a467-91727ed55b56"


def mount(path):
    current = path
    while not current.exists():
        current = current.parent
    return subprocess.check_output(["findmnt", "-n", "-o", "SOURCE,UUID,TARGET", "-T", str(current)], text=True).strip()


def entry(role, path):
    path = Path(path)
    resolved = path.resolve()
    return dict(role=role, declared=str(path), resolved=str(resolved), exists=resolved.exists(), mount=mount(resolved))


def main():
    ensure_storage()
    records = [entry("old_asset_root",OLD),entry("new_output_root",ROOT),
               entry("historical_compatibility_alias","/home/USER/desktop/lch")]
    failures = []
    repo = OLD / "external/official_repos/PySGG"
    for path in [OLD / "checkpoints/sgg/weights/pysgg/vg/shared_detector.pth",
                 assets("transformer","sgcls")[1], OLD / "data/derived/glove/glove.6B.200d.pt",
                 repo/"datasets/vg/VG-SGG-with-attri.h5", repo/"datasets/vg/VG-SGG-dicts-with-attri.json",
                 repo/"datasets/vg/image_data.json", repo/"datasets/vg/stanford_spilt/VG_100k_images"]:
        record=entry("read_only_input",path);records.append(record)
        if not record["exists"]:
            failures.append("Missing input: "+str(path))
    outputs=[ROOT / x for x in ("logs","status","results","checkpoints","cache","tmp","manifests")]
    outputs += [RUN]
    for name in ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","MPLCONFIGDIR","TORCH_HOME",
                 "HF_HOME","TORCH_EXTENSIONS_DIR","CUDA_CACHE_PATH"):
        if not os.environ.get(name):
            failures.append("Unset output environment: "+name)
        else:
            outputs.append(Path(os.environ[name]))
    jobs=json.loads((ROOT/"manifests/R9_jobs.json").read_text())["jobs"]
    for job in jobs:
        outputs.extend(Path(job[k]) for k in ("completion","progress_file"))
    for path in sorted(set(outputs)):
        record=entry("output",path);records.append(record)
        resolved=Path(record["resolved"])
        if ROOT.resolve() not in (resolved,*resolved.parents) or EXPECTED not in record["mount"]:
            failures.append("Output escapes registered disk/workspace: "+str(path))
    for path in RUN.rglob("*"):
        if path.is_symlink() and ROOT.resolve() not in path.resolve().parents:
            failures.append("R9 symlink escapes workspace: "+str(path))
    for path in RUN.rglob("model_provenance.json"):
        import yaml
        cfg=yaml.safe_load(json.loads(path.read_text())["config"])
        value=Path(cfg["OUTPUT_DIR"]).resolve()
        if ROOT.resolve() not in value.parents or EXPECTED not in mount(value):
            failures.append("Saved model OUTPUT_DIR escapes workspace: "+str(path))
    report=dict(status="complete" if not failures else "failed", records=records, failures=failures,
                scope="R9 resolved input roots, output roots, job paths, runtime cache variables and saved configs",
                note="Old inputs and new outputs are sibling directories on the verified migrated disk; Python environments remain on the system disk. No file deletion or migration performed.")
    atomic_json(ROOT/"results/R9_path_audit/summary.json",report)
    for record in records:
        print("[%s] %s -> %s | %s" % (record["role"],record["declared"],record["resolved"],record["mount"]))
    print("[PATHS READY]" if not failures else "[PATHS FAILED] " + repr(failures),flush=True)
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
