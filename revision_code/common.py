"""Disk-scoped output and immutable provenance for the rebuttal experiments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(os.environ.get("SGG_REBUTTAL_ROOT", "/mnt/EXPERIMENT_DISK/kdd_sgg_rebuttal_20260929"))
OLD = Path(os.environ.get("SGG_OLD_ROOT", "/mnt/EXPERIMENT_DISK/kdd_sgg_core_experiments"))


def ensure_storage():
    uuid = subprocess.check_output(
        ["findmnt", "-n", "-o", "UUID", "-T", str(ROOT.parent)], text=True
    ).strip()
    if uuid != "401091e4-9c20-4043-a467-91727ed55b56":
        raise RuntimeError("Wrong experiment disk UUID")
    if json.loads((ROOT.parent / "_migration/latest/status.json").read_text())["phase"] != "complete_verified":
        raise RuntimeError("Migration has not passed verification")


def output_path(path):
    path = Path(path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("New outputs must be inside %s: %s" % (ROOT, path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, payload):
    path = output_path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_path(path):
    # Historical records are never rewritten in place.
    value = str(path).replace("/home/USER/desktop/lch/kdd_sgg_core_experiments", str(OLD))
    return Path(value)


def sole_match(pattern):
    matches = sorted(OLD.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one asset for %s; found %s" % (pattern, matches))
    return matches[0]
