"""Download a read-only, hash-audited subset of the anonymous paper release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = "https://anonymous.4open.science/r/kdd2027-submission-v1-CA7D/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code_inventory", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "-fsSL", "--retry", "2", "--connect-timeout", "15",
                            "--max-time", "90", BASE + name, "-o", str(tmp)], check=True)
            tmp.replace(path)
        return path

    manifest = fetch("MANIFEST.sha256")
    entries = dict(line.split("  ", 1)[::-1] for line in manifest.read_text().splitlines())
    selected = [p for p in entries if p.startswith(("results/reported/experiment_2/",
                                                   "results/reported/experiment_3/"))]
    selected += ["README.md", "REPRODUCIBILITY.md", "results/RESULT_INVENTORY.json",
                 "sgg_core/experiments/experiment_2.py", "sgg_core/experiments/experiment_3.py",
                 "sgg_core/audits/pair_audit.py", "sgg_core/audits/graph_audit.py",
                 "sgg_core/models/adapters/pysgg_live.py", "scripts/pysgg_live_worker.py"]
    if args.code_inventory:
        differences=json.loads(args.code_inventory.read_text())["files"]
        selected += [r["path"] for r in differences if r["status"] in ("missing","different")]
    selected=list(dict.fromkeys(selected))
    rows = []
    for name in selected:
        path = fetch(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = dict(path=name, bytes=path.stat().st_size, sha256=digest,
                   manifest_match=digest == entries[name])
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            row["top_level_keys"] = list(payload) if isinstance(payload, dict) else []
            row["record_candidates"] = []
            def inspect(obj, location=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        loc = location + "/" + key
                        if key.lower() in ("per_image", "records", "image_records", "image_ids", "samples"):
                            row["record_candidates"].append(dict(path=loc, length=len(value) if hasattr(value, "__len__") else None))
                        inspect(value, loc)
                elif isinstance(obj, list):
                    for i, value in enumerate(obj[:2]):
                        inspect(value, location + "/" + str(i))
            inspect(payload)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(source=BASE, retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  manifest_entries=len(entries), downloaded=rows,
                  historical_result_paths=[p for p in entries if p.startswith("results/")],
                  status="complete", all_hashes_match=all(r["manifest_match"] for r in rows))
    (args.output / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
