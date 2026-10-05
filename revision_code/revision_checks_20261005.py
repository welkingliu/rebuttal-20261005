"""Read-only paired channel analysis and static Experiment V scope inventory."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from identity_evidence import paired_micro

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/revision_checks_20261005"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    sources = {
        "plain_motifs_sgcls": ROOT / "output/evidence_completion_20261003/R20_plain_motifs_native_bridge/identity",
        "transformer_sgdet": OUT / "transformer",
    }
    results = {}
    for model, folder in sources.items():
        print("[RUNNING] paired channels: " + model, flush=True)
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("status") != "complete" or not summary.get("invariance_passed"):
            raise RuntimeError("Incomplete or non-invariant source: " + model)
        rows, hashes = [], {}
        files = sorted((folder / "images").glob("*.npz"))
        for file in files:
            with np.load(file, allow_pickle=False) as p:
                gt = p["relation_gt"]
                sem, freq = p["prediction_oracle"], p["prediction_oracle_frequency"]
                if gt.shape != sem.shape or gt.shape != freq.shape:
                    raise RuntimeError("Paired shape mismatch")
                rows.append([int((freq == gt).sum()) - int((sem == gt).sum()), len(gt)])
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        if len(files) != 2000:
            raise RuntimeError("Expected 2000 source images")
        result = paired_micro(rows, seed=17, trials=10000)
        results[model] = dict(contrast="oracle_frequency minus oracle_semantic; same eligible pairs",
                             scope="post-hoc descriptive comparison; not a deployable oracle or cross-model ranking",
                             statistic=result, image_count=len(files))
        save(model + "_counts.json", dict(image_ids=[f.stem for f in files], counts=rows, source_sha256=hashes))
        print("[COMPLETE] " + model + " " + json.dumps(result), flush=True)
    save("paired_channels.json", results)
    p = ROOT / "output/rebuttal_results_20261001/probe_convergence/sam_vit_b/summary.json"
    sam = json.loads(p.read_text())
    rows = []
    for run in sam["runs"]:
        h = run["history"]
        loss = np.array([r["validation_cross_entropy"] for r in h])
        rows.append(dict(seed=run["seed"], architecture=run["architecture"], epochs=len(h),
                         best_epoch=run["best_epoch"], best_nll=float(loss.min()),
                         last_nll=float(loss[-1]), last50_best_improvement=float(loss[:-50].min()-loss.min()) if len(loss)>50 else None,
                         last50_slope=float(np.polyfit(np.arange(min(50,len(loss))),loss[-50:],1)[0]),
                         reached_cap=run["reached_epoch_cap"]))
    save("sam_history_audit.json", rows)
    print("[COMPLETE] SAM history audit " + json.dumps(rows), flush=True)
    source = ROOT / "output/R21_release_execution/source/sgg_core"
    inventory = []
    for file in sorted(source.rglob("*.py")):
        text = file.read_text()
        tree = ast.parse(text)
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and "mitigation" in n.module]
        calls = sorted({n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in {"load_grounding_state_dict", "forward_grounding", "grounding_parameters"}})
        if imports or calls:
            inventory.append(dict(path=str(file.relative_to(source)), imports=imports, calls=calls,
                                  sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    save("v_scope_static_inventory.json", dict(records=inventory,
         status="static_inventory_only", limitations="Dynamic adapters and checkpoint provenance require separate review; absence of a direct import does not prove independence."))
    print("[COMPLETE] static dependency inventory; dynamic scope review remains", flush=True)
    save("status.json", dict(status="analysis_complete", pending=["dynamic scope review", "SAM bounded refit", "release consistency and manuscript edits"]))


if __name__ == "__main__":
    main()
