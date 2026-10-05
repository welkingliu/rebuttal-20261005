"""Read-only inventory reconciliation against the pre-deletion migration record."""
import collections
import json
import os
from pathlib import Path
import time

from common import ROOT, OLD, ensure_storage, atomic_json, sha256


def main():
    ensure_storage()
    out=ROOT/"results/storage_audit_20260929"
    migration=ROOT.parent/"_migration/20260905_lch"
    inventory=migration/"inventory_before_original_deletion.jsonl"
    checked=0;missing=[];changed=[];errors=[];counts=collections.Counter();started=time.monotonic()
    for line in inventory.open():
        record=json.loads(line)
        relative=Path(record["path"])
        if relative.parts[0]!=OLD.name or record["type"]!="file":
            continue
        path=ROOT.parent/relative
        checked+=1
        category="/".join(relative.parts[1:3])
        counts[category]+=1
        try:
            stat=path.stat()
            if stat.st_size!=record["size"]:
                changed.append(dict(path=str(path),previous_size=record["size"],current_size=stat.st_size))
        except FileNotFoundError:
            missing.append(dict(path=str(path),previous_size=record["size"]))
        except OSError as exc:
            errors.append(dict(path=str(path),error=str(exc)))
    candidates=[];trees={};archives=[]
    for folder in [OLD/"artifacts",OLD/"checkpoints",ROOT/"imported"]:
        groups=collections.defaultdict(lambda:dict(files=0,bytes=0,examples=[]))
        for directory,_,names in os.walk(folder):
            for name in names:
                p=Path(directory)/name
                try:s=p.stat()
                except OSError:continue
                rel=p.relative_to(folder);group=rel.parts[0]
                g=groups[group];g["files"]+=1;g["bytes"]+=s.st_size
                if len(g["examples"])<4:g["examples"].append(str(p))
                text=str(rel).lower()
                if any(t in text for t in ["experiment_2","experiment_3","exp2","exp3","intervention","motif_persist"]):
                    candidates.append(dict(path=str(p),bytes=s.st_size))
                if name.endswith((".tar.gz",".zip",".tar.zst",".tgz",".tar")):
                    archives.append(dict(path=str(p),bytes=s.st_size))
        trees[str(folder)]=dict(groups)
    result=dict(status="complete",checked_migration_files=checked,missing_count=len(missing),
        size_changed_count=len(changed),stat_errors=len(errors),missing=missing,size_changed=changed,
        errors=errors,categories=dict(counts),current_trees=trees,candidate_records=candidates,
        archives=archives,inventory=str(inventory),inventory_sha256=sha256(inventory),
        historical_checksum_difference_bytes=(migration/"kdd_sgg_core_experiments.checksum_differences.log").stat().st_size,
        limitations="Current existence/size reconciliation, not a new full-content checksum of every asset; only files in the original migrated lch tree are covered.",
        actions="read-only old assets; no deletions, moves or overwrites",seconds=time.monotonic()-started)
    atomic_json(out/"summary.json",result)
    print(json.dumps({k:result[k] for k in ["status","checked_migration_files","missing_count","size_changed_count","stat_errors","seconds"]}),flush=True)


if __name__=="__main__":main()
