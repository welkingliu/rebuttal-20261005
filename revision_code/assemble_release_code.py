"""Complete the isolated release source snapshot using hash-identical old files."""
import ast
import json
from pathlib import Path
import shutil
import subprocess

from common import ROOT, OLD, atomic_json, output_path, sha256


def main():
    folder=ROOT/"imported/anonymous_release_20260929"
    records=[]
    for line in (folder/"MANIFEST.sha256").read_text().splitlines():
        digest,name=line.split("  ",1)
        if not name.startswith(("scripts/","sgg_core/","tests/","configs/")):
            continue
        path=output_path(folder/name)
        if folder.resolve() not in path.parents:
            raise RuntimeError("Invalid manifest path")
        copied=False
        if not path.exists():
            source=OLD/name
            if not source.is_file() or sha256(source)!=digest:
                raise RuntimeError("No verified source for "+name)
            shutil.copy2(source,path)
            copied=True
        if sha256(path)!=digest:
            raise RuntimeError("Release checksum mismatch: "+name)
        if path.suffix==".py":
            ast.parse(path.read_text(),filename=str(path))
        elif path.suffix==".sh":
            subprocess.run(["bash","-n",str(path)],check=True)
        records.append(dict(path=name,sha256=digest,copied_from_identical_server_file=copied))
    report=dict(status="complete",files=len(records),records=records,
                scope="Only scripts, sgg_core, tests and configs; syntax/hash validation, not full runtime tests",
                old_assets_modified=False)
    atomic_json(ROOT/"results/anonymous_release_audit/assembled_code.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="records"}))


if __name__=="__main__":
    main()
