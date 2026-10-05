"""Portable isolated-source test runner with machine-readable failure records."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args(); source=a.source.resolve(); out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    originals={name:digest for digest,name in (line.split("  ",1) for line in a.manifest.read_text().splitlines())}
    inventory=[]
    for name,digest in originals.items():
        if not name.startswith(("sgg_core/","scripts/","tests/","configs/")) and len(Path(name).parts)>1: continue
        file=source/name
        actual=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        inventory.append(dict(path=name,expected=digest,actual=actual,match=actual==digest))
        if name.startswith(("sgg_core/","scripts/","tests/","configs/")) and actual!=digest:
            raise RuntimeError("Submitted release source mismatch: "+name)
    os.chdir(source); sys.path[0]=str(source)
    import sgg_core
    if source not in Path(sgg_core.__file__).resolve().parents: raise RuntimeError("Wrong imported package origin")
    start=time.monotonic(); suite=unittest.defaultTestLoader.discover(str(source/"tests"),pattern="test_*.py")
    with (out/"unit_suite.log").open("w") as handle:
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    packages={}
    for name in ("torch","torchvision","numpy","h5py","pillow","transformers","timm","setuptools"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    report=dict(status="complete",passed=result.wasSuccessful(),tests=result.testsRun,
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures],
        errors=[dict(test=str(test),traceback=trace) for test,trace in result.errors],
        skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
        python=sys.version,executable=sys.executable,packages=packages,source=str(source),
        inventory=inventory,source_isolated=True,dependencies_reused=True,clean_install_claim=False,
        elapsed_seconds=time.monotonic()-start)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("inventory","errors","failures")}),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=="__main__": main()
