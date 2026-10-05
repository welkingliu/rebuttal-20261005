"""Audit immutable release code and preserve executable verification logs."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/R21_release_execution/source'
OUT = ROOT / 'output/revision_checks_20261005'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original = json.loads((ROOT/'output/rebuttal_results_20261002/R17_original_v/summary.json').read_text())
    hashes = []
    for name, expected in original['protocol']['source_sha256'].items():
        relative = name.split('/sgg_core/',1)
        local = SOURCE/'sgg_core'/relative[1] if len(relative)==2 else SOURCE/'scripts'/Path(name).name
        actual = digest(local)
        hashes.append(dict(path=str(local.relative_to(SOURCE)),expected=expected,actual=actual,match=actual==expected))
    if not all(r['match'] for r in hashes):
        raise RuntimeError('R17 source mismatch')
    graph = {}
    for p in (SOURCE/'sgg_core').rglob('*.py'):
        key='.'.join(p.relative_to(SOURCE).with_suffix('').parts)
        imports=[]
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.ImportFrom) and n.module:
                imports.append(n.module)
            elif isinstance(n,ast.Import):
                imports.extend(x.name for x in n.names)
        graph[key]=imports
    closure={}
    for exp in ['experiment_1a','experiment_1b','experiment_2','experiment_3','experiment_4','experiment_5']:
        todo=['sgg_core.experiments.'+exp];seen=set()
        while todo:
            node=todo.pop()
            if node in seen:continue
            seen.add(node);todo.extend(n for n in graph.get(node,[]) if n in graph)
        closure[exp]=dict(modules=sorted(seen),imports_v_optimizer='sgg_core.mitigation.run_mitigation' in seen)
    env=dict(os.environ,PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1')
    commands=[('release_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py']),
              ('readout_scope_tests',[sys.executable,str(ROOT/'rebuttal/test_v_scope_20261005.py')])]
    commands += [(x+'_help',[sys.executable,'-m','sgg_core.experiments.'+x,'--help']) for x in closure]
    outcomes=[]
    for name, command in commands:
        p=OUT/(name+'.log')
        with p.open('w') as f:
            result=subprocess.run(command,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        outcomes.append(dict(name=name,returncode=result.returncode,log=p.name,sha256=digest(p)))
        print(name,result.returncode,flush=True)
    report=dict(status='complete',source_hashes=hashes,static_import_closure=closure,checks=outcomes,
        all_checks_passed=all(r['returncode']==0 for r in outcomes),
        impact={
            'V_training':'Effective object-readout objective must be corrected; native visual/context/relation modules were not trained.',
            'V_test_and_external':'Use the updated readout; numerical outputs retained as readout calibration, not grounding-aware training.',
            'I_A_I_B':'No V optimizer in static import closure; separate feature/probe and relation-depth fitting. This is not an independent rerun of every historical asset.',
            'II_III_IV':'No V optimizer in entry-point static closure. Shared dynamic adapter exists. Synthetic identity initialization and zero cross-gradient tests pass; historical manifest/checkpoint auditing remains run-specific.',
            'III':'Historical paired raw records unavailable; withdraw historical paired CI and matched-control claims from the current evidence set.',
            'release':'Earlier R21 failure remains archived; current source tests and six help commands do not establish a fresh full-data reproduction.'},
        gradient_evidence=dict(records=original['records'],max_gradient_error=original['max_gradient_equivalence_error']),
        limitations=['Static import traversal does not resolve dynamic imports.','Synthetic adapter tests are not native CUDA inference.','No universal claim that all prior numerical results are unaffected.'])
    (OUT/'v_scope_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_checks_passed']:raise RuntimeError('Verification failed')


if __name__=='__main__':main()
