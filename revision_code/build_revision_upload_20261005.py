"""Build local upload staging, preserving old assets and no network publication."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'uploads/20261005'
SOURCE = ROOT/'output/R21_release_execution/source'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def public_text(s):
    return s.replace('/path/to/revision_workspace','/path/to/revision_workspace').replace('/path/to/local_user','/path/to/local_user').replace('/home/USER','/home/USER').replace('/mnt/EXPERIMENT_DISK','/mnt/EXPERIMENT_DISK').replace('SERVER_HOST','SERVER_HOST')


def copy_text(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(public_text(source.read_text()))


def main():
    git=DEST/'github';drive=DEST/'onedrive';paper=DEST/'manuscript_for_overleaf'
    for d in [git,drive,paper]:d.mkdir(parents=True,exist_ok=True)
    for folder in ['sgg_core','scripts','tests','configs']:
        for p in (SOURCE/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix in {'.py','.sh','.json','.yaml','.yml','.txt','.toml'}:
                copy_text(p,git/p.relative_to(SOURCE))
    for name in ['LICENSE','pyproject.toml','requirements.txt','requirements-foundation.txt','THIRD_PARTY_ASSETS.md','ASSET_SETUP.md','DATASETS.md','FOUNDATION_MODELS.md','INSTALLATION.md','LEGACY_MODEL_SETUP.md','MODELS.md']:
        copy_text(SOURCE/name,git/name)
    # Historical instructions are explicitly separated from current release notes.
    for name in ['REPRODUCIBILITY.md','RUNNING.md']:
        copy_text(SOURCE/name,git/'historical_docs'/name)
    for p in (ROOT/'rebuttal').glob('*.py'):
        copy_text(p,git/'revision_code'/p.name)
    evidence=[ROOT/'output/revision_checks_20261005'/name for name in [
        'paired_channels.json','plain_motifs_sgcls_counts.json','transformer_sgdet_counts.json',
        'sam_history_audit.json','sam_bounded_summary.json','v_scope_audit.json']]
    evidence += [ROOT/'output/rebuttal_results_20261002/R17_original_v/summary.json',
                 ROOT/'output/evidence_completion_20261003/R21_release_execution/server_verified/summary.json',
                 ROOT/'rebuttal/VX_summary_20261005.json']
    index=[]
    for i,p in enumerate(evidence):
        name=p.name if p.parent.name=='revision_checks_20261005' else p.parent.name+'_'+p.name
        if p.name=='VX_summary_20261005.json':name=p.name
        target=git/'evidence'/name
        copy_text(p,target)
        index.append(dict(path=str(target.relative_to(git)),original_sha256=sha(p),
                          public_sha256=sha(target),path_redaction=True))
    (git/'evidence/PROVENANCE.json').write_text(json.dumps(index,indent=2)+'\n')
    shutil.copy2('/path/to/local_user/Downloads/main.tex',paper/'main.tex')
    assets=ROOT/'rebuttal/manuscript_revision_20261003'
    shutil.copy2(assets/'reference.bib',paper/'reference.bib')
    shutil.copytree(assets/'figures',paper/'figures',dirs_exist_ok=True)
    # NPZ records are paired-analysis evidence only, not original images/weights.
    archive=drive/'paired_channel_records_20261005.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for family,folder in [('motifs_sgcls',ROOT/'output/evidence_completion_20261003/R20_plain_motifs_native_bridge/identity/images'),
                              ('transformer_sgdet',ROOT/'output/revision_checks_20261005/transformer/images')]:
            for p in sorted(folder.glob('*.npz')):z.write(p,family+'/'+p.name)
        for p in (git/'evidence').glob('*.json'):z.write(p,'evidence/'+p.name)
    (drive/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(status='staged_not_uploaded',root=str(DEST),evidence_archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
