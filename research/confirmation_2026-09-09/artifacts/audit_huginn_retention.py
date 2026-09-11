#!/usr/bin/env python3
"""Authenticate retained Huginn metadata and search for reconstruction inputs.

Does not deserialize any estimator, rank, activation, or model file.
"""
import os
from pathlib import Path
from recover_estimators import OUT, ROOT, OLD, LEASE, MON, read, record, same, guard, digest, atomic_json, now

review_path=OLD/'analysis/huginn_run01_review/actual_huginn_surviving_fit_review.json'
review=read(review_path)
owner_path=MON/'huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation/OWNER.json'
owner=read(owner_path); owner_rec=record(owner_path)
assert same(owner_rec,review['accepted_evaluation_owner']['record'])
assert digest(owner['identity'])==owner['identity_sha256']==review['accepted_evaluation_owner']['identity_sha256']
fit=owner['identity']['fit']; ident=fit['identity']; sha=digest(ident)
assert sha==fit['fit_identity_sha256']==review['fit_identity_sha256']
fitdir=LEASE/'retrieved/huginn/fit_01'; files={}
for name,key in [('OWNER.json','owner'),('LATEST.json','latest'),('COMPLETE.json','complete')]:
    actual=record(fitdir/name);assert same(actual,fit[key])
    files[name]=dict(actual=actual,expected=fit[key],status='authenticated')
assert read(fitdir/'OWNER.json')==dict(schema_version=1,fit_identity_sha256=sha,identity=ident)
assert read(fitdir/'LATEST.json')==fit['checkpoint_pointer']
assert read(fitdir/'COMPLETE.json')==fit['complete_pointer']
ptr=fit['checkpoint_pointer']; gen=fitdir/ptr['generation']; seal=read(gen/'SEAL.json')
assert same(record(gen/'SEAL.json'),ptr['seal'])
assert set(seal)=={'schema_version','kind','fit_identity_sha256','n_done','files'}
assert seal['schema_version']==1 and seal['fit_identity_sha256']==sha and seal['n_done']==100 and seal['kind']=='checkpoint'
assert set(seal['files'])=={'metadata.json','state.pt'}
metadata=read(gen/'metadata.json');assert same(record(gen/'metadata.json'),seal['files']['metadata.json'])
assert metadata['identity']==ident and metadata['fit_identity_sha256']==sha
assert metadata['n_done']==metadata['next_idx']==100
assert metadata['completed_prompt_sha256']==ident['prompt_sha256']
assert metadata['completed_prefix_sha256']==digest(ident['prompt_sha256'])
assert len(metadata['diagnostics'])==100
assert ident['d_model']==5280 and ident['source_layers']==list(range(32))
for i,row in enumerate(metadata['diagnostics']):
    for k,v in dict(index=i,prompt_sha256=ident['prompt_sha256'][i],token_length=ident['token_lengths'][i],n_valid=ident['n_valid'][i]).items():
        assert row[k]==v
for name in ['SEAL.json','metadata.json']:
    files[f'{ptr["generation"]}/{name}']=dict(actual=record(gen/name),status='authenticated')
inventory=read(OUT/'inventory.json'); key=f'huginn/fit_01/{ptr["generation"]}/state.pt'
assert inventory['binaries'][key]['expected']==seal['files']['state.pt']
partial=inventory['binaries'][key]['copies'];assert len(partial)==1 and partial[0]['status']=='truncated'
assert guard(partial[0]['path'])==partial[0]['actual']['stat_after']
finalkey=f'huginn/fit_01/{fit["complete_pointer"]["generation"]}/lens.pt'
assert inventory['binaries'][finalkey]['expected']==fit['lens']
for name in ['SEAL.json','metadata.json','lens.pt']:
    path=fitdir/fit['complete_pointer']['generation']/name
    assert not path.exists()
    files[f'{fit["complete_pointer"]["generation"]}/{name}']=dict(status='absent')

# Account for retained archives and older feature projects without importing them.
external=[]
for root in [Path('/tmp'),Path('/home/moloch/ouro_project')]:
    for current,dirs,names in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'venv','.venv','.git','node_modules'} and not (Path(current)/d).is_symlink()]
        for name in names:
            p=Path(current)/name
            if ('huginn' in str(p).lower() or 'cursor_000100_' in str(p)) and p.is_file():
                if p.is_symlink():
                    external.append(dict(path=str(p),symlink_target=os.readlink(p),
                        relevance='model/tokenizer snapshot link; not an N100 fitted Jacobian input'))
                    continue
                external.append(dict(path=str(p),stat=guard(p),
                    relevance='older July feature/probe project; not September N100 derivative contributions' if '/20260726' in str(p) or '_20260726' in str(p) else 'inspected path; no September N100 estimator reconstruction input'))
archives=[]
for current,dirs,names in os.walk(OLD):
    dirs[:]=[d for d in dirs if d!='.git' and not (Path(current)/d).is_symlink()]
    for name in names:
        if name.endswith(('.tar','.tar.gz','.tgz','.zip','.tar.zst')):
            p=Path(current)/name
            import tarfile,zipfile
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as z: members=z.namelist()
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as t: members=t.getnames()
            else:
                members=['unrecognized archive format']
            candidates=[m for m in members if m.endswith(('.pt','.pth','.bin','.npy','.npz')) or 'huginn' in m.lower()]
            archives.append(dict(path=str(p),stat=guard(p),member_count=len(members),candidate_members=candidates))
result=dict(schema='confirmation_huginn_retention.v1',recorded_utc=now(),status='metadata_authenticated_bank_unrecoverable',
    prior_review=dict(path=str(review_path),record=record(review_path)),evaluation_owner=dict(path=str(owner_path),record=owner_rec),
    fit_identity_sha256=sha,files=files,diagnostic_rows_checked=100,known_partial=partial[0],
    missing_checkpoint_bytes=seal['files']['state.pt']['bytes']-partial[0]['actual']['bytes'],
    expected_final=fit['lens'],external_search_roots=['/tmp','/home/moloch/ouro_project'],
    external_search_exclusions=['venv','.venv','.git','node_modules','symlink directories'],
    external_candidates=external,retained_archives=archives,
    reconstruction_decision='No sufficient retained inputs found. No full final bank, full FP32 sum, earlier September checkpoint, or per-paragraph derivative bank survives in searched locations. Saved evaluations and July feature caches do not supply missing derivative matrices.',
    partial_deserialized=False,remote_access=False,new_readout_outcomes=False)
atomic_json(OUT/'huginn_retention.json',result)
print(result['status'],len(external),'external candidates',len(archives),'archives inspected')
