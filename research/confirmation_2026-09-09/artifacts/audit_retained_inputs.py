#!/usr/bin/env python3
"""Supplement recovery with upstream text, downstream bindings and saved-array integrity.

Reads only completed historical outputs; performs no model inference or new readout.
"""
import gc
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from recover_estimators import OUT, ROOT, OLD, MON, LEASE, read, record, guard, same, digest, atomic_json, now

sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD/'deployment'))
import evaluate_refits as common
import evaluate_controls as controls
import evaluate_huginn as huginn
torch.set_num_threads(2);torch.set_num_interop_threads(1)
inv=read(OUT/'inventory.json');ledger=read(OUT/'RECOVERY_LEDGER.json')
mainroot=MON/'ouro_handoff_20260909T110503Z/results/ouro_evaluation'
controlroot=mainroot.parent/'controls_evaluation'
hroot=MON/'huginn_handoff_20260909T153351Z_659a482c/results/huginn_evaluation'
oldmain=MON/'main_only_20260909T071754Z/results/ouro_evaluation'
result=dict(schema='confirmation_retained_input_supplement.v1',started_utc=now(),
            prior_ledger_record=record(OUT/'RECOVERY_LEDGER.json'),upstream={},downstream={},evaluation_roots={},files={})

hash_cache={}
def tracked(path,expected=None):
    path=Path(path).resolve()
    if str(path) not in hash_cache: hash_cache[str(path)]=record(path)
    actual=hash_cache[str(path)]
    assert guard(path)==actual['stat_after']
    if expected is not None: assert same(actual,expected),(path,actual,expected)
    return actual

# Actual local calibration bytes and each ordered UTF-8 paragraph, independent of diagnostics.
identities={fit:p['identity'] for fit,p in inv['provenance'].items()}
identities['huginn/fit_01']=read(LEASE/'retrieved/huginn/fit_01/OWNER.json')['identity']
for fit,ident in identities.items():
    calibration=ident['runtime']['calibration']; origin=calibration.get('origin',calibration)
    textpath=(OLD/'deployment'/origin['prompts_path']).resolve();actual=tracked(textpath)
    assert actual['sha256']==origin['sha256']
    paragraphs=read(textpath)
    assert type(paragraphs) is list and len(paragraphs)==100 and all(type(t) is str for t in paragraphs)
    assert digest(paragraphs)==ident['ordered_prompts_sha256']
    hashes=[hashlib.sha256(t.encode('utf-8')).hexdigest() for t in paragraphs]
    assert hashes==ident['prompt_sha256']
    row=dict(status='passed',calibration_path=str(textpath),actual=actual,expected_sha256=origin['sha256'],
             paragraph_count=100,all_ordered_paragraph_hashes_match=True,ordered_prompts_sha256=digest(paragraphs),
             token_length_and_n_valid_scope='sealed declarations and completed diagnostics; this supplement does not retokenize')
    inputs=[]
    for item in ident['runtime']['source_files']:
        if item['root']=='jlens' and (item['path'].startswith('data/evaluations/') or item['path'].endswith(('run_spec.json','combined_contract.json','huginn_calibration.json'))):
            p=ROOT/item['path'];rec=tracked(p);assert rec['sha256']==item['sha256']
            inputs.append(dict(path=str(p),actual=rec,expected_sha256=item['sha256']))
    spec=OLD/'deployment/run_spec.json';assert tracked(spec)['sha256']==ident['runtime']['run_spec_sha256']
    row['bound_input_files']=inputs
    if fit.startswith('huginn'):
        recipe=read(OLD/'deployment/huginn_calibration.json')
        assert recipe==calibration['huginn']
        assert recipe['calibration_sha256']==actual['sha256']
        assert len(recipe['rows'])==100
        assert [x['token_length'] for x in recipe['rows']]==ident['token_lengths']
        assert [x['n_valid'] for x in recipe['rows']]==ident['n_valid']
        row['huginn_token_recipe']=dict(path=str(OLD/'deployment/huginn_calibration.json'),actual=tracked(OLD/'deployment/huginn_calibration.json'),
            exact_runtime_binding=True,token_input_hash_count=100)
    else:
        assert origin['token_lengths']==ident['token_lengths'] and origin['n_valid']==ident['n_valid']
    result['upstream'][fit]=row

for root,schema in [(mainroot,common.SCHEMA),(oldmain,common.SCHEMA),(controlroot,controls.SCHEMA),(hroot,huginn.SCHEMA)]:
    checked=common.validate_output(root,schema=schema)
    result['evaluation_roots'][str(root)]=dict(status='owner_complete_seal_chains_passed',
        owner_record=tracked(root/'OWNER.json'),complete_record=tracked(root/'COMPLETE.json'),
        identity_sha256=checked['owner']['identity_sha256'],sections=checked['complete']['sections'])
    owner=checked['owner']['identity']
    fits=owner.get('fits') or list(owner.get('control_fits',{}).values()) or [owner['fit']]
    bindings=[]
    for fitrecord in fits:
        fit=fitrecord['fit_dir_at_evaluation'].split('/results/')[1]
        ident=fitrecord['identity']; assert ident==identities[fit]
        assert digest(ident)==fitrecord['fit_identity_sha256']
        key=f'{fit}/{fitrecord["complete_pointer"]["generation"]}/lens.pt'
        assert fitrecord['lens']==ledger['binaries'][key]['expected']
        if fit in inv['provenance']:
            p=inv['provenance'][fit]
            assert fitrecord['complete_pointer']==p['pointers']['final']
            assert fitrecord['checkpoint_pointer']==p['pointers']['checkpoint']
        status=ledger['binaries'][key]['current_status']
        bindings.append(dict(fit=fit,fit_identity_sha256=digest(ident),lens_key=key,evaluated_lens_record=fitrecord['lens'],
                             current_binary_status=status,exact_original_or_reconstruction_available=status in ['complete_hash_verified','reconstructed_historical_bytes_verified']))
    result['downstream'][str(root)]=bindings

items=read(mainroot/'common/items.json');names=read(mainroot/'common/task_names.json');forms=read(mainroot/'common/token_forms.json')
plan=read(hroot/'population/eligibility.json')
assert digest([{key:row[key] for key in common.POPULATION_FIELDS} for row in items])==common.POPULATION_SHA256
assert digest(names)==common.TASK_NAMES_SHA256 and digest(forms)==common.TOKEN_FORMS_SHA256

def array_record(a):
    assert a.dtype.kind in 'biufc' and np.isfinite(a).all()
    return dict(dtype=str(a.dtype),shape=list(a.shape),elements=int(a.size),all_finite=True)

def check_ouro(data,columns,control=False):
    fields={'rank','allrank','top1','kl_to_final','rank_of_final_top1'}
    fields|={'kl_to_target_state','rank_of_target_state_top1'} if control else {'kl_to_local','rank_of_local_top1'}
    assert set(data)==fields
    common._validate_arrays(data,items,names,list(range(columns)))
    for key,a in data.items():
        expected_shape=(148,128,columns) if key=='allrank' else ((148,3,columns) if key=='rank' else (148,columns))
        expected_dtype=np.float32 if key.startswith('kl_') else (np.int64 if key=='top1' else np.int32)
        assert a.shape==expected_shape and a.dtype==expected_dtype
        if key.startswith('kl_'): assert np.isfinite(a).all()
        else: assert np.all(a<49152) and np.all(a>=(-1 if key in {'rank','allrank'} else 0))
    # Identical token aliases must preserve identical ranks.
    for i,item in enumerate(items):
        seen={}
        for j,name in enumerate(names[item['task']]):
            tokens=tuple(sorted(forms[item['task']][name]))
            if tokens in seen: assert np.array_equal(data['allrank'][i,j],data['allrank'][i,seen[tokens]])
            seen[tokens]=j

def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        assert len(archive.files)==len(set(archive.files))
        return {k:archive[k] for k in archive.files}

def check_scores(data):
    arms=['target_main','target_raw','target_penultimate','position_main','position_raw','position_sampled_sum','position_diagonal']
    pairs={'target_main_minus_penultimate':('target_main','target_penultimate'),
       'target_main_minus_raw':('target_main','target_raw'),'target_penultimate_minus_raw':('target_penultimate','target_raw'),
       'position_sampled_sum_minus_diagonal':('position_sampled_sum','position_diagonal'),
       'position_main_minus_sampled_sum':('position_main','position_sampled_sum'),
       'position_main_minus_raw':('position_main','position_raw'),
       'position_sampled_sum_minus_raw':('position_sampled_sum','position_raw'),
       'position_diagonal_minus_raw':('position_diagonal','position_raw')}
    fields=['own_layer','control_layer','delta_layer','own_regions','control_regions','delta_regions']
    assert set(data)=={f'{arm}_{f}' for arm in arms for f in [*fields,'eligible']}|{f'contrast_{p}_{f}' for p in pairs for f in fields}
    eligible=np.asarray([any(x['eligible']) for x in items])
    for arm in arms:
        assert data[arm+'_eligible'].dtype==np.bool_ and np.array_equal(data[arm+'_eligible'],eligible)
    for key,a in data.items():
        if key.endswith('_eligible'): continue
        cols=2 if key.endswith('_regions') else (190 if 'target_' in key else 191)
        assert a.dtype==np.float64 and a.shape==(148,cols) and np.isfinite(a).all()
    for pair,(left,right) in pairs.items():
        for field in fields: assert np.array_equal(data[f'contrast_{pair}_{field}'],data[f'{left}_{field}']-data[f'{right}_{field}'])

for root in [mainroot,oldmain,controlroot,hroot]:
    for path in sorted([*root.rglob('*.npz'),*root.rglob('cache.pt')]):
        before=guard(path);seal=read(path.parent/'SEAL.json');expected=seal['files'][path.name]
        actual=tracked(path,expected);entry=dict(expected=expected,actual=actual,status='passed')
        if path.suffix=='.npz':
            data=load_npz(path);entry['arrays']={k:array_record(a) for k,a in data.items()}
            if root==hroot:
                huginn._validate_arrays(data,plan)
                entry['semantics']='Huginn exact schema, vocabulary, common name support, -1 padding, own-slot aliases and coda/native endpoint'
            elif path.name=='scores.npz':
                check_scores(data);entry['semantics']='exact 97 score fields, saved eligibility metadata, fixed support and saved contrast arithmetic; no rank rescoring'
            elif path.name=='paired_ranks.npz':
                expected_keys=set()
                for prefix,columns,source,sourceprefix in [('target_main_',190,mainroot/'fits/fit_01/arrays.npz','jlens_exit3_'),
                    ('target_raw_',190,mainroot/'common/arrays.npz','logitlens_'),
                    ('position_main_',191,mainroot/'fits/fit_01/arrays.npz','jlens_exit3_'),
                    ('position_raw_',191,mainroot/'common/arrays.npz','logitlens_')]:
                    selected={k[len(prefix):]:a for k,a in data.items() if k.startswith(prefix)}
                    check_ouro(selected,columns);original=load_npz(source)
                    for key,a in selected.items():
                        expected_keys.add(prefix+key);assert np.array_equal(a,original[sourceprefix+key][...,:columns])
                    del original
                assert set(data)==expected_keys
                entry['semantics']='exact four paired rank banks equal to fixed support slices of original main/raw arrays'
            elif root==controlroot:
                cols=190 if path.parent.name=='penultimate' else 191
                check_ouro(data,cols,control=True)
                metadata=read(path.parent/'metadata.json')
                entry['semantics']='learned control support, exact dtypes/schema, vocabulary, -1 padding, own slots and token aliases'
            else:
                prefix='logitlens_' if path.parent.name=='common' else 'jlens_exit3_'
                bank={k[len(prefix):]:a for k,a in data.items() if k.startswith(prefix)}
                check_ouro(bank,192)
                assert set(data)=={prefix+k for k in bank}|({'exit_top1'} if prefix=='logitlens_' else set())
                if prefix=='logitlens_':
                    assert data['exit_top1'].dtype==np.int64 and data['exit_top1'].shape==(148,4)
                    assert np.array_equal(data['exit_top1'],bank['top1'][:,[47,95,143,191]])
                else:
                    raw=load_npz(root/'common/arrays.npz')
                    for key in ['rank','allrank','top1']:
                        assert np.array_equal(bank[key][...,191],raw['logitlens_'+key][...,191])
                    del raw
                entry['semantics']='exact main/raw schema, vocabulary, full support, -1 padding, own slots, token aliases and native/identity endpoints'
            del data;gc.collect()
        else:
            if root==hroot:
                cache=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                shapes={'H':(148,32,5280),'target_states':(148,5280),'native_logits':(148,65536)}
                assert set(cache)==set(shapes)
                for k,s in shapes.items(): assert tuple(cache[k].shape)==s
                arr=load_npz(path.parent/'arrays.npz')
                assert np.array_equal(cache['native_logits'].argmax(-1).numpy(),arr['native_top1']);del arr
                tensors=cache
            else:
                cache=controls._cache(torch,root)
                assert tuple(cache['exit_logits'].shape)==(148,4,49152)
                tensors={'H':cache['H'],'exit_logits':cache['exit_logits'],
                         **{f'target_state_logits/{k}':v for k,v in cache['target_state_logits'].items()}}
                arr=load_npz(root/'common/arrays.npz')
                assert np.array_equal(cache['exit_logits'].argmax(-1).numpy(),arr['exit_top1']);del arr
            entry['tensors']={}
            for key,t in tensors.items():
                assert type(t) is torch.Tensor and t.dtype==torch.float32 and t.device.type=='cpu'
                assert t.layout==torch.strided and not t.requires_grad and bool(torch.isfinite(t).all())
                entry['tensors'][key]=dict(dtype=str(t.dtype),shape=list(t.shape),all_finite=True)
            entry['semantics']='exact CPU FP32 cache schema/geometry and saved native argmax equality'
            del tensors,cache,t;gc.collect()
        assert guard(path)==before
        result['files'][str(path)]=entry
        print('VERIFIED',path.relative_to(OLD),flush=True)

# Reconcile pre-existing main-only handoff duplicate numerical records.
duplicates=[]
for p in oldmain.rglob('*'):
    if p.suffix not in {'.pt','.npz'}:continue
    q=mainroot/p.relative_to(oldmain)
    assert same(tracked(p),tracked(q));duplicates.append(dict(first=str(p),second=str(q),same_full_file_sha256=True))
result['duplicate_evaluation_binaries']=duplicates
for path,rec in hash_cache.items():assert guard(path)==rec['stat_after']
result['status']='passed'
result['counts']=dict(evaluation_roots=4,evaluation_binaries=len(result['files']),
    calibration_identities=len(result['upstream']),known_binaries_original_exact=7,reconstructed_exact=1,unresolved=8)
result['reconstructed_fit02_sha256']='101f31db6aa56d97fbae7ecb3e2f241ef1805000484d0e22f5eba5fc0b9acde8'
result['source_records']={str(p):record(p) for p in [Path(__file__),OUT/'recover_estimators.py',OUT/'audit_huginn_retention.py',OUT/'finalize_ledger.py',
    OLD/'deployment/evaluate_refits.py',OLD/'deployment/evaluate_controls.py',OLD/'deployment/evaluate_huginn.py']}
result['all_consumed_file_records']=hash_cache
result['finished_utc']=now()
atomic_json(OUT/'RETAINED_INPUTS_SUPPLEMENT.json',result)
print('SUPPLEMENT',result['counts'],flush=True)
