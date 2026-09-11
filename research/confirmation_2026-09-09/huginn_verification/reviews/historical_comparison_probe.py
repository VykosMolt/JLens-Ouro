#!/usr/bin/env python3
"""Read-only independent checks; hardlinked synthetic acceptance fixture only."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent/'analysis/compare_historical.py'
def record(path):
    raw=Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load_module():
    spec=importlib.util.spec_from_file_location('reviewed_historical_compare',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def child_consumer_probe(config):
    config=json.loads(Path(config).read_text());module=load_module()
    target,alternate,parked=map(Path,(config['target'],config['alternate'],config['parked']))
    original_open=Path.open;swaps=[]
    def switched_open(path,*args,**kwargs):
        consumer_open = args and (args[0]=='r' or (args[0]=='rb' and inspect.currentframe().f_back.f_code.co_name=='raw'))
        if path==target and consumer_open:
            Path(config['seen']).write_text('actual consumer open intercepted')
            target.parent.rename(parked);alternate.rename(target.parent)
            try:
                result=original_open(path,*args,**kwargs);swaps.append(True);return result
            finally:target.parent.rename(alternate);parked.rename(target.parent)
        return original_open(path,*args,**kwargs)
    sys.argv=[str(SCRIPT),'--accepted',config['accepted'],'--expected-receipt-sha256',config['receipt_sha'],'--out',config['out']]
    with patch.object(Path,'open',switched_open):module.main()
    print(json.dumps({'consumer_swaps':len(swaps)}))

def main():
    before=record(SCRIPT);module=load_module()
    # /tmp is another filesystem. This directory permits hardlinks without copying payloads.
    base=Path(tempfile.mkdtemp(prefix='.historical-comparison-fixture-',dir=HERE))
    cases=[]
    value=module.compare_array(np.array([0.,-0.,1.],np.float32),np.array([-0.,0.,2.],np.float32))
    assert value['bitwise_different_entries']==3 and value['numeric_different_entries']==1
    assert value['max_absolute_difference']==1 and abs(value['rms_difference']-(1/3)**.5)<1e-15
    cases.append({'case':'signed_zeros_and_ordinary_inequality','status':'passed','result':value})
    value=module.compare_array(np.array([1,2,3],np.int32),np.array([1,4,3],np.int32))
    assert value['bitwise_different_entries']==value['numeric_different_entries']==1 and value['max_absolute_difference']==2
    cases.append({'case':'saved_rank_integer_difference','status':'passed','result':value})
    for old,new in [(np.array([1],np.int32),np.array([1],np.int64)),(np.array([np.nan]),np.array([0.])),(np.array([1.]),np.array([1.,2.]))]:
        try:module.compare_array(old,new)
        except ValueError:pass
        else:raise AssertionError('Malformed numeric comparison accepted')
    cases.append({'case':'dtype_shape_and_nonfinite_rejected','status':'passed'})
    assert module.compare_json({'a':[1,2]},{'a':[1,3]})==[{'path':'/a/1','historical':2,'rerun':3}]
    cases.append({'case':'json_initialization_summary_difference_path','status':'passed'})
    large=module.compare_array(np.array([2**63],np.uint64),np.array([2**63+1],np.uint64))
    cases.append({'case':'adjacent_large_integer_numeric_difference','status':'scope_limitation' if large['max_absolute_difference']!=1 else 'passed',
                  'result':large,'scope':'Not present in bounded saved token/rank values; generic numeric-helper limitation.'})
    target=base/'canonical'/'file.bin';target.parent.mkdir();target.write_bytes(b'bad!')
    alternate=base/'alternate';alternate.mkdir();(alternate/target.name).write_bytes(b'good')
    parked=base/'parked';original_open=Path.open
    def switched_open(path,*args,**kwargs):
        if path==target and args and args[0]=='rb':
            target.parent.rename(parked);alternate.rename(target.parent)
            try:return original_open(path,*args,**kwargs)
            finally:target.parent.rename(alternate);parked.rename(target.parent)
        return original_open(path,*args,**kwargs)
    try:
        with patch.object(Path,'open',switched_open):observed=module.record(target)
        descriptor_rejected=False
    except ValueError as error:
        observed={'error':str(error)};descriptor_rejected=True
    cases.append({'case':'parent_swap_during_hash_open_descriptor_binding','status':'passed' if descriptor_rejected else 'failed',
                  'reported':observed,'actual_canonical_record':record(target)})
    linked=base/'linked_parent';linked.symlink_to(target.parent,target_is_directory=True)
    try:module.record(linked/target.name)
    except ValueError:cases.append({'case':'linked_parent_rejected','status':'passed'})
    else:cases.append({'case':'linked_parent_rejected','status':'failed'})

    accepted=base/'synthetic_accepted';accepted.mkdir();results=accepted/'results';results.mkdir();files={}
    for seed in module.SEEDS:
        for name in ('arrays.npz','cache.pt','initializations.json','summaries.json'):
            relative=f'readouts/seeds/{seed}/{name}'
            source=module.HISTORY/f'seeds/{seed}/{name}';destination=results/relative
            destination.parent.mkdir(parents=True,exist_ok=True);os.link(source,destination);files[relative]=record(source)
    # The old H bank is absent. This cache-as-bank is explicitly synthetic only.
    lens=results/'fit/lens.pt';lens.parent.mkdir();os.link(module.HISTORY/f'seeds/{module.SEEDS[0]}/cache.pt',lens)
    files['fit/lens.pt']=record(lens)
    binding={'run_id':'synthetic-history-self-pair','output_contract_sha256':'1'*64}
    manifest={'kind':'final','outcome':'complete','binding':binding,'files':files}
    manifest_id=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    (accepted/'MANIFEST.json').write_text(json.dumps(manifest))
    validation={'status':'passed','checks':{'loadability':True,'numerical':True},'run_id':binding['run_id'],
                'contract_sha256':binding['output_contract_sha256'],'manifest_sha256':manifest_id,'checked_files':list(files)}
    def write_chain(value=validation,manifest_sha=manifest_id):
        (accepted/'VALIDATION.json').write_text(json.dumps(value))
        receipt={'status':'passed','kind':'final','binding':binding,'files':files,'manifest_sha256':manifest_sha,
                 'manifest_record':record(accepted/'MANIFEST.json'),'validation_record':record(accepted/'VALIDATION.json')}
        (accepted/'RECEIPT.json').write_text(json.dumps(receipt));return record(accepted/'RECEIPT.json')['sha256']
    receipt_sha=write_chain()
    (base/'SYNTHETIC_ONLY.json').write_text(json.dumps({'synthetic':True,'description':'Retained old readouts self-paired with synthetic receipt and cache-as-bank; no new H results and no actual external acceptance.'}))
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    def run_cli(label,pin=receipt_sha):
        output=base/(label+'.json')
        run=subprocess.run([sys.executable,str(SCRIPT),'--accepted',str(accepted),'--expected-receipt-sha256',pin,'--out',str(output)],
                           capture_output=True,text=True,timeout=120,env=environment)
        return run,output
    run,report=run_cli('self_pair')
    assert run.returncode==0,run.stderr
    value=json.loads(report.read_text())
    arrays=[a for seed in value['seeds'].values() for kind in ('rank_and_argmax_arrays','cached_tensors') for a in seed[kind].values()]
    assert all(a['bitwise_different_entries']==a['numeric_different_entries']==0 and a['max_absolute_difference']==0 for a in arrays)
    assert all(seed[name]['exact_value_equality'] for seed in value['seeds'].values() for name in ('initializations.json','summaries.json'))
    cases.append({'case':'actual_historical_self_pair_arrays_caches_initializations_summaries','status':'passed',
                  'array_count':len(arrays),'entries':sum(a['entries'] for a in arrays),'report':{'path':str(report),**record(report)},
                  'scope':'Synthetic self-pair only; no newly rerun estimator or actual acceptance.'})
    for mode in ('wrong_receipt_pin','failed_validation','wrong_manifest_id','missing_checked_file','failed_numerical_check'):
        if mode=='wrong_receipt_pin':pin='0'*64
        elif mode=='failed_validation':pin=write_chain({**validation,'status':'failed'})
        elif mode=='wrong_manifest_id':pin=write_chain(manifest_sha='0'*64)
        elif mode=='missing_checked_file':pin=write_chain({**validation,'checked_files':list(files)[1:]})
        else:pin=write_chain({**validation,'checks':{'loadability':True,'numerical':False}})
        run,out=run_cli(mode,pin)
        cases.append({'case':mode+'_rejected','status':'passed' if run.returncode!=0 and not out.exists() else 'failed',
                      'returncode':run.returncode,'stderr':run.stderr[-1500:]})
        receipt_sha=write_chain()

    # Change only the consumer-opened inode, restoring its path before final guards.
    for kind,filename in (('json','initializations.json'),('npz','arrays.npz'),('torch','cache.pt')):
        target=results/f'readouts/seeds/{module.SEEDS[0]}/{filename}'
        alternate=base/('consumer_alternate_'+kind);alternate.mkdir()
        if kind=='json':
            altered=json.loads(target.read_text());altered[0]['independent_review_fake_value']=True
            (alternate/target.name).write_text(json.dumps(altered))
        else:(alternate/target.name).write_bytes(b'consumer swap fixture')
        parked=base/('consumer_parked_'+kind);seen=base/('consumer_seen_'+kind)
        output=base/('consumer_swap_'+kind+'_comparison.json')
        config={'target':str(target),'alternate':str(alternate),'parked':str(parked),'accepted':str(accepted),
                'receipt_sha':receipt_sha,'out':str(output),'seen':str(seen)}
        config_path=base/('consumer_probe_'+kind+'_config.json');config_path.write_text(json.dumps(config))
        run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--consumer-probe',str(config_path)],capture_output=True,text=True,timeout=120,env=environment)
        rejection=run.returncode!=0 and not output.exists() and seen.exists()
        case={'case':'parent_swap_during_'+kind+'_consumer_is_rejected','status':'passed' if rejection else 'failed',
              'injection_executed':seen.exists(),'returncode':run.returncode,'stdout':run.stdout[-2000:],'stderr':run.stderr[-2000:]}
        if output.exists():
            swapped=json.loads(output.read_text());case['report']={'path':str(output),**record(output)}
            case['actual_bytes_match_history']=record(target)==record(module.HISTORY/f'seeds/{module.SEEDS[0]}/{filename}')
        cases.append(case)
    after=record(SCRIPT)
    proof={'schema':'historical_comparison_independent_probe.v1','script_record_before':before,'script_record_after':after,
           'source_stable':before==after,'cases':cases,'temporary_root':str(base),'source_record':record(__file__),
           'no_model_or_provider_execution':True,'historical_files_hardlinked_only':True}
    output=base/'PROOF.json';output.write_text(json.dumps(proof,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'proof':str(output),'source_stable':before==after,'failed_cases':[c['case'] for c in cases if c['status']=='failed']}))
if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--consumer-probe':child_consumer_probe(sys.argv[2])
    else:main()
