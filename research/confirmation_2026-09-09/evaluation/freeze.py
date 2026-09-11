#!/usr/bin/env python3
"""Create a new immutable prospective source/data/analysis closure."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile
from artifacts import record, json_save
from measurement import ARMS, SUPPORT, PRIMARY_VIRTUAL, secondary_specs

ROUND=Path(__file__).resolve().parents[1]
REPO=ROUND.parents[1]
OLD=ROUND.parent/'refit_round_2026-09-07'

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:shutil.copyfileobj(src,dst)

def main():
    p=argparse.ArgumentParser()
    for name in ('benchmark','population','annotation-review','implementation-review'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True)
    args=p.parse_args()
    benchmark=json.loads(args.benchmark.read_text());population=json.loads(args.population.read_text())
    for path in (args.annotation_review,args.implementation_review):
        review=json.loads(path.read_text())
        if review.get('status')!='passed' or review.get('new_confirmation_results_exposed') is not False:raise ValueError('Missing successful blinded review')
    annotation_review=json.loads(args.annotation_review.read_text())
    if annotation_review.get('benchmark_record')!=record(args.benchmark) or annotation_review.get('population_record')!=record(args.population):
        raise ValueError('Annotation review is not bound to this exact dataset/population')
    implementation_review=json.loads(args.implementation_review.read_text())
    reviewed=implementation_review.get('reviewed_sources',{})
    required=[*(ROUND/'evaluation').glob('*.py'),*(ROUND/'analysis').glob('*.py')]
    if any(reviewed.get(str(path))!=record(path) for path in required):
        raise ValueError('Implementation review is missing or stale for a packaged source')
    plan=ROUND/'PROSPECTIVE_PLAN.md'
    if 'Draft awaiting' in plan.read_text():raise ValueError('Root must finalize the prospective plan after reviews')
    if population['cross_concept_token_collisions']:raise ValueError('Unresolved control aliases')
    rows=population['rows'];names=population['names']['multihop']
    if len(rows)!=160 or len(names)!=80 or not all(any(r['eligible']) for r in rows):raise ValueError('Frozen 160-item / 80-concept target not met')
    if [r['name'] for r in rows]!=[r['name'] for r in benchmark['items']]:raise ValueError('Population order differs')
    bundle=ROUND/'bundle';bundle.mkdir(exist_ok=False)
    for path in (REPO/'jlens').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path,bundle/'repo'/path.relative_to(REPO))
    for task in ('multihop','order-ops'):copy(REPO/f'data/evaluations/lens-eval-{task}.json',bundle/f'repo/data/evaluations/lens-eval-{task}.json')
    for path in (Path('/home/moloch/ouro_project/src/ouro_jlens')).glob('*.py'):
        if path.name in ('__init__.py','evaluate.py','evaldata.py','recurrent.py','fit_lens.py','evidence.py','bench.py'):
            copy(path,bundle/'ouro_project/src/ouro_jlens'/path.name)
    for name in ('run_refits.py','evaluate_refits.py','cloud_worker.py','environment.json','model_manifest.json','requirements.lock','evaluate_controls.py','run_controls.py'):
        copy(OLD/'deployment'/name,bundle/'legacy'/name)
    for path in (ROUND/'evaluation').glob('*.py'):copy(path,bundle/'evaluation'/path.name)
    for path in (ROUND/'analysis').glob('*.py'):copy(path,bundle/'analysis'/path.name)
    for path in (ROUND/'controller').iterdir():
        if path.is_file() and path.suffix in ('.py','.sh'):copy(path,bundle/'controller'/path.name)
    source_records={p.relative_to(bundle).as_posix():record(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    frozen=bundle/'frozen'
    copy(args.benchmark,frozen/'benchmark.json');copy(args.population,frozen/'population.json')
    copy(plan,frozen/'PROSPECTIVE_PLAN.md')
    copy(args.annotation_review,frozen/'annotation_review.json');copy(args.implementation_review,frozen/'implementation_review.json')
    inputs=json.loads((ROUND/'resources/materialized_inputs.json').read_text())
    bank_records={}
    for key,entry in inputs['banks'].items():
        if key.startswith('ouro/fit_01/'):arms=['fit01'];worker_path='banks/fit01.pt'
        elif key.startswith('ouro/fit_02/'):arms=['fit02'];worker_path='banks/fit02.pt'
        elif 'ouro_penultimate' in key:arms=['penultimate'];worker_path='banks/penultimate.pt'
        else:arms=['sampled_sum','diagonal'];worker_path='banks/positions.pt'
        for arm in arms:bank_records[arm]={'record':entry['record'],'receiver_path':entry['path'],'worker_path':worker_path,'category':entry['category'],'historical_key':key}
    original_meta=json.loads((OLD/'monitoring/attempt_06/ouro_handoff_20260909T110503Z/results/ouro_evaluation/common/metadata.json').read_text())
    olditems=json.loads((REPO/'data/evaluations/lens-eval-multihop.json').read_text())['items']
    spec={'schema':'confirmation_scientific_run.v1','run_id':args.run_id,'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'benchmark_record':record(frozen/'benchmark.json'),'population_record':record(frozen/'population.json'),
          'plan_record':record(frozen/'PROSPECTIVE_PLAN.md'),'annotation_review_record':record(frozen/'annotation_review.json'),
          'implementation_review_record':record(frozen/'implementation_review.json'),
          'planned_items':len(rows),'planned_concepts':len(names),'dependency_groups':len(set(r['dependency_group_id'] for r in rows)),
          'source_records':source_records,'bank_records':bank_records,'arm_support':SUPPORT,
          'model_record':record(bundle/'legacy/model_manifest.json'),'model_identity':original_meta['runtime']['model'],
          'original_precision':original_meta['runtime']['precision'],'development_item_names':[r['name'] for r in olditems[:3]],
          'primary':{'arm':'fit01','reference':'raw','virtual_indices':PRIMARY_VIRTUAL,'metric':'equal-item mean of fixed-layer mean paired excess hit@10'},
          'secondary_specs':secondary_specs(),'uncertainty':{'groups':'dependency_group_id','draws':20000,'seed':2026090901,'primary_interval':'percentile95','secondary_family':'centered bootstrap max-t95, 20 contrasts'}}
    json_save(frozen/'run_spec.json',spec)
    base=['run_spec.json','benchmark.json','population.json','provenance.json','development/native.pt']
    files={p:{'role':'frozen input or native development evidence'} for p in base}
    files['common/cache.pt']={'role':'all new-item hidden states, native exit logits and continuations'}
    stages={'development':sorted(base)}
    for arm in ('raw',*ARMS):
        pair=[f'readouts/{arm}.npz',f'readouts/{arm}_samples.pt']
        files.update({q:{'role':f'{arm} raw per-item ranks/top10 or numerical samples'} for q in pair})
        stages[arm]=sorted(base+['common/cache.pt']+pair)
    contract={'schema':'confirmation_output_contract.v1','run_id':args.run_id,'run_spec_sha256':record(frozen/'run_spec.json')['sha256'],
              'files':files,'stages':stages,'required_checks':['loadability','numerical']}
    json_save(frozen/'output_contract.json',contract)
    config={'schema':'confirmation_run_config.v1','run_id':args.run_id,'run_spec_path':str(frozen/'run_spec.json'),'output_contract_path':str(frozen/'output_contract.json'),
            'semantic_verifier':{'path':str(bundle/'evaluation/validate_outputs.py'),'record':record(bundle/'evaluation/validate_outputs.py'),'callable':'validate_outputs'},
            'setup_budget_seconds':3600,'compute_budget_seconds':3600,'preservation_reserve_seconds':1800,'transfer_timeout_seconds':900}
    json_save(ROUND/'controller/run_config.json',config)
    payload=ROUND/'bundle.tar.gz'
    with tarfile.open(payload,'x:gz') as tar:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():tar.add(path,arcname='bundle/'+path.relative_to(bundle).as_posix(),recursive=False)
    receipt={'schema':'confirmation_prospective_freeze.v1','run_id':args.run_id,'frozen_utc':spec['frozen_utc'],'new_confirmation_results_exposed':False,
             'bundle':record(payload),'files':{p.relative_to(bundle).as_posix():record(p) for p in sorted(bundle.rglob('*')) if p.is_file()},
             'controller_run_config':record(ROUND/'controller/run_config.json')}
    json_save(ROUND/'FREEZE.json',receipt)
    print(json.dumps({'status':'frozen','items':len(rows),'concepts':len(names),'groups':spec['dependency_groups'],'receipt':record(ROUND/'FREEZE.json')}))

if __name__=='__main__':main()
