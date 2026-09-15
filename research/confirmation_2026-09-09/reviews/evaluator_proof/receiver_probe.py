from pathlib import Path
import sys,tempfile,json,hashlib,copy
import numpy as np
import torch
BASE=Path('/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09')
sys.path.insert(0,str(BASE/'evaluation'))
from validate_outputs import validate_outputs
from artifacts import record
R=Path(tempfile.mkdtemp(prefix='confirmation-evaluator-proof-'))
def js(p,v): p.write_text(json.dumps(v))
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True);torch.save(v,p)
def npz(p,v): p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,**v)
n=3; support=list(range(192));vs=[0,47,95,143,170,176,181,189,190,191]
H=torch.arange(2048).float().div(2048).expand(n,192,2048).clone()
rows=[dict(token_ids=[123],n_tokens=1,readout_position=-1,own_index=[0,-1,-1],eligible=[True],control_indices=[[1]],intermediates=['A'],intermediate_tokens={'A':[1,2]}) for _ in range(n)]
pop=dict(rows=rows,token_forms={'multihop':{'A':[1,2],'B':[100,101]}},names={'multihop':['A','B']},cross_concept_token_collisions=[])
js(R/'benchmark.json',{});js(R/'population.json',pop)
J={v:torch.eye(2048).half()*2 for v in vs if v!=191};save(R/'bank.pt',{'J':J});del J
spec=dict(run_id='synthetic-independent-review',benchmark_record=record(R/'benchmark.json'),population_record=record(R/'population.json'),planned_items=n,planned_concepts=2,source_records={},model_record={},bank_records={'fit01':{'receiver_path':str(R/'bank.pt'),'record':record(R/'bank.pt')}},original_precision={},development_item_names=['dev0','dev1','dev2'],arm_support={'raw':support,'fit01':support})
js(R/'run_spec.json',spec)
binding=dict(run_id=spec['run_id'],run_spec_sha256=record(R/'run_spec.json')['sha256'],output_contract_sha256='synthetic')
js(R/'provenance.json',dict(binding=binding,source_records={},model_record={},bank_records=spec['bank_records'],dtype='torch.bfloat16',geometry=[4,48,192,2048],runtime={'precision':{}}))
native=dict(item_names=spec['development_item_names'],virtual_states=H,physical_states=H.clone(),wrapper_states=H.clone(),native_logits=torch.zeros(n,49152),unembedded_logits=torch.zeros(n,49152))
save(R/'development/native.pt',native)
cache=dict(H=H,exit_logits=torch.zeros(n,4,49152),continuations=['']*n);save(R/'common/cache.pt',cache)
A={k:np.zeros((n,192),np.float32 if k.startswith('kl') else np.int64 if k=='top1' else np.int32) for k in ['top1','kl_to_final','kl_to_local','rank_of_final_top1','rank_of_local_top1']}
A['allrank']=np.full((n,128,192),-1,np.int32);A['allrank'][:,0]=1;A['allrank'][:,1]=100
A['rank']=np.full((n,3,192),-1,np.int32);A['rank'][:,0]=1
A['top10_ids']=np.broadcast_to(np.arange(10,dtype=np.int32),(n,192,10)).copy()
S={}
for arm in ['raw','fit01']:
 npz(R/f'readouts/{arm}.npz',A)
 S[arm]=[dict(item_index=i,virtual_indices=vs,transported=torch.stack([H[i,v]*(2 if arm=='fit01' and v!=191 else 1) for v in vs]),logits=-torch.arange(49152).float().expand(len(vs),49152).clone(),sorted_ids=torch.arange(49152,dtype=torch.int32).expand(len(vs),49152).clone()) for i in range(n)]
 save(R/f'readouts/{arm}_samples.pt',S[arm])
files=['run_spec.json','benchmark.json','population.json','provenance.json','development/native.pt','common/cache.pt','readouts/raw.npz','readouts/raw_samples.pt','readouts/fit01.npz','readouts/fit01_samples.pt']
manifest=dict(files={f:{} for f in files},binding=binding,kind='final')
contract=dict(required_checks=['loadability','numerical'],files={f:{} for f in files},run_id=spec['run_id'])
def check():return validate_outputs(root=R,manifest=manifest,contract=contract)
accepted=check();proof={'valid_fixture':accepted['details'],'rejected':{}}
def reject(label,fn):
 try:check()
 except (ValueError,KeyError,TypeError) as e: proof['rejected'][label]=str(e)
 else:raise AssertionError('accepted corruption: '+label)
 fn()
x=copy.deepcopy(native);x['physical_states'][0,0,0]+=1;save(R/'development/native.pt',x);reject('physical hook mismatch',lambda:save(R/'development/native.pt',native))
x=copy.deepcopy(cache);x['H'][0,0,0]=float('nan');save(R/'common/cache.pt',x);reject('nonfinite cache',lambda:save(R/'common/cache.pt',cache))
x=copy.deepcopy(S['raw']);x[0]['transported'][0,0]+=1;save(R/'readouts/raw_samples.pt',x);reject('raw transport mismatch',lambda:save(R/'readouts/raw_samples.pt',S['raw']))
x=copy.deepcopy(S['fit01']);x[0]['transported'][0,0]+=1;save(R/'readouts/fit01_samples.pt',x);reject('Jacobian transport mismatch',lambda:save(R/'readouts/fit01_samples.pt',S['fit01']))
x={k:v.copy() for k,v in A.items()};x['allrank'][0,1,0]=101;npz(R/'readouts/raw.npz',x);reject('sample rank mismatch',lambda:npz(R/'readouts/raw.npz',A))
x={k:v.copy() for k,v in A.items()};x['allrank'][0,1,1]=9;npz(R/'readouts/raw.npz',x);reject('unsampled top10 alias mismatch',lambda:npz(R/'readouts/raw.npz',A))
x=copy.deepcopy(S['raw']);x[0]['sorted_ids'][0,100:102]=torch.tensor([101,100]);save(R/'readouts/raw_samples.pt',x);reject('unsorted full permutation',lambda:save(R/'readouts/raw_samples.pt',S['raw']))
x=copy.deepcopy(S['raw']);x[0]['virtual_indices'][0]=1;save(R/'readouts/raw_samples.pt',x);reject('wrong fixed sample support',lambda:save(R/'readouts/raw_samples.pt',S['raw']))
check();proof.update(status='passed',new_confirmation_results_exposed=False,root=str(R),reviewed_sources={str(p):record(p) for p in sorted([*(BASE/'evaluation').glob('*.py'),*(BASE/'analysis').glob('*.py')])})
js(R/'PROOF.json',proof);print(json.dumps(proof))
