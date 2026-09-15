"""Audit already copied actual engineering gate data only; no model imports/network."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import stat

ROUND=Path('/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07')
BASE=ROUND/'monitoring/attempt_06/20260909T115652Z'
checks=[]
inputs={}
def check(ok,label):
    if not ok:raise ValueError(label)
    checks.append(label)
def record(path):
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('symlink input')
    check(stat.S_ISREG(path.stat().st_mode),'regular '+path.name)
    b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load(path):
    inputs[str(path)]=record(path);return json.loads(path.read_bytes())
def epoch(value):return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def stamp(value):return dt.datetime.fromtimestamp(value,dt.timezone.utc).isoformat()
receipt=load(BASE/'COPY_VERIFIED.json')
check(receipt['status']=='passed' and receipt['lease_name']=='jlens-refit-14ca9417eb6145c8979fb87ea9fde68c' and receipt['pod_id']=='vzwx0cj43g2uw5','copy receipt identity')
check(len(receipt['files'])==len({v['path'] for v in receipt['files']})==14,'exact14 copy members')
docs={}
for row in receipt['files']:
    relative=Path(row['path']);check(not relative.is_absolute() and '..' not in relative.parts,'safe receipt path')
    path=BASE/'results'/relative;docs[row['path']]=load(path)
    check(inputs[str(path)]=={k:row[k] for k in ('bytes','sha256')},'copy member hash '+row['path'])
check({p.relative_to(BASE/'results').as_posix() for p in (BASE/'results').rglob('*') if p.is_file()}==set(docs),'exact copied tree inventory')
h=docs['preflight_huginn/COMPLETE.json'];start=docs['preflight_huginn/START.json'];r=h['profiles'][0]
check(h['status']=='passed' and h['schema_version']==1 and h['kind']=='huginn' and len(h['profiles'])==1,'H passed gate membership')
check(start['status']=='running' and start['profiles']==[],'H start is pre-gate')
check(all(h[k]==v for k,v in start.items() if k not in ('status','profiles')),'H start/complete consistency')
check(r==docs['preflight_huginn/huginn_r8.json'],'H profile/complete equality')
rs=docs['preflight_huginn/huginn_r8_START.json']
check(all(r[k]==v for k,v in rs.items() if k!='memory_cleanup') and all(r['memory_cleanup'][k]==v for k,v in rs['memory_cleanup'].items()),'H profile start binding')
check(h['native_primal']=={'all_34_cells_bitwise_equal':True,'batch':8,'coupled_initial_states':True,'logits_bitwise_equal':True,'sequence_length':128},'all34 native primal/logits coupled B8')
check((h['dim_batch'],h['max_seq_len'],h['skip_first'])==(8,128,16),'frozen benchmark geometry')
check(r['source_layers']==list(range(32)) and r['target_layer']==33 and r['q'] is None and r['name']=='huginn_r8','full dense derivative support')
parity=r['parity'][0]
check(len(r['parity'])==1 and parity['arm']=='dense' and parity['bitwise_equal'] is True and parity['source_count']==32 and parity['output_directions']==5280 and parity['values_compared']==32*5280**2,'full892108800 derivative bitwise values')
diag=r['optimized_diagnostics']
check((diag['dim_batch'],diag['n_passes'],diag['source_count'],diag['target'],diag['sequence_length'],diag['n_valid'])==(8,660,32,33,128,111) and r['n_valid']==128-16-1,'660 complete B8 passes and111 positions')
check(diag['source_edges'] is True and diag['engine']=='cuda_graph','optimized derivative source edges')
check(r['prompt_with_checkpoint_seconds']==r['optimized_seconds']+r['io']['accumulate_write_hash_prune_seconds'],'complete measured compute plus IO exact')
check(r['io']['scratch_removed'] is True and r['io']['state_bytes']==3568444035,'engineering-only IO scratch removed')
for key in ('native_seconds','optimized_seconds','prompt_with_checkpoint_seconds'):
    check(type(r[key]) in (int,float) and math.isfinite(r[key]) and r[key]>0,'finite timing '+key)
memories=[]
def visit(v):
    if isinstance(v,dict):
        if 'effective_available_bytes' in v:
            limits=[z['limit_bytes'] for z in v['cgroup_constraints'] if z['limit_bytes'] is not None]
            room=[z['headroom_bytes'] for z in v['cgroup_constraints'] if z['limit_bytes'] is not None]
            check(all(z['headroom_bytes']==max(0,z['limit_bytes']-z['usage_bytes']) for z in v['cgroup_constraints'] if z['limit_bytes'] is not None),'cgroup headroom arithmetic')
            check(v['effective_total_bytes']==min([v['physical_total_bytes'],*limits]) and v['effective_available_bytes']==min([v['physical_available_bytes'],*room]),'effective memory minima')
            memories.append(v['effective_available_bytes'])
        for z in v.values():visit(z)
    elif isinstance(v,list):
        for z in v:visit(z)
visit(h)
for budget in (h['initial_host_budget'],r['host_offload_budget']):
    check(budget['minimum_host_bytes']==48000000000 and budget['memory']['effective_total_bytes']>=budget['minimum_host_bytes'],'48GB effective-host floor')
    check(budget['non_payload_headroom_bytes']==8*2**30 and budget['reserve_host_bytes']==4*2**30 and budget['max_host_bytes']==min(24*2**30,budget['memory']['effective_available_bytes']-8*2**30),'24GiB cap with8GiB allowance and4GiB reserve')
off=r['native_offloader']
check(off['engine']=='exact_cpu_offload' and off['pin_memory'] is False and off['max_host_bytes']==24*2**30 and off['reserve_host_bytes']==4*2**30,'native exact offload frozen budget')
check(off['live_host_payload_bytes']==0 and off['counts']['offloaded_payloads']==off['counts']['released_payloads']==433 and 0<off['peak_host_payload_bytes']<=off['max_host_bytes'],'433 offload payloads released andzero live')
check(all(0<r[k]<h['runtime']['gpu']['total_memory'] for k in ('native_peak_cuda_bytes','optimized_peak_cuda_bytes')),'observed CUDA peaks within actual device')
lease=load(ROUND/'cloud_leases/attempt_06/LEASE.json')
check(lease['name']==receipt['lease_name'] and lease['pod_id']==receipt['pod_id'],'current lease identity')
lease_fields={k:lease[k] for k in ['name','pod_id','billing_start_utc','work_deadline_utc','watch_deadline_utc','prior_spend_upper_usd','all_in_rate']}
oprofiles=docs['preflight_ouro/COMPLETE.json']['profiles'];by_name={v['name']:v for v in oprofiles}
check(set(by_name)=={'ouro_main','ouro_penultimate','ouro_positions'},'Ouro profile membership')
for row in oprofiles:check(row['prompt_with_checkpoint_seconds']==row['optimized_seconds']+row['io']['accumulate_write_hash_prune_seconds'],'Ouro complete timing '+row['name'])
projection_audit={}
for name in ('initial_budget_projection.json','huginn_budget_projection.json'):
    b=docs[name]
    if name.startswith('initial'):
        main=by_name['ouro_main'];per=main['optimized_seconds']*3.8363+main['io']['accumulate_write_hash_prune_seconds']*(32*5280**2)/(191*2048**2)
        fit=sum(n*by_name[k]['prompt_with_checkpoint_seconds'] for k,n in [('ouro_main',500),('ouro_penultimate',100),('ouro_positions',100)])+100*per;other=10800
    else:per=r['prompt_with_checkpoint_seconds'];fit=100*per;other=3600
    check(math.isclose(per,b['huginn_seconds_per_prompt'],rel_tol=1e-14) and math.isclose(fit,b['fit_seconds'],rel_tol=1e-14),'projection timing arithmetic '+name)
    check(b['timing_margin']==.1 and b['other_work_allowance_seconds']==other and b['retrieval_allowance_seconds']==3600,'projection fixed allowances '+name)
    finish=epoch(lease['billing_start_utc'])+(b['projected_combined_cost_usd']-lease['prior_spend_upper_usd'])/lease['all_in_rate']*3600-3600
    begin=finish-fit*1.1-other
    timestamp_lag=epoch(b['recorded_utc'])-begin
    check(0<=timestamp_lag<.01,'measurement to receipt timestamp lag '+name)
    check(b['fits_before_work_deadline'] is (finish<epoch(lease['work_deadline_utc'])) and b['within_combined_budget'] is (b['projected_combined_cost_usd']<24.85) and b['status']=='passed','projection gate decisions '+name)
    projection_audit[name]={'per_prompt_seconds':per,'fit_seconds':fit,'measurement_to_receipt_seconds':timestamp_lag,'projected_work_finished_utc':stamp(finish),'projected_retrieval_finished_utc':stamp(finish+3600),'projected_charge_usd':b['projected_combined_cost_usd'],'work_deadline_margin_seconds':epoch(lease['work_deadline_utc'])-finish,'watch_deadline_margin_after_retrieval_seconds':epoch(lease['watch_deadline_utc'])-finish-3600}
result={'status':'passed','scope':'Actual copied engineering gates only; no scientific fit outcome, model loading, GPU work or network. Frozen provenance and prerequisite reconciliation supplied separately by independent receipt reviewer.','checks':checks,'inputs':inputs,'lease_fields':lease_fields,'native_primal':h['native_primal'],'parity':parity,'memory':{'native_cuda_peak_bytes':r['native_peak_cuda_bytes'],'optimized_cuda_peak_bytes':r['optimized_peak_cuda_bytes'],'peak_host_payload_bytes':off['peak_host_payload_bytes'],'live_host_payload_bytes':off['live_host_payload_bytes'],'minimum_recorded_effective_available_bytes':min(memories),'host_cap_bytes':off['max_host_bytes']},'timing':{k:r[k] for k in ['native_seconds','optimized_seconds','prompt_with_checkpoint_seconds']},'io_seconds':r['io']['accumulate_write_hash_prune_seconds'],'projections':projection_audit}
output=Path('/tmp/actual_huginn_gate_review.json')
with output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':'passed','checks':len(checks),'output':str(output),'record':record(output),'projections':projection_audit},indent=2))
