from pathlib import Path
import tempfile,sys,json,subprocess
from unittest.mock import patch
BASE=Path('/home/moloch/jacobian-lens/research/confirmation_2026-09-09')
sys.path[:0]=[str(BASE/'evaluation'),str(BASE/'controller')]
import launch,test_lifecycle as t,artifact_handoff as h,lease
P=Path(tempfile.mkdtemp(prefix='confirmation-launch-proof-'));f=t.fixture(P,'launch-review')
s=json.loads((f.root/'LEASE.json').read_text());c=json.loads(Path(s['output_contract_path']).read_text());c['stages']={'development':c['stages']['smoke']};t.io._atomic_json(Path(s['output_contract_path']),c)
s['output_contract_record']=h.checked_record(s['output_contract_path']);s['output_contract_sha256']=s['output_contract_record']['sha256'];t.io._atomic_json(f.root/'LEASE.json',s)
m={'schema':'confirmation_artifact_manifest.v1','binding':h.binding(s),'kind':'stage','stage_id':'development','outcome':'complete','files':h.payload_inventory(f.results)}
p=f.workspace/'manifests/development.json';t.io._mkdir(p.parent);t.io._new_json(p,m)
t.io._new_json(f.workspace/'artifact_index.json',{'schema':'confirmation_artifact_index.v1','binding':h.binding(s),'manifests':[{'path':'manifests/development.json','record':h.checked_record(p)}]})
t.accept(f)
calls=[]
def fake(command,**kw):
 calls.append(command)
 if command[-1]=='cat /workspace/jlens/provider_lease_name':return subprocess.CompletedProcess(command,0,(s['name']+'\n').encode(),b'')
 return subprocess.CompletedProcess(command,0,b'',b'')
def run():
 with patch.object(sys,'argv',['launch','--lease',str(f.root),'--phase','accept-development']),patch.object(lease,'ssh_options',return_value=['ssh']),patch.object(launch,'run_bounded',side_effect=fake):launch.main()
run();ack=json.loads((f.root/'launch/DEVELOPMENT_ACCEPTED.json').read_text());assert ack['binding']==h.binding(s) and ack['stage_manifest_sha256']==h.digest(m)
state=json.loads((f.root/'LEASE.json').read_text());ptr=state['stage_acceptances'][h.digest(m)];receipt=Path(ptr['path']);original=receipt.read_bytes();receipt.write_bytes(original+b' ')
length=len(calls)
try:run()
except ValueError as e:rejection=str(e)
else:raise AssertionError('tampered acceptance authorized')
assert len(calls)==length+1,'tampered receipt caused an upload'
receipt.write_bytes(original)
proof={'status':'passed','new_confirmation_results_exposed':False,'root':str(P),'valid_exact_bound_ack':True,'tampered_acceptance_rejected':rejection,'network_or_provider_calls':False,'launch_source':h.checked_record(BASE/'evaluation/launch.py')}
(P/'PROOF.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
