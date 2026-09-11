"""Time eight bounded SSH streams carrying authorized bank bytes to this lease."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,shlex,sys,time
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'controller'))
import lease
import artifact_handoff as handoff
from transport import run_bounded
root=B/'cloud_leases/attempt_01';state=json.loads((root/'LEASE.json').read_text());binding=handoff.binding(state)
handoff.verify_local_sources(state)
if state.get('halt_requested') or state['status']=='terminated':raise ValueError('Lease stopped')
ssh=lease.ssh_options(root,state);host='root@'+state['ssh']['host']
def remote(command):return run_bounded([*ssh,host,command],timeout=30,operation='transfer_probe_setup').stdout
if remote('cat /workspace/jlens/provider_lease_name').decode().strip()!=state['name']:raise ValueError('Wrong lease')
spec=json.loads(Path(state['run_spec_path']).read_text());source=Path(spec['bank_records']['diagonal']['receiver_path'])
local=B/'resources/transfer_probe';local.mkdir(exist_ok=False)
with source.open('rb') as f:raw=f.read(8*1024**2)
if len(raw)!=8*1024**2:raise ValueError('Bank prefix truncated')
for i in range(8):(local/f'part{i:02}.bin').write_bytes(raw)
remote('mkdir /workspace/jlens/transfer_probe')
start=time.monotonic()
def send(i):
 run_bounded(['rsync','--partial','--protect-args','-t','-e',shlex.join(ssh),str(local/f'part{i:02}.bin'),host+':/workspace/jlens/transfer_probe/'],timeout=120,operation='authorized_transfer_throughput_probe')
 return {'part':i,'elapsed':time.monotonic()-start}
results=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for result in pool.map(send,range(8)):results.append(result)
elapsed=time.monotonic()-start
out={'schema':'confirmation_transfer_probe.v1','binding':binding,'total_bytes':len(raw)*8,'seconds':elapsed,'bytes_per_second':len(raw)*8/elapsed,'part_sha256':hashlib.sha256(raw).hexdigest(),'parts':results}
(local/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
