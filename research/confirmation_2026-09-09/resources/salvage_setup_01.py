from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import json,shlex,sys,time,hashlib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'controller'))
import lease
import artifact_handoff as handoff
from transport import run_bounded
root=B/'cloud_leases/attempt_01';state=json.loads((root/'LEASE.json').read_text());binding=handoff.binding(state)
handoff.verify_local_sources(state);ssh=lease.ssh_options(root,state);host='root@'+state['ssh']['host']
program="""import pathlib,json,hashlib,base64,time
r=pathlib.Path('/workspace/jlens')
files={}
for p in [r/'status.json',r/'bootstrap_stdout.log',r/'artifact_index.json',r/'BANKS_READY.json',*sorted((r/'setup_logs').glob('*')),*sorted((r/'results').rglob('*'))]:
 if not p.is_file():continue
 if p.is_symlink() or p.stat().st_size>8*1024**2:raise ValueError('Unexpected scientific/large evidence; requires separate preservation')
 raw=p.read_bytes();files[str(p.relative_to(r))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
partial={}
for p in (r/'banks').rglob('*'):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   while block:=f.read(8*1024**2):h.update(block)
  partial[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps({'time':time.time(),'lease_name':(r/'provider_lease_name').read_text().strip(),'files':files,'partial_input_records':partial,'banks_ready_present':(r/'BANKS_READY.json').exists(),'artifact_index_present':(r/'artifact_index.json').exists()}))
"""
raw=run_bounded([*ssh,host,'python -c '+shlex.quote(program)],timeout=60,operation='preserve_incomplete_setup_evidence').stdout
evidence=json.loads(raw)
if evidence['lease_name']!=state['name'] or evidence['banks_ready_present'] or evidence['artifact_index_present']:raise ValueError('Worker progressed beyond the expected incomplete setup')
output=B/'resources/SETUP_ATTEMPT_01_PRESERVED.json'
with output.open('x') as f:json.dump({'schema':'confirmation_incomplete_setup_preservation.v1','binding':binding,'collected_utc':datetime.now(timezone.utc).isoformat(),'remote':evidence,'interpretation':'No native development or confirmation computation began. Full original input banks remain verified locally; partial remote input duplicates are recorded by hash/size.'},f,indent=2);f.write('\n')
local=B/'resources/transfer_probe_download';local.mkdir(exist_ok=False)
start=time.monotonic()
def receive(i):
 run_bounded(['rsync','-t','--protect-args','-e',shlex.join(ssh),host+f':/workspace/jlens/transfer_probe/part{i:02}.bin',str(local/f'part{i:02}.bin')],timeout=120,operation='authorized_receiving_throughput_probe')
 return time.monotonic()-start
with ThreadPoolExecutor(max_workers=8) as pool:elapsed_parts=list(pool.map(receive,range(8)))
elapsed=time.monotonic()-start;expected=json.loads((B/'resources/transfer_probe/RESULT.json').read_text())['part_sha256']
for p in local.glob('part*.bin'):
 if p.stat().st_size!=8*1024**2 or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Received probe bytes differ')
record={'schema':'confirmation_receiving_throughput_probe.v1','binding':binding,'total_bytes':64*1024**2,'seconds':elapsed,'bytes_per_second':64*1024**2/elapsed,'part_seconds':elapsed_parts,'part_hashes_verified':True,'finished_utc':datetime.now(timezone.utc).isoformat()}
with (local/'RESULT.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'setup_evidence':str(output),'receiving_probe':record}))
