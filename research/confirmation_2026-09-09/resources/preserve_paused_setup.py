#!/usr/bin/env python3
"""Read-only preservation of an owned worker that never finished input setup."""
import argparse, base64, hashlib, json, shlex, sys
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / 'controller'))
import lease
import artifact_handoff as handoff
from transport import run_bounded
p = argparse.ArgumentParser()
p.add_argument('--lease', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = lease.lease_root(a.lease)
state = json.loads((root / 'LEASE.json').read_text())
handoff.verify_local_sources(state)
if state['status'] != 'running' or not state.get('ssh') or state.get('known_manifests'):
    raise ValueError('Worker is not the expected paused setup')
script = r"""import base64,hashlib,json,pathlib,os,stat,time
r=pathlib.Path('/workspace/jlens')
if (r/'provider_lease_name').read_text().strip()!=EXPECTED:raise ValueError('Wrong owned worker')
if any((r/name).exists() for name in ('BANKS_READY.json','artifact_index.json','DEVELOPMENT_ACCEPTED.json')):raise ValueError('Worker progressed beyond paused input setup')
def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def scan(p,keep):
 for parent in (p,*p.parents):
  if parent.is_symlink():raise ValueError('Linked evidence path')
 before=p.stat()
 if not stat.S_ISREG(before.st_mode) or (keep and before.st_size>8*1024**2):raise ValueError('Unexpected evidence requiring separate preservation')
 digest=hashlib.sha256();parts=[]
 with p.open('rb') as f:
  if signature(os.fstat(f.fileno()))!=signature(before):raise ValueError('Evidence changed before open')
  while block:=f.read(8*1024**2):
   digest.update(block)
   if keep:parts.append(block)
  if signature(os.fstat(f.fileno()))!=signature(before):raise ValueError('Evidence changed during read')
 if signature(p.stat())!=signature(before):raise ValueError('Evidence path changed')
 row={'bytes':before.st_size,'sha256':digest.hexdigest()}
 if keep:row['base64']=base64.b64encode(b''.join(parts)).decode()
 return row
files={};inputs={}
for name in ('provider_lease_name','status.json','bootstrap_stdout.log','WORKER_FAILURE.json'):
 p=r/name
 if p.exists():files[name]=scan(p,True)
for name in ('setup_logs','results'):
 for p in sorted((r/name).rglob('*')):
  if p.is_file():files[p.relative_to(r).as_posix()]=scan(p,True)
for name in ('input_chunks','banks'):
 for p in sorted((r/name).rglob('*')):
  if p.is_file():inputs[p.relative_to(r).as_posix()]=scan(p,False)
print(json.dumps({'worker_time':time.time(),'lease_name':EXPECTED,'files':files,'duplicate_input_records':inputs,'banks_ready_present':False,'artifact_index_present':False,'development_ack_present':False}))
""".replace('EXPECTED', repr(state['name']))
raw = run_bounded([*lease.ssh_options(root, state), 'root@' + state['ssh']['host'], 'python -c ' + shlex.quote(script)], timeout=180, operation='read_only_preserve_paused_setup').stdout
evidence = json.loads(raw)
if evidence['lease_name'] != state['name']:
    raise ValueError('Evidence binding differs')
for entry in evidence['files'].values():
    data = base64.b64decode(entry['base64'], validate=True)
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Returned small evidence differs')
result = {'schema':'confirmation_paused_setup_preservation.v1','binding':handoff.binding(state),'collected_utc':datetime.now(timezone.utc).isoformat(),'remote':evidence,'source_record':{'bytes':Path(__file__).stat().st_size,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'interpretation':'No ready marker, artifact index or development acknowledgement existed. Remote input duplicates are hash-recorded; complete original banks and chunks remain local. This record does not authorize upload or claim scientific acceptance.'}
with a.out.open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'preserved':str(a.out),'small_files':len(evidence['files']),'duplicate_inputs':len(evidence['duplicate_input_records'])}))
