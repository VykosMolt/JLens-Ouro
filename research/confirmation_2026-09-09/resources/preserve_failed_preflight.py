"""Preserve the terminal, pre-development failure of attempt03; no remote writes."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / 'controller'))
import lease
import artifact_handoff as handoff
from transport import run_bounded

root = ROUND / 'cloud_leases/attempt_03'
state = json.loads((root / 'LEASE.json').read_text())
handoff.verify_local_sources(state)
assert state['status'] == 'running'
assert state['computation_finished']['outcome'] == 'failed'
assert state['computation_finished']['manifest']['files'] == {}
assert not state.get('stage_acceptances') and not state.get('retrieval_accepted')
pid = json.loads((root / 'launch/BOOTSTRAP_STARTED.json').read_text())['bootstrap_pid']
script = r'''
import base64, hashlib, json, os, pathlib, stat
r = pathlib.Path('/workspace/jlens')
assert (r/'provider_lease_name').read_text().strip() == EXPECTED
proc = pathlib.Path('/proc')/str(PID)
if proc.exists():
    assert (proc/'stat').read_text().split(') ',1)[1].split()[0] == 'Z', 'Worker is not quiescent'
status = json.loads((r/'status.json').read_text())
assert status['phase'] == 'computation_finished' and status['outcome'] == 'failed'
def sig(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read(p):
    assert not any(x.is_symlink() for x in (p,*p.parents))
    before=p.stat(); assert stat.S_ISREG(before.st_mode) and before.st_size <= 16*1024**2
    with p.open('rb') as f:
        assert sig(os.fstat(f.fileno())) == sig(before)
        b=f.read(); assert sig(os.fstat(f.fileno())) == sig(before)
    assert sig(p.stat()) == sig(before)
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
files={}
for rel in ('provider_lease_name','status.json','bootstrap_stdout.log','worker_config.json',
            'BANKS_READY.json','artifact_index.json','bundle/evaluation/worker.py',
            'bundle/frozen/run_spec.json','bundle/frozen/output_contract.json'):
    p=r/rel
    if p.exists(): files[rel]=read(p)
for rel in ('setup_logs','manifests','results'):
    for p in sorted((r/rel).rglob('*')):
        if p.is_file(): files[p.relative_to(r).as_posix()]=read(p)
assert not (r/'DEVELOPMENT_ACCEPTED.json').exists()
print(json.dumps({'worker_quiescent':True,'bootstrap_pid':PID,'development_ack_present':False,'files':files}))
'''.replace('EXPECTED', repr(state['name'])).replace('PID', str(pid))
raw = run_bounded([*lease.ssh_options(root, state), 'root@'+state['ssh']['host'],
                   'python -c '+shlex.quote(script)], timeout=60,
                  operation='preserve_terminal_preflight_failure').stdout
evidence = json.loads(raw)
for entry in evidence['files'].values():
    value = base64.b64decode(entry['base64'], validate=True)
    assert len(value) == entry['bytes'] and hashlib.sha256(value).hexdigest() == entry['sha256']
out = {'schema':'confirmation_failed_preflight_preservation.v1',
       'recorded_utc':datetime.now(timezone.utc).isoformat(), 'binding':handoff.binding(state),
       'failed_final':state['computation_finished'], 'remote':evidence,
       'note':'Terminal failure before development or confirmation; not scientific acceptance.'}
path = ROUND/'resources/ATTEMPT03_PREFLIGHT_FAILURE_PRESERVED.json'
with path.open('x') as f:
    json.dump(out,f,indent=2); f.write('\n')
print(json.dumps({'record':str(path),'files_preserved':len(evidence['files']),
                  'worker_quiescent':evidence['worker_quiescent']}))
