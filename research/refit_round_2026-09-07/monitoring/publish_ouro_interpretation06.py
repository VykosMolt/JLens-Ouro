"""Publish the accepted Ouro interpretation to the existing worker.

This releases that worker into the frozen Huginn GPU parity/memory/budget gate
and, if it passes, its N100 fit and two-seed evaluation under the current lease.

No P4 staging or execution occurs here. Root must settle the optional P4 branch
before invoking this operator; an active P4 process always refuses publication.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys

ROOT = Path('/home/moloch/jacobian-lens/research/refit_round_2026-09-07')
sys.path.insert(0, str(ROOT / 'deployment'))
import lease

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--p4-outcome', choices=('completed', 'incomplete', 'not_executed'), required=True)
args = parser.parse_args()
base = ROOT / 'monitoring/attempt_06'
prepared = base / 'OURO_INTERPRETATION.prepared.json'
assert lease.io._file_hash(prepared) == 'bc29084b6e87d685dce968a776740a8939a345a6c00d3e2f1d31d25960a6855b'
value = lease.io._json(prepared)
handoff = lease.io._json(base / 'ouro_handoff_20260909T110503Z/COPY_VERIFIED.json')
assert value['bindings'] == handoff['bindings'] and value['lease_name'] == handoff['lease_name']
assert value['status'] == 'interpreted' and len(value['findings'].strip()) >= 100
lease.io._verify_record(ROOT / 'analysis/ouro_run01/COMPLETE.json', value['analysis_complete'])
lease.io._verify_record(ROOT / 'analysis/ouro_run01_review/actual_p2_result_review.json', value['independent_audit'])
value.pop('publication_state')
value['p4_outcome_before_release'] = args.p4_outcome
value['publication_prepared_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
record = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
lease_root = lease.LEDGER / 'attempt_06'
state = lease.io._json(lease_root / 'LEASE.json')
assert state['name'] == handoff['lease_name'] and state['pod_id'] == handoff['pod_id']
assert state['status'] == 'running'
now = datetime.datetime.now(datetime.timezone.utc)
assert now < lease.io._deadline(state['work_deadline_utc'])
config = {'lease_name': state['name'], 'pod_id': state['pod_id'], 'bindings': value['bindings'],
          'payload_record': record, 'p4_outcome': args.p4_outcome}
remote = r'''
from pathlib import Path
import datetime, hashlib, json, os, stat, sys, uuid
config = CONFIG
root = Path('/workspace/jlens')
def regular(path):
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('linked publication path')
    if not stat.S_ISREG(path.stat().st_mode): raise ValueError('nonregular publication input')
def record(path):
    regular(path)
    body = path.read_bytes()
    return {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
def check():
    regular(root / 'status.json')
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'awaiting_ouro_interpretation'
    assert status['lease_name'] == config['lease_name']
    assert status.get('stop_requested', False) is False
    assert status['required_bindings'] == config['bindings']
    regular(root / 'provider_lease_name')
    assert (root / 'provider_lease_name').read_text().strip() == config['lease_name']
    if os.environ.get('RUNPOD_POD_ID') is not None:
        assert os.environ['RUNPOD_POD_ID'] == config['pod_id']
    for name in ('STOP', 'OURO_INTERPRETATION.json'):
        assert not (root / name).exists() and not (root / name).is_symlink()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: cmd = (p / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        assert b'/workspace/jlens/p4/workflow.py' not in cmd, 'P4 is still running'
    p4 = root / 'results/probe_extension'
    if config['p4_outcome'] == 'not_executed':
        assert not p4.exists() and not p4.is_symlink()
        assert not (root / 'p4').exists() and not (root / 'p4').is_symlink()
    else:
        regular(p4 / 'outer_command.json')
        outer = json.loads((p4 / 'outer_command.json').read_text())
        assert outer['quiescent'] is True and outer['remaining_p4_processes'] == []
        if config['p4_outcome'] == 'completed':
            assert outer['status'] == 'passed'
            regular(p4 / 'supervisor.json')
    for family, key in [('ouro_evaluation', 'main_complete'), ('controls_evaluation', 'controls_complete')]:
        assert record(root / 'results' / family / 'COMPLETE.json') == config['bindings'][key]
    for name, key in [('run_spec.json', 'run_spec_sha256'), ('combined_contract.json', 'combined_contract_sha256')]:
        assert record(root / 'results' / name)['sha256'] == config['bindings'][key]
    return status
before = check()
body = sys.stdin.buffer.read(config['payload_record']['bytes'] + 1)
assert {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} == config['payload_record']
value = json.loads(body)
assert value['status'] == 'interpreted' and value['lease_name'] == config['lease_name']
assert value['bindings'] == config['bindings'] and len(value['findings'].strip()) >= 100
temporary = root / ('.OURO_INTERPRETATION.' + uuid.uuid4().hex + '.tmp')
with temporary.open('xb') as stream:
    stream.write(body); stream.flush(); os.fsync(stream.fileno())
check()
target = root / 'OURO_INTERPRETATION.json'
os.link(temporary, target)
temporary.unlink()
fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
try: os.fsync(fd)
finally: os.close(fd)
assert record(target) == config['payload_record']
print(json.dumps({'status': 'published', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'lease_name': config['lease_name'], 'pod_id': config['pod_id'], 'payload': config['payload_record'],
    'p4_outcome': config['p4_outcome'], 'worker_before': before}), flush=True)
'''.replace('CONFIG', repr(config))
destination = base / ('ouro_release_' + now.strftime('%Y%m%dT%H%M%SZ'))
lease.io._no_links(destination).mkdir(exist_ok=False)
with (destination / 'OURO_INTERPRETATION.json').open('xb') as stream:
    stream.write(payload)
lease.io._new_json(destination / 'OPERATOR.json', {'source': lease.io._record(Path(__file__)), 'payload': record,
    'scope': 'Publishes accepted Ouro interpretation, releasing the existing worker into its frozen Huginn GPU parity/memory/budget gate and, if passed, N100 fit and two-seed evaluation under the current lease. No P4 action or new cloud allocation.'})
result = subprocess.run([*lease.ssh_options(lease_root, state), 'root@' + state['ssh']['host'],
    'python -c ' + shlex.quote(remote)], input=payload, capture_output=True, check=True, timeout=60)
published = json.loads(result.stdout)
assert published['status'] == 'published' and published['payload'] == record
lease.io._new_json(destination / 'PUBLICATION.json', published)
print(json.dumps({'directory': str(destination), **published}, indent=2))
