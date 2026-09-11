"""Copy small completed-fit receipts; do not read any model or estimator binary."""
from pathlib import Path
import argparse
import base64
import datetime
import hashlib
import json
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit', type=int, choices=range(1, 6), required=True)
args = parser.parse_args()
deployment = Path(__file__).resolve().parents[1] / 'deployment'
sys.path.insert(0, str(deployment))
import lease

root = lease.LEDGER / 'attempt_06'
state = lease.io._json(root / 'LEASE.json')
assert state['name'] == 'jlens-refit-14ca9417eb6145c8979fb87ea9fde68c'
assert state['pod_id'] == 'vzwx0cj43g2uw5'
script = r'''from pathlib import Path
import base64, hashlib, json, re
fit_id = FIT_ID
workspace = Path('/workspace/jlens')
status = json.loads((workspace / 'status.json').read_text())
assert status['lease_name'] == 'jlens-refit-14ca9417eb6145c8979fb87ea9fde68c'
r = workspace / 'results'
fit = r / 'ouro' / ('fit_%02d' % fit_id)
log = r / 'logs/ouro_fits.log'
log_bytes = log.read_bytes()
assert len(log_bytes) < 16000000
complete_lines = []
for line in log_bytes.splitlines():
    try:
        row = json.loads(line)
    except (ValueError, UnicodeError):
        continue
    if isinstance(row, dict) and row.get('fit_id') == fit_id and row.get('status') == 'complete':
        assert row['n_done'] == row['next_idx'] == 100
        complete_lines.append(row)
assert len(complete_lines) == 1, 'Require the successful finalization stdout record'
paths = [r / 'ouro/RUN_IDENTITY.json', fit / 'OWNER.json', fit / 'LATEST.json', fit / 'COMPLETE.json']
for filename, parent in [('LATEST.json', 'checkpoints'), ('COMPLETE.json', 'final')]:
    pointer = json.loads((fit / filename).read_text())
    assert pointer['n_done'] == pointer['next_idx'] == 100
    assert re.fullmatch(parent + r'/cursor_000100_[0-9a-f]{32}', pointer['generation'])
    generation = fit / pointer['generation']
    paths.extend([generation / 'SEAL.json', generation / 'metadata.json'])
assert len(paths) == len(set(paths)) == 8
rows = []
for p in paths + [log]:
    assert p.is_file() and all(not q.is_symlink() for q in [p, *p.parents])
    body = log_bytes if p == log else p.read_bytes()
    assert len(body) < 16000000
    rows.append({'path': str(p.relative_to(r)), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(), 'data': base64.b64encode(body).decode()})
print(json.dumps({'files': rows, 'completion_stdout': complete_lines[0]}))
'''.replace('FIT_ID', str(args.fit))
result = subprocess.run(
    [*lease.ssh_options(root, state), 'root@' + state['ssh']['host'], 'python -c ' + shlex.quote(script)],
    capture_output=True, text=True, check=True, timeout=45,
)
captured = json.loads(result.stdout)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
destination = deployment.parent / 'monitoring/attempt_06' / f'fit_{args.fit:02d}_{stamp}'
destination.mkdir(parents=True, exist_ok=False)
for row in captured['files']:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    body = base64.b64decode(row.pop('data'), validate=True)
    assert len(body) == row['bytes'] and hashlib.sha256(body).hexdigest() == row['sha256']
    target = destination / 'results' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        handle.write(body)
lease.io._new_json(destination / 'COPY_VERIFIED.json', {
    'status': 'passed', 'lease_name': state['name'], 'pod_id': state['pod_id'],
    'fit_id': args.fit, **captured,
    'scope': 'Eight immutable fit provenance JSONs and a captured log prefix; no estimator binaries read or independently verified, no recovery analysis, and not final experiment retrieval.',
})
print(json.dumps({'directory': str(destination), 'files': captured['files'], 'completion_stdout': captured['completion_stdout']}, indent=2))
