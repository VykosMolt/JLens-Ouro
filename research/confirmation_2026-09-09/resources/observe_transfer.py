#!/usr/bin/env python3
"""Read-only owned-worker input transfer progress; no lifecycle or data mutation."""
import argparse
from datetime import datetime, timezone
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
p = argparse.ArgumentParser()
p.add_argument('--lease', type=Path, required=True)
a = p.parse_args()
root = lease.lease_root(a.lease)
state = json.loads((root / 'LEASE.json').read_text())
handoff.verify_local_sources(state)
if state['status'] != 'running' or not state.get('ssh'):
    raise ValueError('No owned running SSH endpoint')
script = '''import json,pathlib,stat,time
root=pathlib.Path('/workspace/jlens')
if (root/'provider_lease_name').read_text().strip()!=%r:raise ValueError('Wrong owned worker')
rows=[]
for directory in ('input_chunks','banks'):
 parent=root/directory
 for p in parent.rglob('*') if parent.exists() else []:
  if p.is_symlink():raise ValueError('Unexpected linked transfer path')
  s=p.stat()
  if stat.S_ISREG(s.st_mode):rows.append({'path':p.relative_to(root).as_posix(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns})
print(json.dumps({'worker_time':time.time(),'files':rows}))
''' % state['name']
r = run_bounded([*lease.ssh_options(root, state), 'root@' + state['ssh']['host'],
                 'python -c ' + shlex.quote(script)], timeout=30, operation='read_input_transfer_progress')
remote = json.loads(r.stdout)
rows = remote['files']
data = {'schema': 'confirmation_input_transfer_observation.v1',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'binding': handoff.binding(state),
        'remote': remote,
        'duplicate_input_bytes_observed': sum(x['bytes'] for x in rows if '.part' in x['path'] or x['path'].startswith('banks/')),
        'note': 'Sizes of in-progress duplicate input files only; no hash or completed-upload assertion.'}
directory = ROUND / 'resources/transfer_status_retry_02'
directory.mkdir(exist_ok=True)
path = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
with path.open('x') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print(json.dumps({'record': str(path), 'duplicate_input_bytes_observed': data['duplicate_input_bytes_observed'],
                  'files_observed': len(rows)}))
