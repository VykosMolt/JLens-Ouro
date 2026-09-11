"""Copy the completed main evaluation while the frozen control fits continue."""
from pathlib import Path
import argparse
import datetime
import json
import shlex
import subprocess
import sys

import collect_ouro_handoff06 as handoff


EXTRAS = (*handoff.CONTRACTS, "initial_budget_projection.json",
          "logs/ouro_fits.log", "logs/ouro_evaluation.log")
EXPECTED = {name for name in handoff.expected_paths()
            if name.startswith("ouro_evaluation/")} | set(EXTRAS)
REMOTE = r'''
from pathlib import Path
import hashlib, json, os, stat
config = CONFIG
workspace = Path('/workspace/jlens')
def regular(path, directory=False):
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('linked main-copy path')
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('nonregular main-copy path')
def status():
    path = workspace / 'status.json'
    regular(path)
    value = json.loads(path.read_text())
    if (value.get('phase') != 'control_fits'
            or value.get('lease_name') != config['lease_name']
            or value.get('stop_requested') is not False):
        raise ValueError('worker is not actively fitting the controls')
    return value
before = status()
root = workspace / 'results'
regular(root, directory=True)
tree = root / 'ouro_evaluation'
regular(tree, directory=True)
paths = {root / name for name in config['extras']}
def walk_error(error): raise error
for directory, dirs, names in os.walk(tree, followlinks=False, onerror=walk_error):
    for child in dirs: regular(Path(directory) / child, directory=True)
    for child in names: paths.add(Path(directory) / child)
if {p.relative_to(root).as_posix() for p in paths} != set(config['expected']):
    raise ValueError('unexpected or missing main-copy member')
files = {}
for path in sorted(paths):
    regular(path)
    h, size = hashlib.sha256(), 0
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
            size += len(block)
    files[path.relative_to(root).as_posix()] = {'bytes': size, 'sha256': h.hexdigest()}
print(json.dumps({'status_before': before, 'status_after': status(), 'files': files}, sort_keys=True))
'''


def verify_snapshot(snapshot, contracts):
    if len(EXPECTED) != 31 or set(snapshot['files']) != EXPECTED:
        raise ValueError('main-copy inventory differs from the fixed 31 files')
    for key in ('status_before', 'status_after'):
        status = snapshot[key]
        if (status.get('phase') != 'control_fits'
                or status.get('lease_name') != handoff.LEASE_NAME
                or status.get('stop_requested') is not False):
            raise ValueError('invalid active main-copy status')
    if any(snapshot['files'][name] != record for name, record in contracts.items()):
        raise ValueError('remote contract bytes differ from local deployment')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(handoff.DEPLOYMENT))
    import lease

    root = lease.LEDGER / 'attempt_06'
    state = lease.io._json(root / 'LEASE.json')
    if state['name'] != handoff.LEASE_NAME or state['pod_id'] != handoff.POD_ID:
        raise ValueError('unexpected lease or pod')
    contracts = {name: lease.io._record(handoff.DEPLOYMENT / name) for name in handoff.CONTRACTS}
    config = {'lease_name': handoff.LEASE_NAME, 'extras': EXTRAS, 'expected': sorted(EXPECTED)}
    command = [*lease.ssh_options(root, state), 'root@' + state['ssh']['host'],
               'python -c ' + shlex.quote(REMOTE.replace('CONFIG', repr(config)))]

    def snapshot():
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        value = json.loads(result.stdout)
        verify_snapshot(value, contracts)
        return value

    before = snapshot()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = handoff.DEPLOYMENT.parent / 'monitoring/attempt_06' / ('main_only_' + stamp)
    lease.io._no_links(destination)
    destination.mkdir(parents=True, exist_ok=False)
    lease.io._new_json(destination / 'REMOTE_BEFORE.json', before)
    listing = destination / 'FILES_FROM'
    with listing.open('xb') as handle:
        handle.write(b'\0'.join(name.encode() for name in sorted(EXPECTED)) + b'\0')
    copied = destination / 'results'
    copied.mkdir()
    subprocess.run(['rsync', '-rlt', '--partial', '--timeout=60', '--from0',
                    '--files-from=' + str(listing), '-e', shlex.join(lease.ssh_options(root, state)),
                    f"root@{state['ssh']['host']}:/workspace/jlens/results/", str(copied) + '/'],
                   check=True, timeout=900)
    after = snapshot()
    lease.io._new_json(destination / 'REMOTE_AFTER.json', after)
    if before['files'] != after['files'] or handoff.local_inventory(copied) != before['files']:
        raise ValueError('main-copy remote or local bytes changed or differ')
    for name, record in contracts.items():
        lease.io._verify_record(handoff.DEPLOYMENT / name, record)
    lease.io._new_json(destination / 'COPY_VERIFIED.json', {
        'status': 'passed', 'lease_name': handoff.LEASE_NAME, 'pod_id': handoff.POD_ID,
        'files': before['files'], 'main_complete': before['files']['ouro_evaluation/COMPLETE.json'],
        'scope': 'Interim main-only copy: all 24 completed main evaluation files, four contracts, '
                 'initial budget and two finished logs, hashed remotely before/after and locally. '
                 'No binary deserialization, interpretation, Huginn receipt or final retrieval claim.',
    })
    print(json.dumps({'directory': str(destination), 'files': len(EXPECTED),
                      'bytes': sum(row['bytes'] for row in before['files'].values()),
                      'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
