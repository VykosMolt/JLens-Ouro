"""Copy a fixed completed Huginn handoff while final large-file retrieval runs.

Read-only remotely; no models, estimator banks, tensor deserialization, worker
control, final-collector lock or final-retrieval receipt. Requires phase complete.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
ROUND = Path('/home/moloch/jacobian-lens/research/refit_round_2026-09-07')
DEPLOYMENT = ROUND / 'deployment'
MONITORING = ROUND / 'monitoring/attempt_06'
sys.path.insert(0, str(ROUND / 'monitoring'))
import collect_ouro_handoff06 as ouro_handoff
import prefetch_completed_ouro06_v3 as prefetch_transport
sys.path.insert(0, str(DEPLOYMENT))
import lease

io = lease.io
LEASE_ROOT = lease.LEDGER / 'attempt_06'
LEASE_NAME = 'jlens-refit-14ca9417eb6145c8979fb87ea9fde68c'
POD_ID = 'vzwx0cj43g2uw5'
TREE = 'huginn_evaluation'
CONTRACTS = ouro_handoff.CONTRACTS
EXTRAS = (*CONTRACTS, 'EXPERIMENT_COMPLETE.json')
PREREQUISITE_ROOT = MONITORING / 'ouro_handoff_20260909T110503Z/results'
PREREQUISITES = {
    'ouro_evaluation/OWNER.json': {'bytes': 151299, 'sha256': 'be87010f89cf19849c4d587265a13a2a3832ee9284a2fdf7f203f6c382d2ea1b'},
    'ouro_evaluation/COMPLETE.json': {'bytes': 887, 'sha256': 'eeaa3d7e6fe74f59b9d52e6061a841d1942042766c7e8d1cc614db9657eb3d7e'},
    'controls_evaluation/OWNER.json': {'bytes': 77181, 'sha256': '687b9a8205d1e7a980a661a4c26f3e25c609f805d4e8c53226e3e53ccff892d2'},
    'controls_evaluation/COMPLETE.json': {'bytes': 666, 'sha256': '6dceceac5aa6449b89b60878b9c68df3ba735e8fbb5a74dcb418159bcdb2c04f'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_paths():
    files = set(EXTRAS)
    files.update(f'{TREE}/{name}.json' for name in ('OWNER', 'COMPLETE'))
    files.update(f'{TREE}/population/{name}' for name in ('eligibility.json', 'metadata.json', 'SEAL.json'))
    for seed in (2026090803, 2026090804):
        files.update(f'{TREE}/seeds/{seed}/{name}' for name in
                     ('arrays.npz', 'cache.pt', 'initializations.json', 'summaries.json', 'metadata.json', 'SEAL.json'))
    require(len(files) == 22, 'handoff must contain exactly 22 fixed files')
    return files


def expected_directories(paths):
    return {parent.as_posix() for name in paths for parent in Path(name).parents if parent != Path('.')}


REMOTE_SCRIPT = r'''
from pathlib import Path
import hashlib, json, os, stat
config = CONFIG
workspace = Path(config['workspace'])
def require(ok, message):
    if not ok: raise ValueError(message)
def regular(path, directory=False):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'linked handoff path')
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), 'nonregular handoff path')
def status():
    regular(workspace / 'status.json')
    value = json.loads((workspace / 'status.json').read_text())
    stop = workspace / 'STOP'
    require(value.get('phase') == 'complete' and value.get('setup_complete') is True
            and value.get('lease_name') == config['lease_name']
            and value.get('stop_requested', False) is False
            and not stop.exists() and not stop.is_symlink(), 'worker is not successfully complete')
    provider = workspace / 'provider_lease_name'
    regular(provider)
    provider_name = provider.read_text().strip()
    pod_id = os.environ.get('RUNPOD_POD_ID')
    require(provider_name == config['lease_name'] and (pod_id is None or pod_id == config['pod_id']),
            'remote provider lease or pod changed')
    return value, {'provider_lease_name': provider_name, 'pod_id_if_set': pod_id}
def file_record(path, capture=False):
    regular(path)
    before = path.stat()
    digest, size, body = hashlib.sha256(), 0, bytearray()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block); size += len(block)
            if capture: body.extend(block)
    regular(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'handoff file changed while hashing')
    return {'bytes': size, 'sha256': digest.hexdigest()}, bytes(body)
before, ownership_before = status()
root = workspace / 'results'
regular(root, directory=True)
tree = root / config['tree']
regular(tree, directory=True)
paths = {root / name for name in config['extras']}
directories = {config['tree']}
def walk_error(error): raise error
for directory, dirs, names in os.walk(tree, followlinks=False, onerror=walk_error):
    for child in dirs:
        path = Path(directory) / child
        regular(path, directory=True)
        directories.add(path.relative_to(root).as_posix())
    for child in names: paths.add(Path(directory) / child)
require({path.relative_to(root).as_posix() for path in paths} == set(config['expected'])
        and directories == set(config['directories']), 'unexpected or missing Huginn handoff member')
prerequisites = {}
for name, expected in config['prerequisites'].items():
    actual, _ = file_record(root / name)
    require(actual == expected, 'remote accepted Ouro/control prerequisite changed: ' + name)
    prerequisites[name] = actual
files, experiment = {}, None
for path in sorted(paths):
    name = path.relative_to(root).as_posix()
    record, body = file_record(path, capture=name == 'EXPERIMENT_COMPLETE.json')
    files[name] = record
    if name == 'EXPERIMENT_COMPLETE.json': experiment = json.loads(body)
expected_evaluations = {'ouro_evaluation': prerequisites['ouro_evaluation/COMPLETE.json'],
    'controls_evaluation': prerequisites['controls_evaluation/COMPLETE.json'],
    'huginn_evaluation': files['huginn_evaluation/COMPLETE.json']}
require(isinstance(experiment, dict) and set(experiment) == {'status', 'lease_name', 'completed_utc', 'evaluations'}
        and experiment['status'] == 'complete' and experiment['lease_name'] == config['lease_name']
        and isinstance(experiment['completed_utc'], str) and experiment['evaluations'] == expected_evaluations,
        'experiment completion does not bind all three completed evaluations')
bindings = {'evaluations': expected_evaluations, 'experiment_complete': files['EXPERIMENT_COMPLETE.json'],
            'run_spec_sha256': files['run_spec.json']['sha256'],
            'combined_contract_sha256': files['combined_contract.json']['sha256']}
after, ownership_after = status()
require(ownership_before == ownership_after, 'remote ownership changed during hashing')
print(json.dumps({'status_before': before, 'status_after': after, 'ownership_before': ownership_before,
    'ownership_after': ownership_after, 'files': files, 'prerequisites': prerequisites,
    'experiment_complete': experiment, 'bindings': bindings}, sort_keys=True))
'''


def remote_script(workspace='/workspace/jlens'):
    paths = expected_paths()
    config = {'workspace': workspace, 'lease_name': LEASE_NAME, 'pod_id': POD_ID,
              'tree': TREE, 'extras': EXTRAS, 'expected': sorted(paths),
              'directories': sorted(expected_directories(paths)), 'prerequisites': PREREQUISITES}
    return REMOTE_SCRIPT.replace('CONFIG', repr(config), 1)


def verify_snapshot(snapshot, contracts):
    require(set(snapshot) == {'status_before', 'status_after', 'ownership_before', 'ownership_after',
                             'files', 'prerequisites', 'experiment_complete', 'bindings'}, 'remote snapshot schema changed')
    files = snapshot['files']
    require(set(files) == expected_paths(), 'remote handoff differs from fixed 22-file list')
    for name, record in files.items():
        require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'}
                and type(record['bytes']) is int and record['bytes'] > 0
                and isinstance(record['sha256'], str) and re.fullmatch('[0-9a-f]{64}', record['sha256']),
                'invalid handoff file record: ' + name)
    require(snapshot['prerequisites'] == PREREQUISITES, 'remote prerequisite pins differ')
    require(all(files[name] == contracts[name] for name in CONTRACTS), 'remote contract bytes differ from local deployment')
    evaluations = {'ouro_evaluation': PREREQUISITES['ouro_evaluation/COMPLETE.json'],
                   'controls_evaluation': PREREQUISITES['controls_evaluation/COMPLETE.json'],
                   'huginn_evaluation': files['huginn_evaluation/COMPLETE.json']}
    experiment = snapshot['experiment_complete']
    require(isinstance(experiment, dict) and set(experiment) == {'status', 'lease_name', 'completed_utc', 'evaluations'}
            and experiment['status'] == 'complete' and experiment['lease_name'] == LEASE_NAME
            and isinstance(experiment['completed_utc'], str) and experiment['evaluations'] == evaluations,
            'experiment completion schema or evaluation bindings changed')
    bindings = {'evaluations': evaluations, 'experiment_complete': files['EXPERIMENT_COMPLETE.json'],
                'run_spec_sha256': contracts['run_spec.json']['sha256'],
                'combined_contract_sha256': contracts['combined_contract.json']['sha256']}
    require(snapshot['bindings'] == bindings, 'handoff bindings differ from copied files and accepted prerequisites')
    for suffix in ('before', 'after'):
        status = snapshot['status_' + suffix]
        require(status.get('phase') == 'complete' and status.get('setup_complete') is True
                and status.get('lease_name') == LEASE_NAME and status.get('stop_requested', False) is False,
                'handoff requires the successfully completed worker')
        ownership = snapshot['ownership_' + suffix]
        require(set(ownership) == {'provider_lease_name', 'pod_id_if_set'}
                and ownership['provider_lease_name'] == LEASE_NAME and ownership['pod_id_if_set'] in (None, POD_ID),
                'remote ownership binding differs')
    require(snapshot['ownership_before'] == snapshot['ownership_after'], 'remote ownership changed during hashing')


def local_inventory(root):
    root = io._no_links(root)
    require(root.is_dir(), 'missing local copied tree')
    directories = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_dir()}
    require(directories == expected_directories(expected_paths()), 'unexpected or missing local handoff directory')
    return ouro_handoff.local_inventory(root)


def local_lease():
    state = io._json(io._no_links(LEASE_ROOT / 'LEASE.json'))
    require(state.get('name') == LEASE_NAME and state.get('pod_id') == POD_ID
            and state.get('status') == 'running' and state.get('halt_requested', False) is False,
            'local lease is not the running owned attempt 06 pod')
    final = LEASE_ROOT / 'RETRIEVAL_VERIFIED.json'
    require(not final.exists() and not final.is_symlink(), 'final retrieval is already available; use its verified tree')
    return state


def local_inputs():
    contracts = {name: io._record(DEPLOYMENT / name) for name in CONTRACTS}
    for name, record in PREREQUISITES.items():
        io._verify_record(PREREQUISITE_ROOT / name, record)
    return contracts


def publish_copy_receipt(destination, value):
    # Publish only a fully written receipt, using an exclusive link. Interrupted
    # writing leaves a pending file and never a partial COPY_VERIFIED marker.
    pending, complete = destination / 'COPY_PENDING.json', destination / 'COPY_VERIFIED.json'
    io._new_json(pending, value)
    os.link(pending, complete, follow_symlinks=False)
    try:
        io._fsync_dir(destination)
    except BaseException:
        if complete.exists() and os.path.samefile(complete, pending):
            complete.unlink()
        raise



def run_bounded(command, *, operation, timeout, processes):
    """Reap our complete process group before returning or propagating failure."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    receipt = {'operation': operation, 'pid': process.pid, 'timeout_seconds': timeout,
               'process_group_quiescent': False}
    processes.append(receipt)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as error:
        receipt['error_type'] = type(error).__name__
        raise
    finally:
        try:
            prefetch_transport.stop_process(process)
            receipt['process_group_quiescent'] = True
        except BaseException as error:
            receipt['cleanup_error_type'] = type(error).__name__
            receipt['cleanup_error'] = str(error)
            raise
        finally:
            receipt['returncode'] = process.returncode
            process.stdout.close()
            process.stderr.close()


def collect():
    state = local_lease()
    contracts = local_inputs()
    source = io._record(Path(__file__))
    reused_sources = {str(Path(module.__file__)): io._record(Path(module.__file__))
                      for module in (ouro_handoff, lease, io, prefetch_transport)}
    processes = []
    command = [*lease.ssh_options(LEASE_ROOT, state), 'root@' + state['ssh']['host'],
               'python -c ' + shlex.quote(remote_script())]

    def snapshot():
        current = local_lease()
        require(current['ssh'] == state['ssh'] and current['ssh_identity'] == state['ssh_identity'], 'local SSH lease endpoint changed')
        result = run_bounded(command, operation='snapshot', timeout=120, processes=processes)
        value = json.loads(result.stdout)
        verify_snapshot(value, contracts)
        local_lease()
        return value

    before = snapshot()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = io._no_links(MONITORING / ('huginn_handoff_' + stamp + '_' + uuid.uuid4().hex[:8]))
    destination.mkdir(parents=True, exist_ok=False)
    try:
        io._new_json(destination / 'REMOTE_BEFORE.json', before)
        listing = destination / 'FILES_FROM'
        with listing.open('xb') as handle:
            handle.write(b'\0'.join(name.encode() for name in sorted(expected_paths())) + b'\0')
        copied = destination / 'results'
        copied.mkdir()
        transport = shlex.join(lease.ssh_options(LEASE_ROOT, state))
        run_bounded(['rsync', '-rlt', '--partial', '--timeout=60', '--from0',
                        '--files-from=' + str(listing), '-e', transport,
                        f"root@{state['ssh']['host']}:/workspace/jlens/results/", str(copied) + '/'],
                    operation='transfer', timeout=900, processes=processes)
        after = snapshot()
        io._new_json(destination / 'REMOTE_AFTER.json', after)
        require(before['files'] == after['files'] and before['prerequisites'] == after['prerequisites']
                and before['bindings'] == after['bindings'], 'remote handoff hashes or bindings changed across copy')
        require(local_inventory(copied) == before['files'], 'local copied 22-file inventory differs from remote bytes')
        require(io._json(copied / 'EXPERIMENT_COMPLETE.json') == before['experiment_complete'], 'copied experiment marker differs from bound payload')
        require(local_inputs() == contracts, 'local contracts or accepted prerequisites changed during copy')
        local_lease()
        io._verify_record(Path(__file__), source)
        for name, record in reused_sources.items():
            io._verify_record(Path(name), record)
        publish_copy_receipt(destination, {'schema': 'huginn_small_handoff.v1', 'status': 'passed',
            'lease_name': LEASE_NAME, 'pod_id': POD_ID, 'files': before['files'],
            'prerequisites': PREREQUISITES, 'local_prerequisite_root': str(PREREQUISITE_ROOT),
            'bindings': before['bindings'], 'source': source, 'source_path': str(Path(__file__)),
            'reused_sources': reused_sources, 'processes': processes,
            'scope': 'Small handoff of 17 completed Huginn evaluation files, the complete experiment marker and four frozen contracts. '
                     'All 22 files hashed remotely before/after copy and locally; four accepted Ouro/control prerequisite files '
                     'verified remotely and locally before/after. No model/estimator loading, scientific interpretation, '
                     'worker control, final collector lock or RETRIEVAL_VERIFIED publication. '
                     'The complete final retrieved tree still requires separate acceptance.'})
    except BaseException as error:
        require(not (destination / 'COPY_VERIFIED.json').exists(), 'cannot mark a published handoff incomplete')
        io._new_json(destination / 'INCOMPLETE.json', {'status': 'incomplete', 'lease_name': LEASE_NAME,
            'pod_id': POD_ID, 'stopped_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'error_type': type(error).__name__, 'reason': str(error), 'processes': processes,
            'scope': 'Interrupted small handoff; no COPY_VERIFIED receipt. Partial copies are outside final retrieved.'})
        raise
    return {'directory': str(destination), 'files': len(before['files']),
            'bytes': sum(row['bytes'] for row in before['files'].values()), 'status': 'passed'}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(collect(), indent=2))


if __name__ == '__main__':
    main()
