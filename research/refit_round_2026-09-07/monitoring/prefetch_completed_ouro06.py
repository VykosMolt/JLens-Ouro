"""Prefetch only 14 sealed Ouro/control binaries while Huginn fitting is active.

Remote operations are reads only. Files are verified outside retrieved, then
published atomically under the final collector's short exclusive lock. This
does not run final retrieval, publish RETRIEVAL_VERIFIED, or control the worker.
The default command only describes the frozen whitelist; execution needs run.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROUND = Path('/home/moloch/jacobian-lens/research/refit_round_2026-09-07')
sys.path.insert(0, str(ROUND / 'deployment'))
import lease

io = lease.io
LEASE_ROOT = lease.LEDGER / 'attempt_06'
NAME = 'jlens-refit-14ca9417eb6145c8979fb87ea9fde68c'
POD = 'vzwx0cj43g2uw5'
REMOTE = '/workspace/jlens/results'
TOTAL_BYTES = 38429057822
POLL_SECONDS = 5.0
STATUS_TIMEOUT = 20.0
HASH_TIMEOUT = 900.0
TRANSFER_TIMEOUT = 3600.0
PINS = [
    ('fit_01_20260908T235301Z', 1990, 'a8465a6b0caa50b312cd0bdf7ade6a952b717d7e1cd504b6d21af56a0c62caed', 'ouro/fit_01'),
    ('fit_02_20260909T014024Z', 1990, '49e19883b4a2b9054fda25576cc1639e54555647b05f4eabfe034b90c2079fcc', 'ouro/fit_02'),
    ('fit_03_20260909T032940Z', 1990, '1bc3d560e95f042aabdba46ae94388b3083c496614e47af0b00ac489ee8ff06a', 'ouro/fit_03'),
    ('fit_04_20260909T051802Z', 1990, 'dd5e8d1cea2420cf68a610dd1f55f7a2043391a7ac4b86c9cc654732d533041f', 'ouro/fit_04'),
    ('fit_05_20260909T070535Z', 1991, '7dd387f3cfca9e929fee7fa0c7414989f1a20300ffb78de69bdd8f72673316fa', 'ouro/fit_05'),
    ('ouro_penultimate_20260909T085729Z', 2241, '65539980465006027901268d1ee1c8c20ffc3e97f4651235fb8cbce3932861d1', 'controls/ouro_penultimate/fit_01'),
    ('ouro_positions_20260909T110032Z', 2221, 'd8e7fb4ae0338cd539ae46fdf019ee98a85ffb063d355dfdf3c0e8a78b9fa468', 'controls/ouro_positions/fit_01'),
]


class StopPrefetch(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def checked_record(path, guard=lambda: None):
    path = io._no_links(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'nonregular input: ' + str(path))
    digest, size, checked = hashlib.sha256(), 0, time.monotonic()
    guard()
    with path.open('rb') as handle:
        require(signature(os.fstat(handle.fileno())) == signature(before), 'input changed before opening')
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
            if time.monotonic() - checked >= POLL_SECONDS:
                guard()
                checked = time.monotonic()
        require(signature(os.fstat(handle.fileno())) == signature(before), 'input changed during hashing')
    require(signature(path.stat()) == signature(before), 'input pathname changed during hashing')
    guard()
    return {'bytes': size, 'sha256': digest.hexdigest()}, signature(before)


def expected_inventory():
    """Pinned accepted small receipts determine every permitted remote path."""
    metadata, binaries, consumed = {}, {}, {}
    calibration = None
    for folder, size, sha, relative in PINS:
        base = ROUND / 'monitoring/attempt_06' / folder
        copy_path = base / 'COPY_VERIFIED.json'
        copy_record = {'bytes': size, 'sha256': sha}
        io._verify_record(copy_path, copy_record)
        consumed[str(copy_path)] = copy_record
        receipt = io._json(copy_path)
        require(receipt['status'] == 'passed' and receipt['lease_name'] == NAME and receipt['pod_id'] == POD,
                'accepted receipt lease identity')
        require(len(receipt['files']) == 9 and len({r['path'] for r in receipt['files']}) == 9,
                'accepted receipt membership')
        root = base / 'results'
        records = {}
        for row in receipt['files']:
            path = Path(row['path'])
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe accepted path')
            record = {key: row[key] for key in ('bytes', 'sha256')}
            io._verify_record(root / path, record)
            consumed[str(root / path)] = record
            if path.suffix == '.json':
                require(row['path'] not in metadata or metadata[row['path']] == record, 'conflicting shared metadata')
                metadata[row['path']] = records[row['path']] = record
        fit = root / relative
        owner = io._json(fit / 'OWNER.json')
        identity = owner['identity']
        identity_sha = io._digest(identity)
        require(owner == {'schema_version': 1, 'identity': identity, 'fit_identity_sha256': identity_sha},
                'accepted owner binding')
        require(identity['n_prompts'] == 100 and identity['cpu_test'] is False, 'nonproduction accepted owner')
        if relative == 'ouro/fit_01':
            calibration = {key: identity[key] for key in ('ordered_prompts_sha256', 'prompt_sha256', 'token_lengths', 'n_valid')}
        if relative.startswith('controls/'):
            require(all(identity[key] == value for key, value in calibration.items()), 'control calibration differs from fit01')
        pointers, metas = {}, {}
        for filename, kind, parent, binary, stdout_key in (
            ('LATEST.json', 'checkpoint', 'checkpoints', 'state.pt', 'checkpoint_path'),
            ('COMPLETE.json', 'final', 'final', 'lens.pt', 'lens_path'),
        ):
            pointer = io._json(fit / filename)
            require(pointer['kind'] == kind and pointer['fit_identity_sha256'] == identity_sha
                    and pointer['n_done'] == pointer['next_idx'] == 100, 'accepted pointer identity/count')
            require(re.fullmatch(parent + r'/cursor_000100_[0-9a-f]{32}', pointer['generation']), 'accepted generation name')
            generation = fit / pointer['generation']
            io._verify_record(generation / 'SEAL.json', pointer['seal'])
            seal, meta = io._json(generation / 'SEAL.json'), io._json(generation / 'metadata.json')
            require(seal['kind'] == kind and seal['fit_identity_sha256'] == identity_sha and seal['n_done'] == 100,
                    'accepted seal identity')
            require(set(seal['files']) == {binary, 'metadata.json'}, 'accepted sealed members')
            io._verify_record(generation / 'metadata.json', seal['files']['metadata.json'])
            require(meta['identity'] == identity and meta['fit_identity_sha256'] == identity_sha
                    and meta['kind'] == kind and meta['n_done'] == meta['next_idx'] == 100,
                    'accepted metadata identity')
            require(meta['completed_prompt_sha256'] == identity['prompt_sha256']
                    and meta['completed_prefix_sha256'] == io._digest(identity['prompt_sha256'])
                    and len(meta['diagnostics']) == 100, 'accepted completed prefix')
            path = relative + '/' + pointer['generation'] + '/' + binary
            require(receipt['completion_stdout'][stdout_key] == REMOTE + '/' + path, 'accepted completion path')
            require(path not in binaries, 'duplicate binary whitelist member')
            binaries[path] = seal['files'][binary]
            pointers[kind], metas[kind] = pointer, meta
        require(metas['final']['fp32_checkpoint'] == pointers['checkpoint']
                and metas['final']['diagnostics'] == metas['checkpoint']['diagnostics'], 'final/checkpoint binding')
    require(len(binaries) == 14 and len(metadata) == 51 and sum(r['bytes'] for r in binaries.values()) == TOTAL_BYTES,
            'frozen prefetch inventory geometry')
    return {'binaries': dict(sorted(binaries.items())), 'metadata': dict(sorted(metadata.items())), 'consumed': consumed}


REMOTE_SCRIPT = r'''
from pathlib import Path
import base64, datetime, hashlib, json, os, stat, sys, time
config = CONFIG
workspace = Path('/workspace/jlens')
root = workspace / 'results'
def require(ok, message):
    if not ok: raise ValueError(message)
def regular(path):
    for parent in (path, *path.parents): require(not parent.is_symlink(), 'linked remote path')
    require(stat.S_ISREG(path.stat().st_mode), 'nonregular remote file')
def status():
    regular(workspace / 'status.json')
    value = json.loads((workspace / 'status.json').read_text())
    require(value.get('lease_name') == config['lease_name'], 'remote lease changed')
    require((workspace / 'provider_lease_name').read_text().strip() == config['lease_name'], 'provider lease changed')
    if os.environ.get('RUNPOD_POD_ID') is not None:
        require(os.environ['RUNPOD_POD_ID'] == config['pod_id'], 'remote pod changed')
    require(value.get('phase') == 'huginn_fit' and value.get('setup_complete') is True
            and value.get('stop_requested', False) is False, 'Huginn fitting is not active')
    require(type(value.get('child_pid')) is int and value['child_pid'] > 0
            and Path('/proc', str(value['child_pid'])).exists(), 'Huginn stage child is not active')
    require(not (workspace / 'STOP').exists() and not (workspace / 'STOP').is_symlink(), 'STOP is present')
    return value
def small(path):
    regular(path)
    body = path.read_bytes()
    return body, {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
before = status()
gate_paths = ['preflight_huginn/COMPLETE.json', 'huginn_budget_projection.json']
gates = {}
for name in gate_paths:
    body, record = small(root / name)
    value = json.loads(body)
    gates[name] = {'record': record, 'data': base64.b64encode(body).decode()}
    if name.startswith('preflight'):
        require(value.get('schema_version') == 1 and value.get('status') == 'passed'
                and value.get('kind') == 'huginn' and len(value.get('profiles', [])) == 1, 'Huginn preflight not passed')
        native = value.get('native_primal', {})
        require(all(native.get(k) is True for k in ('all_34_cells_bitwise_equal', 'logits_bitwise_equal', 'coupled_initial_states'))
                and native.get('batch') == 8, 'native Huginn gate changed')
        profile = value['profiles'][0]
        require(profile.get('name') == 'huginn_r8' and profile.get('target_layer') == 33
                and profile.get('source_layers') == list(range(32)), 'Huginn profile changed')
        require(len(profile.get('parity', [])) == 1 and profile['parity'][0].get('arm') == 'dense'
                and profile['parity'][0].get('source_count') == 32
                and profile['parity'][0].get('output_directions') == 5280
                and profile['parity'][0].get('bitwise_equal') is True, 'Huginn parity gate changed')
    else:
        require(value.get('status') == 'passed' and value.get('fits_before_work_deadline') is True
                and value.get('within_combined_budget') is True
                and value.get('basis') == 'measured complete Huginn B8 paragraph and checkpoint I/O',
                'measured Huginn budget gate not passed')
gate_records = {name: row['record'] for name, row in gates.items()}
if config.get('gate_records') is not None:
    require(gate_records == config['gate_records'], 'Huginn gate bytes changed')
result = {'status_before': before, 'gates': gates}
if config.get('binary') is not None:
    for name, expected in config['metadata'].items():
        _, actual = small(root / name)
        require(actual == expected, 'accepted remote metadata changed: ' + name)
    name = config['binary']
    expected = config['binaries'][name]
    path = root / name
    regular(path)
    prior = path.stat()
    require(prior.st_size == expected['bytes'], 'remote binary size changed')
    digest, count, checked = hashlib.sha256(), 0, time.monotonic()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block); count += len(block)
            if time.monotonic() - checked >= 5:
                status(); checked = time.monotonic()
    after = path.stat()
    require((prior.st_dev, prior.st_ino, prior.st_size, prior.st_mtime_ns, prior.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'remote binary changed during hashing')
    actual = {'bytes': count, 'sha256': digest.hexdigest()}
    require(actual == expected, 'remote binary hash differs from accepted seal')
    result['binary'] = {'path': name, 'record': actual, 'mtime_ns': after.st_mtime_ns}
result['status_after'] = status()
print(json.dumps(result, sort_keys=True), flush=True)
'''


@contextmanager
def sync_lock(root):
    path = io._no_links(root / '.sync.lock')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StopPrefetch('Final retrieval holds .sync.lock') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install_verified(staged, target, expected, mtime_ns, root, recheck, guard=lambda: None):
    """Hash outside the lock; publish with atomic no-overwrite hard link."""
    staged, target = io._no_links(staged), io._no_links(target)
    record, staged_signature = checked_record(staged, guard)
    require(record == expected and staged.stat().st_mtime_ns == mtime_ns, 'staged bytes/mtime are not verified')
    existing_signature = None
    if target.exists():
        existing, existing_signature = checked_record(target, guard)
        require(existing == expected and target.stat().st_mtime_ns == mtime_ns,
                'existing destination bytes/mtime differ; preserving the existing file')
    with sync_lock(root):
        recheck()
        io._no_links(staged)
        io._no_links(target)
        require(signature(staged.stat()) == staged_signature, 'staged file changed before publication')
        if target.exists():
            require(existing_signature is not None and signature(target.stat()) == existing_signature,
                    'destination appeared or changed; refusing overwrite')
            return 'already_present'
        require(existing_signature is None, 'existing destination disappeared')
        target.parent.mkdir(parents=True, exist_ok=True)
        io._no_links(target.parent)
        require(staged.stat().st_dev == target.parent.stat().st_dev, 'atomic install requires the same filesystem')
        os.link(staged, target, follow_symlinks=False)
        # Only our verified staging name is removed; destination bytes are never overwritten.
        staged.unlink()
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return 'installed'


def stop_process(process):
    """Clean only our start_new_session group, including an exited leader's children."""
    def active_members():
        members = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                members.append(int(path.name))
        return members

    def group_signal(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    if active_members():
        group_signal(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while active_members() and time.monotonic() < deadline:
            process.poll()
            time.sleep(.05)
        if active_members():
            group_signal(signal.SIGKILL)
            deadline = time.monotonic() + 2
            while active_members() and time.monotonic() < deadline:
                process.poll()
                time.sleep(.05)
    process.wait(timeout=5)
    if active_members():
        raise StopPrefetch('Started subprocess group did not become quiescent')


def run_polled(command, stdout_path, stderr_path, guard, timeout):
    started = time.monotonic()
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                guard()
                if time.monotonic() - started >= timeout:
                    raise StopPrefetch('Bounded subprocess deadline reached')
                time.sleep(min(POLL_SECONDS, max(0.01, timeout - (time.monotonic() - started))))
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            guard()
        finally:
            stop_process(process)


class Prefetch:
    def __init__(self, expected, operation):
        self.expected, self.operation = expected, operation
        self.gate_records = None
        self.counter = 0
        self.state = None

    def local_active(self, check_lock=True):
        state = io._json(LEASE_ROOT / 'LEASE.json')
        if state.get('name') != NAME or state.get('pod_id') != POD or state.get('status') != 'running' or state.get('halt_requested'):
            raise StopPrefetch('Local lease is no longer the active authorized attempt')
        if (LEASE_ROOT / 'RETRIEVAL_VERIFIED.json').exists():
            raise StopPrefetch('Final retrieval is already verified')
        if time.time() >= lease.epoch(state['work_deadline_utc']):
            raise StopPrefetch('Work deadline reached')
        if check_lock:
            with sync_lock(LEASE_ROOT):
                pass
        self.state = state

    def remote_command(self, binary=None):
        config = {'lease_name': NAME, 'pod_id': POD, 'gate_records': self.gate_records, 'binary': binary,
                  'metadata': self.expected['metadata'] if binary else {}, 'binaries': self.expected['binaries'] if binary else {}}
        source = REMOTE_SCRIPT.replace('CONFIG', repr(config), 1)
        return [*lease.ssh_options(LEASE_ROOT, self.state), 'root@' + self.state['ssh']['host'],
                'python -c ' + shlex.quote(source)]

    def status(self, check_lock=True, timeout=STATUS_TIMEOUT):
        self.local_active(check_lock)
        result = subprocess.run(self.remote_command(), capture_output=True, text=True, timeout=timeout, check=True)
        value = json.loads(result.stdout)
        records = {}
        for name, row in value['gates'].items():
            body = base64.b64decode(row['data'], validate=True)
            require({'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} == row['record'], 'transported gate bytes differ')
            records[name] = row['record']
        if self.gate_records is None:
            self.gate_records = records
            io._new_json(self.operation / 'INITIAL_STATUS.json', value)
            for name, row in value['gates'].items():
                target = self.operation / ('HUGINN_PREFLIGHT.json' if name.startswith('preflight') else 'HUGINN_BUDGET.json')
                with target.open('xb') as handle:
                    handle.write(base64.b64decode(row['data'], validate=True))
        require(records == self.gate_records, 'remote gate records changed')
        return value

    def snapshot(self, relative, label):
        self.status()
        stem = self.operation / f'{self.counter:02d}_{label}'
        stdout, stderr = stem.with_suffix('.json'), stem.with_suffix('.stderr')
        run_polled(self.remote_command(relative), stdout, stderr, self.status, HASH_TIMEOUT)
        value = json.loads(stdout.read_text())
        require(value['binary']['path'] == relative and value['binary']['record'] == self.expected['binaries'][relative],
                'remote snapshot differs from frozen binary')
        return value

    def run(self):
        self.status()
        installed = []
        for relative, expected in self.expected['binaries'].items():
            self.counter += 1
            self.status()
            print(json.dumps({'event': 'binary_started', 'index': self.counter, 'files': 14,
                              'path': relative, 'bytes': expected['bytes']}), flush=True)
            before = self.snapshot(relative, 'before')
            target = io._no_links(LEASE_ROOT / 'retrieved' / relative)
            staged = self.operation / 'staging' / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                actual, existing_signature = checked_record(target, self.status)
                require(actual == expected, 'existing target has different bytes; refusing overwrite')
                after = self.snapshot(relative, 'after')
                require(before['binary'] == after['binary'], 'remote binary or mtime changed during verification')
                # Recheck under the same lock; hashing was completed outside it.
                with sync_lock(LEASE_ROOT):
                    self.status(check_lock=False, timeout=5)
                    io._no_links(target)
                    require(signature(target.stat()) == existing_signature, 'existing target changed before acceptance')
                    require(target.stat().st_mtime_ns == before['binary']['mtime_ns'],
                            'existing target mtime differs; preserving it for the final collector')
                outcome = 'already_present'
            else:
                transport = shlex.join(lease.ssh_options(LEASE_ROOT, self.state))
                command = ['rsync', '-rlt', '--partial', '--timeout=60', '-e', transport,
                           'root@' + self.state['ssh']['host'] + ':' + REMOTE + '/' + relative, str(staged)]
                run_polled(command, self.operation / f'{self.counter:02d}_rsync.log',
                           self.operation / f'{self.counter:02d}_rsync.stderr', self.status, TRANSFER_TIMEOUT)
                after = self.snapshot(relative, 'after')
                require(before['binary'] == after['binary'], 'remote binary or mtime changed during transfer')
                os.utime(staged, ns=(before['binary']['mtime_ns'], before['binary']['mtime_ns']))
                with staged.open('rb') as handle:
                    os.fsync(handle.fileno())
                outcome = install_verified(staged, target, expected, before['binary']['mtime_ns'], LEASE_ROOT,
                    lambda: self.status(check_lock=False, timeout=5), self.status)
            result = {'path': relative, 'record': expected, 'remote_mtime_ns': before['binary']['mtime_ns'],
                      'local_mtime_ns': target.stat().st_mtime_ns,
                      'outcome': outcome, 'finished_utc': utc()}
            installed.append(result)
            io._new_json(self.operation / f'{self.counter:02d}_VERIFIED.json', result)
            print(json.dumps({'event': 'binary_verified', **result}), flush=True)
        self.status()
        for name, record in self.expected['consumed'].items():
            io._verify_record(Path(name), record)
        return installed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('describe', 'run'), nargs='?', default='describe')
    args = parser.parse_args(argv)
    expected = expected_inventory()
    if args.command == 'describe':
        print(json.dumps({'lease_name': NAME, 'pod_id': POD, 'required_phase': 'huginn_fit',
            'destination': str(LEASE_ROOT / 'retrieved'), 'files': expected['binaries'],
            'metadata_records': len(expected['metadata']), 'bytes': TOTAL_BYTES,
            'execution_performed': False}, indent=2))
        return 0
    operation = io._no_links(LEASE_ROOT / ('prefetch_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    operation.mkdir(exist_ok=False)
    io._new_json(operation / 'PLAN.json', {'lease_name': NAME, 'pod_id': POD, 'required_phase': 'huginn_fit',
        'source': io._record(Path(__file__)), 'inventory': expected, 'created_utc': utc(),
        'scope': 'Read-only remote prefetch; all operation/staging files outside retrieved; final collector unchanged.'})
    print(json.dumps({'event': 'prefetch_started', 'operation': str(operation)}), flush=True)
    runner = Prefetch(expected, operation)
    try:
        rows = runner.run()
    except BaseException as error:
        io._new_json(operation / 'INCOMPLETE.json', {'status': 'incomplete', 'stopped_utc': utc(),
            'error_type': type(error).__name__, 'reason': str(error),
            'scope': 'Only already verified published files may exist under retrieved; staged partials remain outside it.'})
        raise
    io._new_json(operation / 'PREFETCH_VERIFIED.json', {'status': 'passed', 'lease_name': NAME, 'pod_id': POD,
        'finished_utc': utc(), 'files': rows, 'bytes': TOTAL_BYTES,
        'scope': 'Prefetch only. Every selected binary matched accepted seals remotely before/after and locally; '
                 'this is not final quiescent retrieval and is not a RETRIEVAL_VERIFIED receipt.'})
    print(json.dumps({'event': 'prefetch_complete', 'operation': str(operation), 'files': len(rows), 'bytes': TOTAL_BYTES}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
