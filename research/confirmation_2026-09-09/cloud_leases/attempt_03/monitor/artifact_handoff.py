"""Receiver-owned, manifest-bound preservation for the confirmation experiment.

No import performs transport or tensor loading. The semantic verifier is pinned
by the local controller intent and runs only after complete payload hashing.
"""
from __future__ import annotations

from contextlib import contextmanager
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import sys
import time
import types
import uuid

import run_refits as io
from transport import run_bounded

sys.dont_write_bytecode = True
SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
MANIFEST_KEYS = {'schema', 'binding', 'kind', 'stage_id', 'outcome', 'files'}
SOURCE_NAMES = {'lease.py', 'artifact_handoff.py', 'transport.py',
                'run_refits.py', 'runpod_api.py', 'pod_entry.sh'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(io._canonical(value)).hexdigest()


manifest_digest = digest


def safe_path(value):
    require(isinstance(value, str) and value and '\\' not in value and '\x00' not in value,
            'invalid relative artifact path')
    parts = value.split('/')
    require(not value.startswith('/') and all(part not in ('', '.', '..') for part in parts)
            and all(not any(ord(c) < 32 or ord(c) == 127 for c in part) for part in parts),
            'unsafe relative artifact path')
    require(PurePosixPath(value).as_posix() == value, 'artifact path is not canonical')
    return value


def valid_record(record):
    require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'}
            and type(record['bytes']) is int and record['bytes'] >= 0
            and isinstance(record['sha256'], str) and SHA.fullmatch(record['sha256']),
            'invalid artifact record')
    return record


def signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def checked_record(path):
    path = io._no_links(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'nonregular artifact: ' + str(path))
    result, count = hashlib.sha256(), 0
    with path.open('rb') as handle:
        require(signature(os.fstat(handle.fileno())) == signature(before), 'artifact changed before open')
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
            count += len(block)
        require(signature(os.fstat(handle.fileno())) == signature(before), 'artifact changed during hash')
    require(signature(path.stat()) == signature(before) and count == before.st_size,
            'artifact pathname changed during hash')
    return {'bytes': count, 'sha256': result.hexdigest()}


def verify_record(path, record):
    require(checked_record(path) == valid_record(record), 'artifact hash mismatch: ' + str(path))


def binding(state):
    require(isinstance(state.get('name'), str) and SAFE_ID.fullmatch(state['name'])
            and isinstance(state.get('pod_id'), str) and SAFE_ID.fullmatch(state['pod_id'])
            and isinstance(state.get('run_id'), str) and SAFE_ID.fullmatch(state['run_id'])
            and isinstance(state.get('account_id'), str) and state['account_id'], 'incomplete run identity')
    for key in ('run_spec_sha256', 'output_contract_sha256'):
        require(isinstance(state.get(key), str) and SHA.fullmatch(state[key]), 'invalid run contract hash')
    sources = state.get('controller_sources')
    require(isinstance(sources, dict) and set(sources) == SOURCE_NAMES, 'controller source set differs')
    for record in sources.values():
        valid_record(record)
    return {'account_id': state['account_id'], 'lease_name': state['name'], 'pod_id': state['pod_id'],
            'run_id': state['run_id'], 'run_spec_sha256': state['run_spec_sha256'],
            'output_contract_sha256': state['output_contract_sha256'], 'controller_sources': sources}


def verify_local_sources(state):
    here = Path(__file__).parent
    for name, record in state['controller_sources'].items():
        verify_record(here / name, record)
    verify_record(state['run_spec_path'], state['run_spec_record'])
    require(state['run_spec_record']['sha256'] == state['run_spec_sha256'], 'run-spec pin differs')
    verify_record(state['semantic_verifier']['path'], state['semantic_verifier']['record'])


def load_contract(state):
    verify_record(state['output_contract_path'], state['output_contract_record'])
    require(state['output_contract_record']['sha256'] == state['output_contract_sha256'],
            'output-contract pin differs')
    contract = io._json(state['output_contract_path'])
    require(isinstance(contract, dict) and set(contract) == {
        'schema', 'run_id', 'run_spec_sha256', 'files', 'stages', 'required_checks'}, 'invalid output contract schema')
    require(contract['schema'] == 'confirmation_output_contract.v1'
            and contract['run_id'] == state['run_id']
            and contract['run_spec_sha256'] == state['run_spec_sha256'], 'output contract identity differs')
    require(isinstance(contract['files'], dict) and contract['files'], 'empty output contract')
    for name, role in contract['files'].items():
        safe_path(name)
        require(isinstance(role, dict) and set(role) == {'role'} and isinstance(role['role'], str)
                and role['role'], 'invalid artifact role')
    require(isinstance(contract['stages'], dict), 'invalid stage list')
    for stage, paths in contract['stages'].items():
        require(isinstance(stage, str) and SAFE_ID.fullmatch(stage) and isinstance(paths, list)
                and paths and all(isinstance(p, str) for p in paths)
                and paths == sorted(set(paths)) and set(paths) <= set(contract['files']), 'invalid stage membership')
    checks = contract['required_checks']
    require(isinstance(checks, list) and all(isinstance(c, str) and SAFE_ID.fullmatch(c) for c in checks)
            and len(set(checks)) == len(checks) and {'loadability', 'numerical'} <= set(checks),
            'required semantic checks missing')
    return contract


def validate_manifest(state, manifest, contract=None):
    contract = load_contract(state) if contract is None else contract
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS
            and manifest['schema'] == 'confirmation_artifact_manifest.v1'
            and manifest['binding'] == binding(state), 'manifest identity or schema differs')
    require(manifest['kind'] in ('stage', 'final') and manifest['outcome'] in ('complete', 'stopped', 'failed'),
            'invalid manifest outcome')
    if manifest['kind'] == 'stage':
        require(manifest['stage_id'] in contract['stages'] and manifest['outcome'] == 'complete', 'invalid stage')
        expected = set(contract['stages'][manifest['stage_id']])
    else:
        require(manifest['stage_id'] is None, 'final manifest has a stage ID')
        expected = set(contract['files'])
    require(isinstance(manifest['files'], dict), 'invalid manifest files')
    actual = set(manifest['files'])
    require(actual == expected if manifest['outcome'] == 'complete' else actual <= expected,
            'manifest omits or adds contracted output')
    for name, record in manifest['files'].items():
        safe_path(name)
        valid_record(record)
    return manifest


@contextmanager
def handoff_lock(root, *, nonblocking=False):
    root = io._mkdir(root)
    descriptor = os.open(io._no_links(root / '.handoff.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+b') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def payload_inventory(root):
    root = io._no_links(root)
    require(root.is_dir(), 'payload directory missing')
    files, directories = {}, set()
    def walk_error(error):
        raise error
    for directory, children, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in children:
            path = io._no_links(Path(directory) / name)
            require(path.is_dir(), 'invalid payload directory')
            directories.add(path.relative_to(root).as_posix())
        for name in names:
            path = io._no_links(Path(directory) / name)
            files[path.relative_to(root).as_posix()] = checked_record(path)
    expected_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != '.'}
    require(directories == expected_dirs, 'unexpected empty payload directory')
    return files


def verify_payload(root, files):
    require(payload_inventory(root) == files, 'local payload membership or bytes differ from manifest')


def _validation_result(result, state, manifest, contract):
    keys = {'schema', 'status', 'run_id', 'manifest_sha256', 'contract_sha256', 'checked_files', 'checks'}
    require(isinstance(result, dict) and set(result) in (keys, keys | {'details'})
            and result['schema'] == 'confirmation_validation.v1' and result['status'] == 'passed'
            and result['run_id'] == state['run_id'] and result['manifest_sha256'] == digest(manifest)
            and result['contract_sha256'] == state['output_contract_sha256']
            and result['checked_files'] == sorted(manifest['files'])
            and isinstance(result['checks'], dict)
            and all(result['checks'].get(check) is True for check in contract['required_checks'])
            and ('details' not in result or isinstance(result['details'], dict)), 'semantic verification did not pass exact scope')
    return result


def semantic_validate(root, state, manifest, contract):
    verifier = state['semantic_verifier']
    require(isinstance(verifier, dict) and set(verifier) == {'path', 'record', 'callable'}
            and verifier['callable'] == 'validate_outputs', 'invalid semantic verifier binding')
    path = io._no_links(verifier['path'])
    require(path.is_absolute() and path.suffix == '.py', 'invalid local verifier path')
    verify_record(path, verifier['record'])
    name = '_confirmation_validator_' + verifier['record']['sha256']
    source = path.read_bytes()
    require({'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()} == verifier['record'],
            'verifier changed before compilation')
    # Compile the pinned source bytes directly: an existing .pyc is not authority.
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
        function = getattr(module, verifier['callable'])
        require(callable(function), 'local semantic verifier is not callable')
        result = function(root=Path(root), manifest=manifest, contract=contract)
        verify_record(path, verifier['record'])
        return _validation_result(result, state, manifest, contract)
    finally:
        sys.modules.pop(name, None)


def generation_path(root, state, manifest, *, accepted):
    require(SAFE_ID.fullmatch(state['run_id']), 'unsafe run ID')
    return io._no_links(Path(root) / 'handoff' / ('accepted' if accepted else 'staging') /
                        state['run_id'] / digest(manifest))


def _receipt(generation, state, manifest, contract, *, rehash=True):
    receipt_path = io._no_links(generation / 'RECEIPT.json')
    receipt = io._json(receipt_path)
    expected = {'schema', 'status', 'binding', 'kind', 'manifest_sha256', 'manifest_record',
                'files', 'verifier', 'validation_record', 'accepted_utc'}
    require(isinstance(receipt, dict) and set(receipt) == expected
            and receipt['schema'] == 'confirmation_retrieval_acceptance.v1' and receipt['status'] == 'passed'
            and receipt['binding'] == binding(state) and receipt['kind'] == manifest['kind']
            and receipt['manifest_sha256'] == digest(manifest) and receipt['files'] == manifest['files']
            and receipt['verifier'] == state['semantic_verifier'] and manifest['outcome'] == 'complete',
            'acceptance receipt binding differs')
    verify_record(generation / 'MANIFEST.json', receipt['manifest_record'])
    require(io._json(generation / 'MANIFEST.json') == manifest, 'accepted manifest differs')
    verify_record(generation / 'VALIDATION.json', receipt['validation_record'])
    _validation_result(io._json(generation / 'VALIDATION.json'), state, manifest, contract)
    verify_record(state['semantic_verifier']['path'], state['semantic_verifier']['record'])
    if rehash:
        verify_payload(generation / 'results', manifest['files'])
    return {'path': str(receipt_path), 'record': checked_record(receipt_path), 'manifest_sha256': digest(manifest)}


def acceptance_pointer(root, state, manifest, *, rehash=True):
    contract = load_contract(state)
    validate_manifest(state, manifest, contract)
    verify_local_sources(state)
    return _receipt(generation_path(root, state, manifest, accepted=True), state, manifest, contract, rehash=rehash)


def accepted_for_current_run(root, state, *, rehash=True):
    try:
        pointer = state.get('retrieval_accepted')
        terminal = state.get('computation_finished')
        require(isinstance(pointer, dict) and set(pointer) == {'path', 'record', 'manifest_sha256'}
                and isinstance(terminal, dict) and terminal.get('binding') == binding(state)
                and terminal.get('outcome') == 'complete'
                and pointer['manifest_sha256'] == terminal['manifest_sha256'], 'no current final acceptance')
        manifest = terminal['manifest']
        require(manifest['kind'] == 'final' and digest(manifest) == terminal['manifest_sha256'], 'invalid final binding')
        actual = acceptance_pointer(root, state, manifest, rehash=rehash)
        require(actual == pointer, 'stale or modified acceptance pointer')
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def read_index(state, transport):
    raw = transport.read_bytes('artifact_index.json')
    if raw is None:
        return []
    index = json.loads(raw)
    require(isinstance(index, dict) and set(index) == {'schema', 'binding', 'manifests'}
            and index['schema'] == 'confirmation_artifact_index.v1'
            and index['binding'] == binding(state) and isinstance(index['manifests'], list), 'invalid artifact index')
    require(len(index['manifests']) <= 256, 'too many artifact manifests')
    manifests, seen, final = [], set(), False
    for ref in index['manifests']:
        require(isinstance(ref, dict) and set(ref) == {'path', 'record'}, 'invalid manifest reference')
        name = safe_path(ref['path'])
        require(name.startswith('manifests/') and len(PurePosixPath(name).parts) == 2 and name.endswith('.json')
                and name not in seen, 'invalid or duplicate manifest path')
        seen.add(name)
        valid_record(ref['record'])
        raw = transport.read_bytes(name)
        require(raw is not None and {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == ref['record'],
                'manifest reference bytes differ')
        manifest = json.loads(raw)
        validate_manifest(state, manifest)
        require(not (manifest['kind'] == 'final' and final), 'multiple final manifests')
        final |= manifest['kind'] == 'final'
        transport.manifest_refs[digest(manifest)] = ref
        manifests.append(manifest)
    return manifests


def _reuse_stages(root, state, manifest, destination):
    parent = generation_path(root, state, manifest, accepted=True).parent
    if not parent.exists():
        return
    for generation in sorted(parent.iterdir()):
        if not generation.is_dir() or generation.name == digest(manifest):
            continue
        try:
            previous = io._json(io._no_links(generation / 'MANIFEST.json'))
            validate_manifest(state, previous)
            require(previous['kind'] == 'stage', 'not a stage generation')
            _receipt(generation, state, previous, load_contract(state), rehash=True)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for name, expected in previous['files'].items():
            target = io._no_links(destination / name)
            if manifest['files'].get(name) != expected or target.exists():
                continue
            io._mkdir(target.parent)
            # Published generations are immutable. No later transfer uses in-place writes.
            os.link(io._no_links(generation / 'results' / name), target, follow_symlinks=False)
            io._fsync_dir(target.parent)


def collect_manifest(root, state, manifest, transport, *, progress=None, fault_hook=None):
    root = io._no_links(root)
    progress = progress or (lambda **fields: None)
    fault_hook = fault_hook or (lambda stage: None)
    with handoff_lock(root):
        contract = load_contract(state)
        validate_manifest(state, manifest, contract)
        verify_local_sources(state)
        expected_digest = digest(manifest)
        staging = generation_path(root, state, manifest, accepted=False)
        accepted = generation_path(root, state, manifest, accepted=True)
        progress(manifest_sha256=expected_digest, staging=str(staging), phase='verifying_source')
        if accepted.exists():
            try:
                require(io._json(io._no_links(accepted / 'MANIFEST.json')) == manifest,
                        'published generation identity changed')
                if (accepted / 'RECEIPT.json').exists():
                    return _receipt(accepted, state, manifest, contract)
                # A crash after publication requires fresh complete revalidation.
                verify_payload(accepted / 'results', manifest['files'])
                _validation_result(io._json(io._no_links(accepted / 'VALIDATION.json')), state, manifest, contract)
                staging = accepted
            except (OSError, ValueError, KeyError, TypeError) as error:
                # Preserve the damaged generation verbatim, including missing or
                # malformed metadata. Repair never writes into published payloads.
                quarantine = io._mkdir(root / 'handoff' / 'quarantine' / state['run_id'])
                moved = quarantine / (expected_digest + '_' + uuid.uuid4().hex)
                try:
                    old_receipt = checked_record(accepted / 'RECEIPT.json')
                except (OSError, ValueError):
                    old_receipt = None
                io._new_json(quarantine / (moved.name + '.json'), {
                    'schema': 'confirmation_generation_quarantine.v1', 'binding': binding(state),
                    'manifest_sha256': expected_digest, 'source': str(accepted), 'preserved': str(moved),
                    'old_receipt_record': old_receipt, 'reason': str(error), 'utc': time.time()})
                os.rename(accepted, moved)
                io._fsync_dir(quarantine)
                io._fsync_dir(accepted.parent)
        if staging != accepted:
            io._mkdir(staging)
            pending = {'schema': 'confirmation_handoff_pending.v1', 'binding': binding(state),
                       'manifest_sha256': expected_digest}
            if (staging / 'MANIFEST.json').exists():
                require(io._json(staging / 'MANIFEST.json') == manifest, 'staged manifest changed')
            else:
                io._new_json(staging / 'MANIFEST.json', manifest)
            if (staging / 'PENDING.json').exists():
                require(io._json(staging / 'PENDING.json') == pending, 'staging identity changed')
            else:
                io._new_json(staging / 'PENDING.json', pending)
            destination = io._mkdir(staging / 'results')
            transport.snapshot(manifest)
            _reuse_stages(root, state, manifest, destination)
            progress(manifest_sha256=expected_digest, staging=str(staging), phase='transferring')
            transport.transfer(sorted(manifest['files']), destination, manifest['files'])
            fault_hook('after_transfer')
            transport.snapshot(manifest)
        verify_payload(staging / 'results', manifest['files'])
        if manifest['outcome'] != 'complete':
            progress(manifest_sha256=expected_digest, staging=str(staging), phase='preserved_incomplete')
            return None
        progress(manifest_sha256=expected_digest, staging=str(staging), phase='validating')
        result = semantic_validate(staging / 'results', state, manifest, contract)
        verify_payload(staging / 'results', manifest['files'])
        verify_local_sources(state)
        load_contract(state)
        if (staging / 'VALIDATION.json').exists():
            _validation_result(io._json(staging / 'VALIDATION.json'), state, manifest, contract)
        else:
            io._new_json(staging / 'VALIDATION.json', result)
        fault_hook('after_validation')
        if staging != accepted:
            io._mkdir(accepted.parent)
            require(not accepted.exists(), 'accepted generation must not be overwritten')
            # Remove only our protocol marker/rsync temporaries, never payloads.
            (staging / 'PENDING.json').unlink()
            fault_hook('after_pending_unlink')
            partial = staging / 'partials'
            if partial.exists():
                io._no_links(partial)
                shutil.rmtree(partial)
            os.rename(staging, accepted)
            io._fsync_dir(accepted.parent)
        fault_hook('after_publication')
        progress(manifest_sha256=expected_digest, staging=str(accepted), phase='publishing_acceptance')
        receipt = {'schema': 'confirmation_retrieval_acceptance.v1', 'status': 'passed', 'binding': binding(state),
                   'kind': manifest['kind'], 'manifest_sha256': expected_digest,
                   'manifest_record': checked_record(accepted / 'MANIFEST.json'), 'files': manifest['files'],
                   'verifier': state['semantic_verifier'], 'validation_record': checked_record(accepted / 'VALIDATION.json'),
                   'accepted_utc': time.time()}
        pending = accepted / ('RECEIPT_PENDING_' + uuid.uuid4().hex + '.json')
        io._new_json(pending, receipt)
        fault_hook('before_receipt_link')
        os.link(pending, accepted / 'RECEIPT.json', follow_symlinks=False)
        try:
            io._fsync_dir(accepted)
        except BaseException:
            if (accepted / 'RECEIPT.json').exists() and os.path.samefile(pending, accepted / 'RECEIPT.json'):
                (accepted / 'RECEIPT.json').unlink()
            raise
        fault_hook('after_receipt_link')
        return _receipt(accepted, state, manifest, contract)


class LocalTransport:
    """Real local-byte transport for offline fixtures; never contacts a worker."""
    def __init__(self, workspace, *, interrupt_after_bytes=None):
        self.workspace = io._no_links(workspace)
        self.manifest_refs = {}
        self.interrupt_after_bytes = interrupt_after_bytes
        self.transferred_bytes = 0
        self.resumed_files = []

    def read_bytes(self, relative):
        path = io._no_links(self.workspace / safe_path(relative))
        if not path.exists():
            return None
        before = checked_record(path)
        raw = path.read_bytes()
        require({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == before, 'metadata changed during read')
        return raw

    def snapshot(self, manifest):
        ref = self.manifest_refs[digest(manifest)]
        raw = self.read_bytes(ref['path'])
        require(raw is not None and {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == ref['record']
                and json.loads(raw) == manifest, 'source manifest changed')
        for name, record in manifest['files'].items():
            verify_record(self.workspace / 'results' / name, record)

    def transfer(self, files, destination, expected_records):
        partial_root = io._mkdir(destination.parent / 'partials')
        for name in files:
            target = io._no_links(destination / safe_path(name))
            if target.exists() and checked_record(target) == expected_records[name]:
                continue
            source = io._no_links(self.workspace / 'results' / name)
            temporary = io._no_links(partial_root / name)
            io._mkdir(temporary.parent)
            io._mkdir(target.parent)
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset:
                # Verify the partial prefix; a bad prefix must be replaced, not appended.
                with source.open('rb') as a, temporary.open('rb') as b:
                    if a.read(offset) != b.read():
                        offset = 0
                self.resumed_files.append(name)
            with source.open('rb') as incoming, temporary.open('ab' if offset else 'wb') as outgoing:
                incoming.seek(offset)
                while block := incoming.read(256):
                    outgoing.write(block)
                    self.transferred_bytes += len(block)
                    if self.interrupt_after_bytes is not None and self.transferred_bytes >= self.interrupt_after_bytes:
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                        raise TimeoutError('synthetic interrupted copy')
                outgoing.flush()
                os.fsync(outgoing.fileno())
            verify_record(temporary, expected_records[name])
            os.replace(temporary, target)
            io._fsync_dir(target.parent)


class SSHTransport:
    def __init__(self, root, state, ssh_options):
        self.root, self.state, self.options = Path(root), state, list(ssh_options)
        self.manifest_refs, self.processes = {}, []

    def _python(self, script, *, timeout, operation):
        command = [*self.options, 'root@' + self.state['ssh']['host'], 'python -c ' + shlex.quote(script)]
        return run_bounded(command, timeout=timeout, operation=operation, processes=self.processes).stdout

    def read_bytes(self, relative):
        relative = safe_path(relative)
        encoded = base64.b64encode(relative.encode()).decode()
        script = """import base64,os,pathlib,stat,sys
p=pathlib.Path('/workspace/jlens')/base64.b64decode('%s').decode()
def sig(s): return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
for parent in (p,*p.parents):
 if parent.is_symlink(): raise ValueError('linked manifest path')
if not p.exists(): sys.exit(0)
s=p.stat()
if not stat.S_ISREG(s.st_mode) or s.st_size>4000000: raise ValueError('invalid metadata file')
with p.open('rb') as f:
 data=f.read()
 if sig(os.fstat(f.fileno()))!=sig(s): raise ValueError('metadata changed')
if sig(p.stat())!=sig(s): raise ValueError('metadata path changed')
sys.stdout.buffer.write(data)
""" % encoded
        return self._python(script, timeout=30, operation='read_metadata') or None

    def snapshot(self, manifest):
        ref = self.manifest_refs[digest(manifest)]
        raw = self.read_bytes(ref['path'])
        require(raw is not None and {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == ref['record']
                and json.loads(raw) == manifest, 'remote source manifest changed')
        encoded = base64.b64encode(io._canonical(manifest['files'])).decode()
        script = """import base64,hashlib,json,os,pathlib,stat
files=json.loads(base64.b64decode('%s')); root=pathlib.Path('/workspace/jlens/results')
def sig(s): return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
for name,expected in files.items():
 p=root/name
 for parent in (p,*p.parents):
  if parent.is_symlink(): raise ValueError('linked payload')
 s=p.stat()
 if not stat.S_ISREG(s.st_mode): raise ValueError('nonregular payload')
 h=hashlib.sha256(); n=0
 with p.open('rb') as f:
  if sig(os.fstat(f.fileno()))!=sig(s): raise ValueError('payload changed before open')
  while block:=f.read(8*1024*1024): h.update(block); n+=len(block)
  if sig(os.fstat(f.fileno()))!=sig(s): raise ValueError('payload changed while reading')
 if sig(p.stat())!=sig(s) or {'bytes':n,'sha256':h.hexdigest()}!=expected: raise ValueError('payload mismatch')
print('passed')
""" % encoded
        self._python(script, timeout=self.state['transfer_timeout_seconds'], operation='source_hashes')

    def transfer(self, files, destination, expected_records):
        listing = destination.parent / 'FILES_FROM'
        content = b'\0'.join(safe_path(name).encode() for name in files) + b'\0'
        if listing.exists():
            require(listing.read_bytes() == content, 'transfer path list changed')
        else:
            with listing.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        partial = io._mkdir(destination.parent / 'partials')
        command = ['rsync', '-rlt', '--checksum', '--partial', '--partial-dir=' + str(partial),
                   '--timeout=60', '--from0', '--files-from=' + str(listing),
                   '-e', shlex.join(self.options), 'root@' + self.state['ssh']['host'] + ':/workspace/jlens/results/',
                   str(destination) + '/']
        run_bounded(command, timeout=self.state['transfer_timeout_seconds'], operation='transfer', processes=self.processes)
