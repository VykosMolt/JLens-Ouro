#!/usr/bin/env python3
"""Pre-stage four frozen banks with eight bounded SSH streams, then original launch."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
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
import tempfile
import threading
import time
import types
import uuid

sys.dont_write_bytecode = True
ROUND = Path(__file__).resolve().parents[1]
CONTROLLER = ROUND / 'controller'
CHUNK_BYTES = 64 * 1024 * 1024
STREAMS = 8
SETUP_RESERVE = 300
TEMP_LIMIT = 8 * 1024**3
WORK = '/workspace/jlens'
CONTROLLER_PINS = {'lease.py': {'bytes': 56750, 'sha256': 'c59d303693253017e56f44daebd261fbf3fe38b765602c9411902eaaa9248366'}, 'artifact_handoff.py': {'bytes': 29606, 'sha256': '879a9b6bd9c282dc5bd46f2fb86e20a7d63511940e86f0c7e5c3911d97d58154'}, 'transport.py': {'bytes': 2638, 'sha256': 'acec7a803383119706ee852d8c0b21c4380df0b91b16aa4c198462c6d7bd199a'}, 'run_refits.py': {'bytes': 51999, 'sha256': 'a8841cd456434b5582e578f115cbeff62829af89e1809a43a189a5dba0ff0fd4'}, 'runpod_api.py': {'bytes': 9601, 'sha256': '6124d239c0e1ec6b7c8e86e3f845716b17bd67943423f88416a9cae771051849'}, 'pod_entry.sh': {'bytes': 1223, 'sha256': '49770f538686d3798547742b57838fca2ba9246ccda054b8691da6ff28bf5cad'}}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Linked path: ' + str(part))
    return path


def signature(path):
    info = no_links(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Nonregular source: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def record(path):
    path = no_links(path)
    before = signature(path)
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before:
            raise ValueError('Source changed before hash')
        while block := handle.read(8 * 1024 * 1024):
            count += len(block)
            digest.update(block)
    if signature(path) != before or count != before[2]:
        raise ValueError('Source changed during hash')
    return {'bytes': count, 'sha256': digest.hexdigest()}


def verify(path, expected):
    if record(path) != expected:
        raise ValueError('Hash mismatch: ' + str(path))


def json_new(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(canonical(value) + b'\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        with open_directory(path.parent) as directory:
            os.fsync(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def open_directory(path):
    descriptor = os.open(no_links(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def load_controller():
    sources = {}
    for name, pin in CONTROLLER_PINS.items():
        path = CONTROLLER / name
        verify(path, pin)
        raw = path.read_bytes()
        if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != pin:
            raise ValueError('Controller changed before compilation')
        sources[name] = raw
    for name in ('run_refits', 'runpod_api', 'transport', 'artifact_handoff', 'lease'):
        module = types.ModuleType(name); module.__file__ = str(CONTROLLER / (name + '.py'))
        sys.modules[name] = module
        exec(compile(sources[name + '.py'], module.__file__, 'exec'), module.__dict__)
    return sys.modules['lease'], sys.modules['artifact_handoff'], sys.modules['transport']


def split_bank(source, expected, cache, *, chunk_bytes=CHUNK_BYTES, check=lambda: None):
    """One source pass; published chunks are immutable and resume by SHA-256."""
    source = no_links(source); before = signature(source)
    if before[2] != expected['bytes'] or not 0 < chunk_bytes <= CHUNK_BYTES:
        raise ValueError('Bank size or chunk size differs')
    destination = no_links(cache / expected['sha256'])
    destination.mkdir(parents=True, exist_ok=True)
    chunks, digest, offset = [], hashlib.sha256(), 0
    with no_links(destination / '.materialize.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with source.open('rb') as reader:
            info = os.fstat(reader.fileno())
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != before:
                raise ValueError('Bank changed before chunking')
            while block := reader.read(chunk_bytes):
                check(); digest.update(block)
                name = f'{offset:012d}.part'
                pin = {'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()}
                part = no_links(destination / name)
                if part.exists():
                    verify(part, pin)
                else:
                    temporary = part.with_name(name + '.writing-' + uuid.uuid4().hex)
                    try:
                        with temporary.open('xb') as writer:
                            writer.write(block); writer.flush(); os.fsync(writer.fileno())
                        os.link(temporary, part)
                    finally:
                        temporary.unlink(missing_ok=True)
                chunks.append({'offset': offset, 'name': name, 'record': pin})
                offset += len(block)
        if signature(source) != before or {'bytes': offset, 'sha256': digest.hexdigest()} != expected:
            raise ValueError('Canonical bank changed during materialization or bank hash differs')
        result = {'schema': 'confirmation_input_chunks.v1', 'bank_record': expected,
                  'source_path': str(source), 'source_signature': list(before), 'source_mtime_ns': before[3],
                  'chunk_bytes': chunk_bytes, 'chunks': chunks}
        path = destination / 'MANIFEST.json'
        if path.exists():
            if json.loads(path.read_text()) != result:
                raise ValueError('Existing chunk manifest differs from canonical source')
        else:
            json_new(path, result)
        with open_directory(destination) as directory:
            os.fsync(directory)
    return result


REMOTE_PROGRAM = r'''
import fcntl,hashlib,json,os,pathlib,re,stat,time,uuid
WORK=pathlib.Path('/workspace/jlens')
request=json.loads(__REQUEST__)
manifest=request['manifest']; bank=manifest['bank_record']; sha=bank['sha256']
def no_links(path):
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('linked remote path')
    return path
def guard():
    marker=no_links(WORK/'provider_lease_name')
    if marker.read_text().strip()!=request['lease_name']:raise ValueError('wrong owned SSH marker')
    if (WORK/'STOP').exists():raise ValueError('worker STOP is present')
    if request['setup_deadline_epoch']-time.time()<300:raise ValueError('fixed setup reserve reached')
def hash_file(path):
    path=no_links(path); before=path.stat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular remote file')
    digest=hashlib.sha256();count=0
    key=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    with path.open('rb') as source:
        if key(os.fstat(source.fileno()))!=key(before):raise ValueError('remote descriptor differs before hash')
        while block:=source.read(8*1024*1024):
            guard();digest.update(block);count+=len(block)
        if key(os.fstat(source.fileno()))!=key(before):raise ValueError('remote descriptor changed during hash')
    after=path.stat()
    key=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    if key(before)!=key(after):raise ValueError('remote file changed while hashing')
    return {'bytes':count,'sha256':digest.hexdigest()}
def fsync_dir(path):
    fd=os.open(no_links(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
guard()
if not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('invalid bank hash')
if request['worker_path'] not in ('banks/fit01.pt','banks/fit02.pt','banks/penultimate.pt','banks/positions.pt'):raise ValueError('unapproved bank destination')
if not 0<manifest['chunk_bytes']<=64*1024*1024:raise ValueError('invalid chunk bound')
expected_offset=0
for index,chunk in enumerate(manifest['chunks']):
    if chunk['offset']!=expected_offset or chunk['name']!=f'{expected_offset:012d}.part':raise ValueError('chunk order/path mismatch')
    if not 0<chunk['record']['bytes']<=manifest['chunk_bytes']:raise ValueError('invalid chunk bytes')
    if index<len(manifest['chunks'])-1 and chunk['record']['bytes']!=manifest['chunk_bytes']:raise ValueError('short nonfinal chunk')
    if not re.fullmatch('[0-9a-f]{64}',chunk['record']['sha256']):raise ValueError('invalid chunk hash')
    expected_offset+=chunk['record']['bytes']
if expected_offset!=bank['bytes']:raise ValueError('chunk total differs from bank')
chunks=no_links(WORK/'input_chunks'/sha); chunks.mkdir(parents=True,exist_ok=True)
banks=no_links(WORK/'banks');banks.mkdir(parents=True,exist_ok=True)
destination=no_links(WORK/request['worker_path'])
manifest_path=no_links(chunks/'MANIFEST.json')
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':')).encode()
if manifest_path.exists():
    if json.loads(manifest_path.read_text())!=manifest:raise ValueError('remote manifest changed')
else:
    with manifest_path.open('xb') as writer:
        writer.write(canonical(manifest)+b'\n');writer.flush();os.fsync(writer.fileno())
    fsync_dir(chunks)
operation=request['operation']
if operation=='check':
    name=request['chunk_name']
    if name not in {chunk['name'] for chunk in manifest['chunks']}:raise ValueError('unknown upload chunk')
    no_links(chunks/name)
    no_links(chunks/'.rsync-partial')
    print(json.dumps({'owned':True,'chunk':name}))
elif operation=='prepare':
    complete=destination.exists()
    if complete and hash_file(destination)!=bank:raise ValueError('changed canonical remote bank')
    print(json.dumps({'already_complete':complete}))
elif operation=='assemble':
    with no_links(chunks/'assembly.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if destination.exists():
            if hash_file(destination)!=bank:raise ValueError('changed canonical remote bank')
        else:
            temporary=no_links(banks/('.'+sha+'.assembling'))
            try:
                digest=hashlib.sha256();count=0
                with temporary.open('w+b') as writer:
                    for chunk in manifest['chunks']:
                        guard();source=no_links(chunks/chunk['name']);before=source.stat()
                        if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular chunk')
                        one=hashlib.sha256();size=0;key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                        with source.open('rb') as reader:
                            if key(os.fstat(reader.fileno()))!=key(before):raise ValueError('chunk descriptor differs before read')
                            while block:=reader.read(8*1024*1024):
                                guard();one.update(block);digest.update(block);writer.write(block);size+=len(block);count+=len(block)
                            if key(os.fstat(reader.fileno()))!=key(before):raise ValueError('chunk descriptor changed during read')
                        after=source.stat();key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                        if key(before)!=key(after) or {'bytes':size,'sha256':one.hexdigest()}!=chunk['record']:raise ValueError('chunk hash/stat mismatch')
                    if {'bytes':count,'sha256':digest.hexdigest()}!=bank:raise ValueError('assembled original bank hash mismatch')
                    writer.flush();os.fsync(writer.fileno())
                    sealed=os.fstat(writer.fileno()); key=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
                    if key(no_links(temporary).stat())!=key(sealed):raise ValueError('publication temp descriptor/path differ')
                    writer.seek(0); reread=hashlib.sha256(); checked=0
                    while block:=writer.read(8*1024*1024):guard();reread.update(block);checked+=len(block)
                    if key(os.fstat(writer.fileno()))!=key(sealed) or {'bytes':checked,'sha256':reread.hexdigest()}!=bank:raise ValueError('publication temp changed before seal')
                    os.utime(writer.fileno(),ns=(manifest['source_mtime_ns'],manifest['source_mtime_ns']))
                    os.fsync(writer.fileno()); sealed=os.fstat(writer.fileno())
                    guard()
                    if key(no_links(temporary).stat())!=key(sealed):raise ValueError('publication temp pathname changed')
                    if destination.exists():raise ValueError('canonical bank appeared during assembly')
                    os.replace(temporary,destination)
                    # Rename changes ctime; device/inode/size/mtime must stay bound.
                    identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns)
                    if identity(no_links(destination).stat())!=identity(sealed) or identity(os.fstat(writer.fileno()))!=identity(sealed):raise ValueError('published bank descriptor differs')
                    fsync_dir(banks)
            finally:
                temporary.unlink(missing_ok=True)
        guard();os.utime(destination,ns=(manifest['source_mtime_ns'],manifest['source_mtime_ns']))
        with destination.open('rb') as bank_file:os.fsync(bank_file.fileno())
        fsync_dir(banks)
        # Delete only declared verified transfer chunks after durable publication.
        for chunk in manifest['chunks']:no_links(chunks/chunk['name']).unlink(missing_ok=True)
        fsync_dir(chunks)
        result={'status':'verified_published','worker_path':request['worker_path'],'bank_record':bank,'mtime_ns':destination.stat().st_mtime_ns}
        with no_links(chunks/('ASSEMBLED-'+uuid.uuid4().hex+'.json')).open('x') as proof:
            json.dump(result,proof);proof.flush();os.fsync(proof.fileno())
        fsync_dir(chunks);print(json.dumps(result))
else:raise ValueError('unsupported bounded operation')
'''


def remote_program(request):
    return REMOTE_PROGRAM.replace('__REQUEST__', repr(json.dumps(request, sort_keys=True)))


class Processes:
    def __init__(self, transport, stop, log):
        self.transport, self.stop, self.log = transport, stop, log
        self.lock = threading.RLock()
        owner = self
        class Records(list):
            def append(self, entry):
                # run_bounded registers immediately after Popen. Cancellation
                # before that registration must also stop the late child.
                with owner.lock:
                    super().append(entry)
                    if owner.stop.is_set():
                        owner.signal(entry)
        self.records = Records()
    @staticmethod
    def signal(entry):
        if not entry.get('process_group_quiescent'):
            try:
                # Cancellation must not wait for a possibly TERM-ignoring SSH child.
                os.killpg(entry['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    def cancel(self):
        with self.lock:
            self.stop.set()
            for entry in self.records:
                self.signal(entry)
    def run(self, command, timeout, operation):
        if self.stop.is_set():
            raise RuntimeError('Upload cancelled')
        try:
            result = self.transport.run_bounded(command, timeout=timeout,
                                                operation=operation, processes=self.records)
            if self.stop.is_set():
                raise RuntimeError('Upload cancelled before subprocess completion')
            return result
        except BaseException as error:
            self.stop.set()
            self.log('process_failed', operation=operation, error_type=type(error).__name__,
                     stderr=getattr(error, 'stderr', b'').decode(errors='replace')[-4000:] if isinstance(getattr(error, 'stderr', None), bytes) else None)
            raise


def assert_live(state, initial, handoff, *, at=None):
    at = time.time() if at is None else at
    if (handoff.binding(state) != handoff.binding(initial) or state.get('ssh') != initial.get('ssh')
            or state.get('ssh_identity') != initial.get('ssh_identity')
            or state.get('setup_deadline_utc') != initial.get('setup_deadline_utc')
            or state.get('work_deadline_utc') != initial.get('work_deadline_utc')
            or state.get('status') != 'running' or state.get('halt_requested')
            or state.get('computation_stop_reason')):
        raise ValueError('Lease identity, admission or fixed deadlines changed')
    deadline = datetime.fromisoformat(state['setup_deadline_utc'].replace('Z', '+00:00')).timestamp()
    if deadline - at < SETUP_RESERVE:
        raise ValueError('Fixed setup deadline reserve reached')
    return deadline - at - SETUP_RESERVE


def upload(args):
    lease, handoff, transport = load_controller()
    root = lease.lease_root(args.lease)
    state_path = root / 'LEASE.json'
    initial = json.loads(no_links(state_path).read_text())
    binding = handoff.binding(initial)
    handoff.verify_local_sources(initial); handoff.load_contract(initial)
    if initial['controller_sources'] != CONTROLLER_PINS:
        raise ValueError('Lease uses another controller revision')
    assert_live(initial, initial, handoff)
    if not initial.get('ssh'):
        raise ValueError('No controller-bound SSH endpoint')
    spec = json.loads(Path(initial['run_spec_path']).read_text())
    launcher = ROUND / 'evaluation/launch.py'
    launcher_pin = spec['source_records']['evaluation/launch.py']
    for name in ('launch.py', 'artifacts.py'):
        verify(ROUND / 'evaluation' / name, spec['source_records']['evaluation/' + name])
    unique = {}
    for entry in spec['bank_records'].values():
        worker_path = entry['worker_path']
        if worker_path not in ('banks/fit01.pt', 'banks/fit02.pt', 'banks/penultimate.pt', 'banks/positions.pt'):
            raise ValueError('Unexpected bank destination')
        if worker_path in unique and unique[worker_path] != entry:
            raise ValueError('Duplicate bank destination has different provenance')
        unique[worker_path] = entry
    if len(unique) != 4 or len({entry['record']['sha256'] for entry in unique.values()}) != 4:
        raise ValueError('Exactly four unique frozen bank files are required')
    if sum(entry['record']['bytes'] for entry in unique.values()) > TEMP_LIMIT:
        raise ValueError('Chunk cache exceeds the fixed 8 GiB local limit')
    progress = no_links(ROUND / 'resources/parallel_upload_runs' / initial['name'])
    progress.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    def log(event, **fields):
        line = {'utc': now(), 'event': event, **fields}
        with log_lock, no_links(progress / 'EVENTS.jsonl').open('a') as handle:
            handle.write(json.dumps(line, sort_keys=True) + '\n'); handle.flush(); os.fsync(handle.fileno())
    stop = threading.Event(); processes = Processes(transport, stop, log)
    signals = {}
    def cancel(signum, frame):
        stop.set(); processes.cancel()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signals[signum] = signal.signal(signum, cancel)
    snapshots = {}
    def guard():
        if stop.is_set():
            raise RuntimeError('Upload cancelled')
        state = json.loads(no_links(state_path).read_text())
        remaining = assert_live(state, initial, handoff)
        for source, expected in snapshots.items():
            if signature(Path(source)) != expected:
                raise ValueError('Canonical local bank changed')
        if (root / 'STOP').exists():
            raise ValueError('Local STOP is present')
        return remaining
    commands = lease.ssh_options(root, initial); host = 'root@' + initial['ssh']['host']
    deadline = lease.epoch(initial['setup_deadline_utc'])
    def remote(operation, entry, manifest, **fields):
        remaining = guard()
        request = {'operation': operation, 'lease_name': initial['name'], 'setup_deadline_epoch': deadline,
                   'worker_path': entry['worker_path'], 'manifest': manifest, **fields}
        script = remote_program(request)
        command = 'python -c ' + shlex.quote(script)
        result = processes.run([*commands, host, command], min(600, remaining), 'parallel_remote_' + operation)
        return json.loads(result.stdout)
    invocation = {'schema': 'confirmation_parallel_upload.v1', 'binding': binding, 'streams': STREAMS,
                  'chunk_bytes': CHUNK_BYTES, 'setup_reserve_seconds': SETUP_RESERVE,
                  'setup_deadline_utc': initial['setup_deadline_utc'], 'bank_records': spec['bank_records'],
                  'helper_record': record(__file__), 'controller_sources': CONTROLLER_PINS,
                  'launcher_record': launcher_pin, 'run_spec_record': initial['run_spec_record']}
    lock = no_links(progress / '.lock').open('a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        json_new(progress / ('INVOCATION-' + uuid.uuid4().hex + '.json'), invocation)
        log('begin', binding=binding, streams=STREAMS)
        manifests = {}
        # Materialization is local only; every original bank receives its own
        # before/after stat and full-SHA verification before any chunk is sent.
        for worker_path, entry in unique.items():
            guard(); source = no_links(entry['receiver_path'])
            verify(source, entry['record']); snapshots[str(source)] = signature(source)
            manifest = split_bank(source, entry['record'], ROUND / 'resources/input_chunks', check=guard)
            if manifest['chunk_bytes'] != CHUNK_BYTES:
                raise ValueError('Production chunk size changed')
            manifests[worker_path] = manifest
            log('materialized', worker_path=worker_path, manifest=manifest)
        for worker_path, entry in unique.items():
            manifest = manifests[worker_path]
            ready = remote('prepare', entry, manifest)
            if not ready['already_complete']:
                def send(chunk):
                    guard(); handoff.verify_local_sources(json.loads(state_path.read_text()))
                    source = ROUND / 'resources/input_chunks' / entry['record']['sha256'] / chunk['name']
                    verify(source, chunk['record'])
                    remote('check', entry, manifest, chunk_name=chunk['name'])
                    remaining = guard()
                    destination = WORK + '/input_chunks/' + entry['record']['sha256'] + '/' + chunk['name']
                    processes.run(['rsync', '--partial', '--partial-dir=.rsync-partial', '--protect-args', '-t',
                                   '-e', shlex.join(commands), str(source), host + ':' + destination],
                                  remaining, 'parallel_chunk_upload')
                    log('chunk_uploaded', worker_path=worker_path, chunk=chunk)
                with ThreadPoolExecutor(max_workers=STREAMS) as executor:
                    pending = {executor.submit(send, chunk) for chunk in manifest['chunks']}
                    try:
                        while pending:
                            guard()
                            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                            for future in done:
                                future.result()
                    except BaseException:
                        processes.cancel()
                        for future in pending:
                            future.cancel()
                        raise
            assembled = remote('assemble', entry, manifest)
            if (assembled.get('bank_record') != entry['record']
                    or assembled.get('mtime_ns') != manifest['source_mtime_ns']):
                raise ValueError('Remote assembly receipt differs')
            log('bank_verified_published', result=assembled)
        guard()
        for entry in unique.values():
            verify(entry['receiver_path'], entry['record'])
        handoff.verify_local_sources(json.loads(state_path.read_text()))
        for name in ('launch.py', 'artifacts.py'):
            verify(ROUND / 'evaluation' / name, spec['source_records']['evaluation/' + name])
        log('unchanged_launcher_start', launcher_record=launcher_pin)
        # Original bootstrap, rsync skips, BANKS_READY, native-development gate,
        # and external science acknowledgement stay in their unchanged code.
        launched = processes.run([sys.executable, '-B', str(launcher), '--lease', str(root), '--phase', 'start'],
                                 guard(), 'original_confirmation_launcher')
        log('completed', launcher_stdout=launched.stdout.decode(errors='replace'), processes=processes.records)
        print(json.dumps({'status': 'banks_prestaged_and_original_start_completed', 'progress': str(progress), 'binding': binding}))
    except BaseException as error:
        processes.cancel()
        log('failed', error_type=type(error).__name__, error=str(error), processes=processes.records)
        raise
    finally:
        lock.close()
        for signum, previous in signals.items():
            signal.signal(signum, previous)


def self_test():
    """Tiny files exercise the actual remote program locally, without SSH."""
    import shutil
    lease, handoff, transport = load_controller()
    base = Path(tempfile.mkdtemp(prefix='parallel-upload-self-test-'))
    source = base / 'bank.pt'; source.write_bytes(bytes(range(251)) * 3)
    os.utime(source, ns=(1700000000123456789, 1700000000123456789))
    expected = record(source)
    manifest = split_bank(source, expected, base / 'cache', chunk_bytes=64)
    cases = []
    def rejects(operation, label):
        try:
            operation()
        except Exception:
            cases.append(label); return
        raise AssertionError('Unexpected acceptance: ' + label)
    def stage(name, value):
        work = base / name;work.mkdir();(work / 'provider_lease_name').write_text('test-lease')
        chunkroot = work / 'input_chunks' / value['bank_record']['sha256'];chunkroot.mkdir(parents=True)
        for chunk in value['chunks']:
            shutil.copyfile(base / 'cache' / expected['sha256'] / chunk['name'], chunkroot / chunk['name'])
        return work
    def execute(work, value=manifest, prefix='', **fields):
        request = {'operation': 'assemble', 'lease_name': 'test-lease', 'setup_deadline_epoch': time.time() + 900,
                   'worker_path': 'banks/fit01.pt', 'manifest': value, **fields}
        program = remote_program(request).replace("WORK=pathlib.Path('/workspace/jlens')", 'WORK=pathlib.Path(' + repr(str(work)) + ')')
        return transport.run_bounded([sys.executable, '-I', '-B', '-c', prefix + program], timeout=10, operation='self_test_remote_program')
    work = stage('correct', manifest);execute(work)
    assert record(work / 'banks/fit01.pt') == expected
    assert (work / 'banks/fit01.pt').stat().st_mtime_ns == source.stat().st_mtime_ns
    assert not list((work / 'input_chunks' / expected['sha256']).glob('*.part'))
    execute(work);cases.append('real_remote_program_reassembles_hashes_publishes_mtime_and_resumes')
    work = stage('bad_chunk', manifest);part=work/'input_chunks'/expected['sha256']/manifest['chunks'][0]['name']
    part.write_bytes(b'x' * part.stat().st_size)
    rejects(lambda: execute(work), 'changed_chunk_rejected_without_canonical_publication')
    assert not (work/'banks/fit01.pt').exists()
    bad = json.loads(json.dumps(manifest));bad['bank_record']['sha256'] = '0' * 64
    work = stage('bad_bank', bad);rejects(lambda: execute(work,bad), 'whole_bank_sha_mismatch_rejected')
    work = stage('wrong_identity', manifest);(work/'provider_lease_name').write_text('other-lease')
    rejects(lambda: execute(work), 'remote_identity_rejected')
    work = stage('deadline', manifest)
    rejects(lambda: execute(work,setup_deadline_epoch=time.time()+299), 'remote_fixed_deadline_reserve_rejected')
    work = stage('stopped', manifest);(work/'STOP').write_text('stop')
    rejects(lambda: execute(work), 'remote_STOP_rejected')
    work = stage('canonical_changed', manifest);(work/'banks').mkdir();(work/'banks/fit01.pt').write_bytes(b'changed')
    rejects(lambda: execute(work), 'changed_existing_canonical_bank_not_overwritten')
    assert (work/'banks/fit01.pt').read_bytes() == b'changed'
    # Swap the parent only during open, restoring it before any path check.
    # The file descriptor still references another inode with expected bytes.
    for attack in ('canonical_parent_swap', 'chunk_parent_swap'):
        work=stage(attack,manifest)
        if attack=='canonical_parent_swap':
            target=work/'banks/fit01.pt';target.parent.mkdir();target.write_bytes(b'x'*expected['bytes'])
        else:
            target=work/'input_chunks'/expected['sha256']/manifest['chunks'][0]['name']
        alternate=work/'alternate';alternate.mkdir()
        (alternate/target.name).write_bytes(source.read_bytes() if attack=='canonical_parent_swap' else target.read_bytes())
        parked=work/'parked'
        prefix="import pathlib\n_original_open=pathlib.Path.open\n" + \
            "def _swapped_open(path,*args,**kwargs):\n" + \
            "    if str(path)=="+repr(str(target))+" and args and args[0]=='rb':\n" + \
            "        parent=path.parent; parked=pathlib.Path("+repr(str(parked))+"); alternate=pathlib.Path("+repr(str(alternate))+")\n" + \
            "        parent.rename(parked);alternate.rename(parent)\n" + \
            "        try:return _original_open(path,*args,**kwargs)\n" + \
            "        finally:parent.rename(alternate);parked.rename(parent)\n" + \
            "    return _original_open(path,*args,**kwargs)\npathlib.Path.open=_swapped_open\n"
        rejects(lambda:execute(work,prefix=prefix,operation='prepare' if attack=='canonical_parent_swap' else 'assemble'),attack+'_descriptor_rejected')
        if attack=='canonical_parent_swap':assert target.read_bytes()==b'x'*expected['bytes']
        else:assert not (work/'banks/fit01.pt').exists()
    original = source.read_bytes();changed=[False]
    def change_during_split():
        if not changed[0]:
            changed[0]=True;source.write_bytes(b'z'*len(original))
    rejects(lambda: split_bank(source,expected,base/'changed_cache',chunk_bytes=64,check=change_during_split), 'canonical_source_change_during_split_rejected')
    state={'name':'test-lease','pod_id':'test-pod','run_id':'test-run','account_id':'test-account',
           'run_spec_sha256':'1'*64,'output_contract_sha256':'2'*64,'controller_sources':CONTROLLER_PINS,
           'status':'running','halt_requested':False,'ssh':{'host':'127.0.0.1','port':22},'ssh_identity':'/tmp/key',
           'setup_deadline_utc':lease.stamp(time.time()+900),'work_deadline_utc':lease.stamp(time.time()+1800)}
    assert_live(state,state,handoff)
    rejects(lambda: assert_live({**state,'name':'other'},state,handoff), 'local_controller_identity_rejected')
    rejects(lambda: assert_live({**state,'halt_requested':True},state,handoff), 'local_controller_halt_rejected')
    rejects(lambda: assert_live(state,state,handoff,at=lease.epoch(state['setup_deadline_utc'])-299), 'local_setup_reserve_rejected')
    stop=threading.Event();runner=Processes(transport,stop,lambda *args,**kwargs:None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(runner.run,[sys.executable,'-c','import time;time.sleep(30)'],30,'self_test_cancel')
        until=time.time()+5
        while not runner.records and time.time()<until:time.sleep(.01)
        runner.cancel();rejects(future.result,'active_process_group_cancelled')
    assert runner.records and all(row['process_group_quiescent'] for row in runner.records)
    entered=threading.Event();release=threading.Event()
    def delayed_spawn(*args,**kwargs):
        entered.set()
        if not release.wait(5):raise AssertionError('late-spawn test not released')
        return transport.run_bounded(*args,**kwargs)
    runner=Processes(types.SimpleNamespace(run_bounded=delayed_spawn),threading.Event(),lambda *args,**kwargs:None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(runner.run,[sys.executable,'-c','import time;time.sleep(30)'],30,'self_test_late_spawn')
        assert entered.wait(5)
        runner.cancel();release.set()
        rejects(lambda:future.result(timeout=5),'one_cancel_before_child_registration_quiesces_late_child')
        assert future.done(), 'single cancellation failed to quiesce late child promptly'
    assert runner.records and all(row['process_group_quiescent'] for row in runner.records)
    ready=base/'term_ignoring_ready'
    runner=Processes(transport,threading.Event(),lambda *args,**kwargs:None)
    command=[sys.executable,'-c',"import pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("+repr(str(ready))+").touch();time.sleep(30)"]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(runner.run,command,30,'self_test_term_ignoring')
        until=time.time()+5
        while not ready.exists() and time.time()<until:time.sleep(.01)
        assert ready.exists()
        runner.cancel()
        rejects(lambda:future.result(timeout=5),'one_cancel_quiesces_SIGTERM_ignoring_child')
        assert future.done(), 'TERM-ignoring child survived cancellation'
    assert runner.records and all(row['process_group_quiescent'] for row in runner.records)
    proof={'schema':'parallel_upload_self_test.v1','status':'passed','cases':cases,'source_record':record(__file__),
           'controller_sources':CONTROLLER_PINS,'no_cloud_or_ssh':True,'temporary_root':str(base)}
    json_new(base/'PROOF.json',proof)
    print(json.dumps({'status':'passed','cases':len(cases),'proof':str(base/'PROOF.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if args.lease is not None:parser.error('--self-test cannot accept a lease')
        self_test()
    else:
        if args.lease is None:parser.error('--lease is required')
        upload(args)


if __name__=='__main__':main()
