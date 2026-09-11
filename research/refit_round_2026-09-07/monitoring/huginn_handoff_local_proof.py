"""Focused synthetic proof; no SSH, real remote state, tensors or outcomes."""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import stat
import signal
import time
import subprocess
import sys
import tempfile
from unittest import mock

sys.dont_write_bytecode = True
SOURCE = Path('/tmp/collect_huginn_handoff06.py')
PROOF = Path(tempfile.mkdtemp(prefix='huginn-handoff-proof-'))
spec = importlib.util.spec_from_file_location('handoff_candidate', SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
REAL_RUN = subprocess.run
GROUPS = []

def record(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + '\n')

def expect_error(fn, text=None):
    try:
        fn()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, KeyboardInterrupt, OSError, RuntimeError) as error:
        if text is not None:
            assert text in str(error), (type(error).__name__, str(error), text)
        return type(error).__name__
    raise AssertionError('invalid fixture was accepted')

class Fixture:
    def __init__(self, name):
        self.root = PROOF / name
        self.remote = self.root / 'remote'
        self.results = self.remote / 'results'
        self.deployment = self.root / 'deployment'
        self.monitoring = self.root / 'monitoring'
        self.ledger = self.root / 'ledger'
        self.prior = self.root / 'prior'
        self.source = self.root / 'candidate.py'
        self.source.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, self.source)
        self.status = {'schema_version': 1, 'lease_name': M.LEASE_NAME,
                       'phase': 'complete', 'updated_utc': 'synthetic-time', 'setup_complete': True}
        dump(self.remote / 'status.json', self.status)
        (self.remote / 'provider_lease_name').write_text(M.LEASE_NAME + '\n')
        self.state = {'name': M.LEASE_NAME, 'pod_id': M.POD_ID, 'status': 'running',
                      'ssh': {'host': '127.0.0.1', 'port': 2222}, 'ssh_identity': '/tmp/synthetic-no-key'}
        dump(self.ledger / 'LEASE.json', self.state)
        self.pins = {}
        for name in M.PREREQUISITES:
            path = self.prior / name
            dump(path, {'synthetic_prerequisite': name})
            other = self.results / name
            other.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, other)
            self.pins[name] = record(path)
        for name in M.expected_paths():
            path = self.results / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately opaque invalid tensor/archive bytes: collector only hashes.
            path.write_bytes(('opaque synthetic member ' + name + '\n').encode())
            if name in M.CONTRACTS:
                self.deployment.mkdir(exist_ok=True)
                shutil.copyfile(path, self.deployment / name)
        self.experiment = {'status': 'complete', 'lease_name': M.LEASE_NAME,
                          'completed_utc': '2026-09-09T00:00:00+00:00',
                          'evaluations': {'ouro_evaluation': self.pins['ouro_evaluation/COMPLETE.json'],
                                          'controls_evaluation': self.pins['controls_evaluation/COMPLETE.json'],
                                          'huginn_evaluation': record(self.results / 'huginn_evaluation/COMPLETE.json')}}
        dump(self.results / 'EXPERIMENT_COMPLETE.json', self.experiment)
        self.calls = []
        self.after_copy = None
        self.copy_exception = None
        self.after_snapshot = None
        self.snapshots = 0

    @contextlib.contextmanager
    def bind(self):
        values = {'DEPLOYMENT': self.deployment, 'MONITORING': self.monitoring,
                  'LEASE_ROOT': self.ledger, 'PREREQUISITE_ROOT': self.prior,
                  'PREREQUISITES': self.pins, '__file__': str(self.source)}
        with mock.patch.dict(M.__dict__, values):
            yield self

    def remote_run(self, script=None, pod=None, check=True):
        env = os.environ.copy()
        env.pop('RUNPOD_POD_ID', None)
        if pod is not None:
            env['RUNPOD_POD_ID'] = pod
        return REAL_RUN([sys.executable, '-c', script or M.remote_script(str(self.remote))],
                        capture_output=True, text=True, check=check, timeout=5, env=env)

    def snapshot(self):
        return json.loads(self.remote_run().stdout)

    def fake_run(self, command, **kwargs):
        self.calls.append({'program': command[0], 'timeout': kwargs.get('timeout')})
        kwargs['processes'].append({'operation': kwargs['operation'], 'pid': 0, 'timeout_seconds': kwargs['timeout'], 'returncode': 0, 'process_group_quiescent': True})
        if command[0] == 'ssh':
            assert command[-1] == 'python -c ' + shlex.quote(M.remote_script())
            assert command[-2] == 'root@127.0.0.1'
            assert kwargs.get('timeout') == 120
            self.snapshots += 1
            value = self.snapshot()
            if self.after_snapshot:
                self.after_snapshot(self, self.snapshots)
            return subprocess.CompletedProcess(command, 0, json.dumps(value), '')
        assert command[0] == 'rsync', command
        assert kwargs.get('timeout') == 900
        assert '-rlt' in command and '--partial' in command and '--timeout=60' in command and '--from0' in command
        assert not any(arg.startswith('--inplace') for arg in command)
        listing = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--files-from=')))
        names = listing.read_bytes().split(b'\0')
        assert names[-1] == b''
        names = [name.decode() for name in names[:-1]]
        assert names == sorted(M.expected_paths()) and len(names) == 22
        copied = Path(command[-1])
        assert copied.name == 'results' and copied.parent.parent == self.monitoring
        for name in names:
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.results / name, target)
        if self.after_copy:
            self.after_copy(self, copied)
        if self.copy_exception:
            raise self.copy_exception
        return subprocess.CompletedProcess(command, 0)

    def collect(self):
        # Fake only the SSH/rsync transport boundary; remote snapshot code runs
        # in a real local Python process against small synthetic files.
        with self.bind(), mock.patch.object(M, 'run_bounded', self.fake_run):
            return M.collect()

    def operation(self):
        operations = list(self.monitoring.glob('huginn_handoff_*'))
        assert len(operations) == 1, operations
        return operations[0]

    def assert_incomplete(self):
        operation = self.operation()
        assert not (operation / 'COPY_VERIFIED.json').exists()
        assert json.loads((operation / 'INCOMPLETE.json').read_text())['status'] == 'incomplete'
        assert not (self.ledger / 'retrieved').exists()
        assert not (self.ledger / 'RETRIEVAL_VERIFIED.json').exists()
        assert not (self.ledger / '.sync.lock').exists()
        return operation


def group(name, fn):
    fn()
    GROUPS.append(name)


def positive():
    f = Fixture('positive')
    result = f.collect()
    operation = f.operation()
    receipt = json.loads((operation / 'COPY_VERIFIED.json').read_text())
    assert result['files'] == 22 and result['status'] == 'passed'
    assert set(receipt['files']) == M.expected_paths()
    assert receipt['prerequisites'] == f.pins
    assert receipt['bindings']['evaluations'] == f.experiment['evaluations']
    assert receipt['source'] == record(SOURCE)
    assert len(receipt['processes']) == 3 and all(row['process_group_quiescent'] for row in receipt['processes'])
    assert str(Path(M.prefetch_transport.__file__)) in receipt['reused_sources']
    assert not (operation / 'INCOMPLETE.json').exists()
    assert (operation / 'COPY_PENDING.json').samefile(operation / 'COPY_VERIFIED.json')
    assert [row['program'] for row in f.calls] == ['ssh', 'rsync', 'ssh']
    assert set(path.relative_to(operation / 'results').as_posix() for path in (operation / 'results').rglob('*') if path.is_file()) == M.expected_paths()
    assert not (f.ledger / '.sync.lock').exists() and not (f.ledger / 'RETRIEVAL_VERIFIED.json').exists()
    # Prerequisites are validation inputs, never transfer members.
    assert not any(name in receipt['files'] for name in f.pins)

group('complete_22_file_handoff_with_raw_completion_bindings_and_opaque_binaries', positive)


def remote_status_cases():
    cases = [('phase', 'failed'), ('phase', 'awaiting_ouro_interpretation'),
             ('phase', 'stopped'), ('setup_complete', False), ('setup_complete', 1),
             ('lease_name', 'other'), ('stop_requested', True)]
    for index, (key, value) in enumerate(cases):
        f = Fixture('remote-status-' + str(index))
        f.status[key] = value
        dump(f.remote / 'status.json', f.status)
        with f.bind():
            assert f.remote_run(check=False).returncode != 0
    for kind in ('regular-stop', 'dangling-stop-link', 'provider', 'provider-link', 'pod'):
        f = Fixture('remote-ownership-' + kind)
        if kind == 'regular-stop': (f.remote / 'STOP').touch()
        elif kind == 'dangling-stop-link': (f.remote / 'STOP').symlink_to(f.remote / 'missing')
        elif kind == 'provider': (f.remote / 'provider_lease_name').write_text('wrong')
        elif kind == 'provider-link':
            (f.remote / 'provider_lease_name').unlink()
            (f.remote / 'provider_lease_name').symlink_to(f.remote / 'status.json')
        with f.bind():
            assert f.remote_run(pod='wrong' if kind == 'pod' else None, check=False).returncode != 0
    f = Fixture('remote-optional-correct-pod')
    with f.bind():
        assert f.remote_run(pod=M.POD_ID).returncode == 0

group('terminal_setup_stop_lease_provider_and_optional_pod_gates', remote_status_cases)


def remote_after_status():
    for kind in ('stop', 'phase', 'owner'):
        f = Fixture('remote-after-status-' + kind)
        with f.bind():
            script = M.remote_script(str(f.remote))
            injection = { 'stop': "(workspace / 'STOP').touch()\n",
                          'phase': "v=json.loads((workspace/'status.json').read_text());v['phase']='failed';(workspace/'status.json').write_text(json.dumps(v))\n",
                          'owner': "(workspace/'provider_lease_name').write_text('wrong')\n"}[kind]
            script = script.replace('after, ownership_after = status()', injection + 'after, ownership_after = status()')
            assert f.remote_run(script=script, check=False).returncode != 0

group('remote_terminal_and_ownership_gates_repeat_after_hashing', remote_after_status)


def remote_geometry():
    for kind in ('missing-seed-file', 'extra-file', 'extra-empty-directory', 'symlink', 'fifo'):
        f = Fixture('geometry-' + kind)
        target = f.results / 'huginn_evaluation/seeds/2026090804/cache.pt'
        if kind == 'missing-seed-file': target.unlink()
        elif kind == 'extra-file': (target.parent / 'unexpected').write_bytes(b'extra')
        elif kind == 'extra-empty-directory': (target.parent / 'unexpected').mkdir()
        elif kind == 'symlink':
            target.unlink(); target.symlink_to(target.parent / 'arrays.npz')
        else:
            target.unlink(); os.mkfifo(target)
        with f.bind():
            assert f.remote_run(check=False).returncode != 0

group('exact_remote_geometry_rejects_missing_extra_linked_and_special_members', remote_geometry)


def bindings_and_prerequisites():
    for kind in ('raw-prerequisite', 'experiment-lease', 'experiment-status', 'experiment-complete-record', 'experiment-extra-key'):
        f = Fixture('bindings-' + kind)
        if kind == 'raw-prerequisite':
            (f.results / 'ouro_evaluation/OWNER.json').write_text('wrong')
        else:
            exp = copy.deepcopy(f.experiment)
            if kind == 'experiment-lease': exp['lease_name'] = 'wrong'
            elif kind == 'experiment-status': exp['status'] = 'incomplete'
            elif kind == 'experiment-complete-record': exp['evaluations']['huginn_evaluation']['sha256'] = '0' * 64
            else: exp['required_bindings'] = {}
            dump(f.results / 'EXPERIMENT_COMPLETE.json', exp)
        with f.bind():
            assert f.remote_run(check=False).returncode != 0

group('raw_prerequisite_pins_and_exact_experiment_completion_schema', bindings_and_prerequisites)


def file_changes_during_remote_hash():
    f = Fixture('remote-hash-drift')
    with f.bind():
        script = M.remote_script(str(f.remote))
        script = script.replace('    regular(path)\n    after = path.stat()',
             "    if path.name == 'cache.pt': path.write_bytes(path.read_bytes() + b'drift')\n    regular(path)\n    after = path.stat()")
        result = f.remote_run(script=script, check=False)
        assert result.returncode != 0 and 'changed while hashing' in result.stderr

group('remote_per_file_stat_signature_detects_mid_hash_mutation', file_changes_during_remote_hash)


def snapshot_schema_and_contracts():
    f = Fixture('snapshot-schema')
    with f.bind():
        snapshot = f.snapshot()
        contracts = M.local_inputs()
        M.verify_snapshot(snapshot, contracts)
        for kind in ('contract', 'bool-size', 'extra-file', 'ownership', 'binding', 'status', 'prerequisite'):
            changed = copy.deepcopy(snapshot)
            if kind == 'contract': changed['files']['environment.json']['sha256'] = '0' * 64
            elif kind == 'bool-size': changed['files']['huginn_evaluation/OWNER.json']['bytes'] = True
            elif kind == 'extra-file': changed['files']['unlisted'] = {'bytes': 1, 'sha256': '0' * 64}
            elif kind == 'ownership': changed['ownership_after']['pod_id_if_set'] = 'wrong'
            elif kind == 'binding': changed['bindings']['experiment_complete']['sha256'] = '0' * 64
            elif kind == 'status': changed['status_after']['phase'] = 'failed'
            else: changed['prerequisites']['controls_evaluation/OWNER.json']['sha256'] = '0' * 64
            expect_error(lambda: M.verify_snapshot(changed, contracts))

group('local_snapshot_verifier_independently_enforces_schema_contracts_and_bindings', snapshot_schema_and_contracts)


def changed_copy_and_local_inputs():
    for kind in ('remote-content', 'local-content', 'local-extra-directory', 'local-extra-file', 'local-symlink',
                 'local-contract', 'local-prerequisite', 'source', 'local-halt', 'endpoint'):
        f = Fixture('copy-drift-' + kind)
        def mutate(f, copied, kind=kind):
            if kind == 'remote-content': (f.results / 'huginn_evaluation/seeds/2026090803/arrays.npz').write_bytes(b'drift')
            elif kind == 'local-content': (copied / 'huginn_evaluation/seeds/2026090803/cache.pt').write_bytes(b'drift')
            elif kind == 'local-extra-directory': (copied / 'unexpected').mkdir()
            elif kind == 'local-extra-file': (copied / 'unexpected').write_bytes(b'drift')
            elif kind == 'local-symlink':
                target = copied / 'huginn_evaluation/seeds/2026090803/cache.pt'
                target.unlink(); target.symlink_to(target.parent / 'arrays.npz')
            elif kind == 'local-contract': (f.deployment / 'run_spec.json').write_bytes(b'drift')
            elif kind == 'local-prerequisite': (f.prior / 'controls_evaluation/OWNER.json').write_bytes(b'drift')
            elif kind == 'source': f.source.write_bytes(f.source.read_bytes() + b'\n# drift\n')
            else:
                state = copy.deepcopy(f.state)
                if kind == 'local-halt': state['halt_requested'] = True
                else: state['ssh']['port'] = 2223
                dump(f.ledger / 'LEASE.json', state)
        f.after_copy = mutate
        expect_error(f.collect)
        f.assert_incomplete()

group('copy_remote_local_contract_prerequisite_source_and_lease_drift_never_publish', changed_copy_and_local_inputs)


def local_entry_gates():
    for kind in ('lease', 'pod', 'status', 'halt', 'final', 'final-dangling-link', 'prerequisite'):
        f = Fixture('local-entry-' + kind)
        state = copy.deepcopy(f.state)
        if kind == 'lease': state['name'] = 'wrong'
        elif kind == 'pod': state['pod_id'] = 'wrong'
        elif kind == 'status': state['status'] = 'terminated'
        elif kind == 'halt': state['halt_requested'] = True
        elif kind == 'final': dump(f.ledger / 'RETRIEVAL_VERIFIED.json', {'status': 'passed'})
        elif kind == 'final-dangling-link': (f.ledger / 'RETRIEVAL_VERIFIED.json').symlink_to(f.ledger / 'missing')
        else: (f.prior / 'ouro_evaluation/OWNER.json').write_text('wrong')
        dump(f.ledger / 'LEASE.json', state)
        expect_error(f.collect)
        assert not f.calls and not list(f.monitoring.glob('huginn_handoff_*'))

group('local_lease_final_retrieval_and_prerequisite_gates_precede_all_transport', local_entry_gates)


def failure_receipts():
    for kind in ('timeout', 'interrupt', 'transport'):
        f = Fixture('failure-' + kind)
        f.copy_exception = {'timeout': subprocess.TimeoutExpired(['synthetic-rsync'], 900),
                            'interrupt': KeyboardInterrupt(),
                            'transport': subprocess.CalledProcessError(255, ['synthetic-rsync'])}[kind]
        expect_error(f.collect)
        op = f.assert_incomplete()
        assert (op / 'REMOTE_BEFORE.json').exists() and not (op / 'REMOTE_AFTER.json').exists()

group('transport_timeout_interruption_and_error_leave_incomplete_without_acceptance', failure_receipts)


def receipt_atomicity():
    for kind in ('pending-write', 'link', 'publication-fsync'):
        f = Fixture('receipt-' + kind)
        old_new, old_link, old_fsync = M.io._new_json, M.os.link, M.io._fsync_dir
        def new_json(path, value):
            if kind == 'pending-write' and path.name == 'COPY_PENDING.json':
                path.write_text('{partial')
                raise OSError('synthetic write failure')
            return old_new(path, value)
        def link(src, dst, **kwargs):
            if kind == 'link' and Path(dst).name == 'COPY_VERIFIED.json':
                raise OSError('synthetic link failure')
            return old_link(src, dst, **kwargs)
        def fsync(path):
            if kind == 'publication-fsync' and (Path(path) / 'COPY_VERIFIED.json').exists():
                raise OSError('synthetic directory sync failure')
            return old_fsync(path)
        with mock.patch.object(M.io, '_new_json', new_json), mock.patch.object(M.os, 'link', link), mock.patch.object(M.io, '_fsync_dir', fsync):
            expect_error(f.collect)
        f.assert_incomplete()

group('receipt_write_link_and_fsync_failures_never_leave_a_valid_marker', receipt_atomicity)


PROCESS_CASES = []

def process_cleanup_cases():
    for kind in ('success', 'nonzero', 'timeout', 'interruption', 'exited-leader', 'cleanup-failure'):
        directory = PROOF / ('process-' + kind)
        directory.mkdir()
        leader_file, child_file = directory / 'leader.pid', directory / 'child.pid'
        script = "import os,sys,time\nfrom pathlib import Path\n"
        script += 'Path(' + repr(str(leader_file)) + ').write_text(str(os.getpid()))\n'
        if kind in ('nonzero', 'timeout', 'interruption', 'exited-leader'):
            script += "child=os.fork()\nif child == 0:\n"
            script += '    Path(' + repr(str(child_file)) + ').write_text(str(os.getpid()))\n'
            script += '    os.close(1); os.close(2)\n    time.sleep(60)\n    os._exit(0)\n'
            # Ensure the child exists and recorded its PID before leader exit.
            script += 'while not Path(' + repr(str(child_file)) + ').exists(): time.sleep(.005)\n'
        if kind == 'nonzero': script += "print('stdout-proof'); print('stderr-proof', file=sys.stderr); sys.exit(7)\n"
        elif kind in ('timeout', 'interruption'): script += 'time.sleep(60)\n'
        else: script += "print('completed-proof')\n"
        receipts = []
        timeout = .5 if kind == 'timeout' else 3
        command = [sys.executable, '-c', script]
        error = None
        old_handler = signal.getsignal(signal.SIGALRM)
        def interrupt(signum, frame): raise KeyboardInterrupt('synthetic process interrupt')
        def cleanup_fail(process):
            M.prefetch_transport.stop_process_original(process)
            raise M.prefetch_transport.StopPrefetch('synthetic cleanup failure after actual cleanup')
        M.prefetch_transport.stop_process_original = M.prefetch_transport.stop_process
        try:
            if kind == 'interruption':
                signal.signal(signal.SIGALRM, interrupt)
                signal.setitimer(signal.ITIMER_REAL, .5)
            with mock.patch.object(M.prefetch_transport, 'stop_process', cleanup_fail) if kind == 'cleanup-failure' else contextlib.nullcontext():
                try:
                    result = M.run_bounded(command, operation='synthetic-' + kind, timeout=timeout, processes=receipts)
                except BaseException as caught:
                    error = caught
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            del M.prefetch_transport.stop_process_original
        assert len(receipts) == 1
        receipt = receipts[0]
        if kind in ('success', 'exited-leader'):
            assert error is None and result.returncode == 0 and result.stdout == 'completed-proof\n'
        elif kind == 'nonzero':
            assert isinstance(error, subprocess.CalledProcessError) and error.returncode == 7
            assert error.stdout == 'stdout-proof\n' and error.stderr == 'stderr-proof\n'
        elif kind == 'timeout': assert isinstance(error, subprocess.TimeoutExpired) and error.timeout == .5
        elif kind == 'interruption': assert isinstance(error, KeyboardInterrupt)
        else:
            assert isinstance(error, M.prefetch_transport.StopPrefetch)
            assert receipt['process_group_quiescent'] is False and receipt['cleanup_error_type'] == 'StopPrefetch'
        if kind != 'cleanup-failure': assert receipt['process_group_quiescent'] is True
        # Check the specifically created leader and descendant are gone or dead;
        # this is actual local /proc evidence, not absence in a sandbox namespace.
        pids = [int(leader_file.read_text())]
        if kind in ('nonzero', 'timeout', 'interruption', 'exited-leader'):
            pids.append(int(child_file.read_text()))
        observed = {}
        for pid in pids:
            path = Path('/proc') / str(pid) / 'stat'
            try: state = path.read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError: state = 'absent'
            assert state in ('absent', 'Z', 'X'), (kind, pid, state)
            observed[str(pid)] = state
        PROCESS_CASES.append({'case': kind, 'receipt': receipt, 'observed_local_states': observed,
                              'exception_type': type(error).__name__ if error else None})
    dump(PROOF / 'PROCESS_PROOF.json', {'status': 'passed', 'cases': PROCESS_CASES})

group('real_local_process_success_nonzero_timeout_interrupt_exited_leader_and_cleanup_failure', process_cleanup_cases)


def cleanup_failure_blocks_publication():
    f = Fixture('cleanup-prevents-copy')
    def fail_cleanup(f, copied):
        raise M.prefetch_transport.StopPrefetch('synthetic runner cleanup failure')
    f.after_copy = fail_cleanup
    expect_error(f.collect)
    f.assert_incomplete()

group('runner_cleanup_failure_prevents_copy_receipt', cleanup_failure_blocks_publication)

# Keep static acceptance evidence explicit: no model/array/scientific imports or
# lease control/final collector call sites are added by this helper.
tree = ast.parse(SOURCE.read_text())
imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
assert not imports.intersection({'torch', 'numpy', 'analyze_huginn', 'evaluate_huginn'})
assert 'rsync' in SOURCE.read_text() and '--files-from=' in SOURCE.read_text()
assert 'flock' not in SOURCE.read_text() and 'terminate_pod' not in SOURCE.read_text()
GROUPS.append('static_no_model_deserialization_worker_control_or_final_collector_mutation')

receipt = {'status': 'passed', 'source_path': str(SOURCE), 'source': record(SOURCE),
           'proof_script': record(Path(__file__)), 'fixture_groups': GROUPS,
           'scope': 'Synthetic local fixtures only. Real embedded remote snapshot code runs in local Python; SSH and rsync calls are stubbed. Opaque tiny PT/NPZ bytes are hashed, never deserialized. No actual operation or cloud/outcome inputs read. Transport integration fixtures establish receipt behavior; separate real local Python process fixtures prove group cleanup on success, nonzero, timeout, interruption, an exited leader with a live descendant, and cleanup failure propagation.'}
dump(PROOF / 'PROOF.json', receipt)
print(json.dumps({'proof': str(PROOF / 'PROOF.json'), 'groups': len(GROUPS), 'source': receipt['source']}, indent=2))
