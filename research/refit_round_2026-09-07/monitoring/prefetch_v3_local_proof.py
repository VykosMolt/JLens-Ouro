"""Focused local synthetic verification of the frozen V2-to-V3 delta."""
from pathlib import Path
import ast
import base64
import contextlib
import difflib
import hashlib
import importlib.util
import io as streamio
import json
import os
import tempfile

old_path = Path('/tmp/prefetch_completed_ouro06_v2.py')
source = Path('/tmp/prefetch_completed_ouro06_v3.py')
old, new = old_path.read_text(), source.read_text()
spec = importlib.util.spec_from_file_location('prefetch_v3', source)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
root = Path(tempfile.mkdtemp(prefix='prefetch06-v3-proof-'))
checks = []


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def passed(name):
    checks.append(name)


def rejects(operation, kind=ValueError):
    try:
        operation()
    except kind:
        return
    raise AssertionError('unexpected acceptance')


old_record = {'bytes': 39855, 'sha256': '5284ff464fdeed0029497d95feeb9cd4365cc2d5139e5de2615a32d5e068d06e'}
assert record(old_path) == old_record
changes = [
    ('"""V2: bounded transport retry/resume', '"""V3: bounded transport retry/resume', 1),
    ('STATUS_TIMEOUT = 20.0', 'STATUS_TIMEOUT = 45.0\nPUBLICATION_STATUS_TIMEOUT = 15.0', 1),
    ("r'prefetch_(?:v2_)?", "r'prefetch_(?:v[23]_)?", 1),
    ("source in ({'bytes': 25901, 'sha256': V1_SOURCE_SHA256}, io._record(Path(__file__)))",
     "source in ({'bytes': 25901, 'sha256': V1_SOURCE_SHA256},\n                       {'bytes': 39855, 'sha256': '5284ff464fdeed0029497d95feeb9cd4365cc2d5139e5de2615a32d5e068d06e'},\n                       io._record(Path(__file__)))", 1),
    ('self.status(check_lock=False, timeout=5)', 'self.status(check_lock=False, timeout=PUBLICATION_STATUS_TIMEOUT)', 2),
    ("            'overall_seconds_per_file': TRANSFER_TIMEOUT,\n",
     "            'overall_seconds_per_file': TRANSFER_TIMEOUT,\n            'status_timeout_seconds': STATUS_TIMEOUT, 'publication_status_timeout_seconds': PUBLICATION_STATUS_TIMEOUT,\n", 1),
    ("('prefetch_v2_' +", "('prefetch_v3_' +", 1),
    ("'retry_policy': {'maximum_attempts_per_file': MAX_ATTEMPTS, 'overall_seconds_per_file': TRANSFER_TIMEOUT,\n",
     "'retry_policy': {'maximum_attempts_per_file': MAX_ATTEMPTS, 'overall_seconds_per_file': TRANSFER_TIMEOUT,\n                         'status_timeout_seconds': STATUS_TIMEOUT, 'publication_status_timeout_seconds': PUBLICATION_STATUS_TIMEOUT,\n", 1),
]
expected = old
for before, after, count in changes:
    assert expected.count(before) == count, before
    expected = expected.replace(before, after)
assert new == expected
passed('Exact source delta contains only the eight frozen replacement kinds, including exactly two publication calls')

old_ast, new_ast = ast.parse(old), ast.parse(new)
def definitions(tree):
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
od, nd = definitions(old_ast), definitions(new_ast)
assert od.keys() == nd.keys()
for name in od.keys() - {'resume_manifest', 'Prefetch', 'main'}:
    assert ast.dump(od[name]) == ast.dump(nd[name]), name
def methods(node):
    return {child.name: child for child in node.body if isinstance(child, ast.FunctionDef)}
om, nm = methods(od['Prefetch']), methods(nd['Prefetch'])
assert om.keys() == nm.keys()
for name in om.keys() - {'attempt_file'}:
    assert ast.dump(om[name]) == ast.dump(nm[name]), name
assert w.POLL_SECONDS == 5 and w.MAX_ATTEMPTS == 5 and w.TRANSFER_TIMEOUT == 3600
assert 'process.wait(timeout=5)' in new
passed('All other functions and class methods, managed groups, failure priority, deadlines, guards, hashes, locks and resume copying are AST-identical')

# Status process is stubbed locally to capture the requested/effective timeout;
# the actual Prefetch.status implementation still performs its gate-byte check.
payload = b'{"synthetic":"gate bytes"}'
gate = {'record': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()},
        'data': base64.b64encode(payload).decode()}
response = {'gates': {'preflight_huginn/COMPLETE.json': gate, 'huginn_budget_projection.json': gate}}
calls = []
active_context = None
real_run_polled = w.run_polled
real_clock = w.time.monotonic
real_lease_root = w.LEASE_ROOT
w.time.monotonic = lambda: 1000.0


def process_stub(command, stdout, stderr, guard, timeout, *, operation='transfer'):
    guard()
    calls.append({'operation': operation, 'timeout': timeout, 'context': active_context})
    if operation == 'status':
        stdout.write_text(json.dumps(response)); stderr.write_bytes(b'')
    else:
        # New-publication branch receives a complete tiny synthetic payload.
        Path(command[0]).write_bytes(b'synthetic sealed file')
        stdout.write_bytes(b''); stderr.write_bytes(b'')


w.run_polled = process_stub


class LocalStatus(w.Prefetch):
    def local_active(self, check_lock=True):
        self.remaining(); self.state = {}
    def remote_command(self, binary=None):
        return ['synthetic status only']
    def status(self, check_lock=True, timeout=w.STATUS_TIMEOUT):
        global active_context
        previous = active_context
        active_context = {'check_lock': check_lock, 'requested_timeout': timeout}
        try:
            return super().status(check_lock, timeout)
        finally:
            active_context = previous


def status_fixture(name, remaining=None):
    operation = root / name; operation.mkdir()
    runner = LocalStatus({'binaries': {}, 'metadata': {}, 'consumed': {}}, operation)
    if remaining is not None:
        runner.file_deadline = 1000 + remaining
    return runner


runner = status_fixture('default_status'); runner.status()
assert calls[-1]['timeout'] == 45 and calls[-1]['context']['requested_timeout'] == 45
runner = status_fixture('clipped_default_status', 7.25); runner.status()
assert calls[-1]['timeout'] == 7.25 and calls[-1]['context']['requested_timeout'] == 45
passed('Default status receives45 seconds and remains clipped to the original per-file remaining time')


class PublicationFixture(LocalStatus):
    def snapshot(self, relative, label):
        return {'binary': {'path': relative, 'record': self.expected['binaries'][relative], 'mtime_ns': 1700000000000000123}}
    def transfer_command(self, relative, staged):
        return [str(staged)]


relative = 'controls/frozen/state.pt'
sealed_bytes = b'synthetic sealed file'
expected_record = {'bytes': len(sealed_bytes), 'sha256': hashlib.sha256(sealed_bytes).hexdigest()}
for exists in (False, True):
    for remaining in (None, 4.25):
        label = f'publication_{exists}_{remaining}'
        lease_root = root / label; lease_root.mkdir(); w.LEASE_ROOT = lease_root
        operation = lease_root / 'operation'; operation.mkdir()
        runner = PublicationFixture({'binaries': {relative: expected_record}, 'metadata': {}, 'consumed': {}}, operation)
        runner.attempt_dir = operation / 'attempt'; runner.attempt_dir.mkdir()
        if remaining is not None:
            runner.file_deadline = 1000 + remaining
        target = lease_root / 'retrieved' / relative
        if exists:
            target.parent.mkdir(parents=True); target.write_bytes(sealed_bytes)
            os.utime(target, ns=(1700000000000000123, 1700000000000000123))
        start = len(calls)
        result = runner.attempt_file(relative, expected_record)
        publication = [call for call in calls[start:] if call['operation'] == 'status' and call['context']['check_lock'] is False]
        assert len(publication) == 1 and publication[0]['context']['requested_timeout'] == 15
        assert publication[0]['timeout'] == (15 if remaining is None else remaining)
        assert result['outcome'] == ('already_present' if exists else 'installed')
        assert record(target) == expected_record
passed('Both existing-target and new-publication branches explicitly receive15 seconds and clip it to4.25 remaining seconds')

runner = status_fixture('expired_status', -1)
before = len(calls)
rejects(runner.status, w.StopPrefetch)
assert len(calls) == before
runner = status_fixture('expired_publication', 0)
rejects(lambda: runner.status(check_lock=False, timeout=w.PUBLICATION_STATUS_TIMEOUT), w.StopPrefetch)
assert len(calls) == before
passed('Expired original deadline refuses ordinary and publication status before launching any subprocess')
w.run_polled = real_run_polled; w.time.monotonic = real_clock

# Only tiny synthetic metadata and staged bytes exercise resume compatibility.
lease_root = root / 'resume_lease'; lease_root.mkdir(); w.LEASE_ROOT = lease_root
inventory = {'binaries': {relative: {'bytes': 100, 'sha256': '0' * 64}}, 'metadata': {}, 'consumed': {}}


def prior(name, source_record):
    path = lease_root / name; path.mkdir()
    staged = path / 'staging' / relative; staged.parent.mkdir(parents=True); staged.write_bytes(b'partial')
    (path / 'PLAN.json').write_text(json.dumps({'lease_name': w.NAME, 'pod_id': w.POD,
        'required_phase': 'huginn_fit', 'inventory': inventory, 'source': source_record}))
    (path / 'INCOMPLETE.json').write_text(json.dumps({'status': 'incomplete', 'error_type': 'TransportFailure',
                                                    'stopped_utc': '2026-09-09T00:00:00Z'}))
    return path


v2 = prior('prefetch_v2_20260909T000000Z_01234567', old_record)
assert w.resume_manifest(v2, inventory)['source'] == old_record
v3 = prior('prefetch_v3_20260909T000001Z_12345678', record(source))
assert w.resume_manifest(v3, inventory)['source'] == record(source)
v1 = prior('prefetch_20260909T000002Z', {'bytes': 25901, 'sha256': w.V1_SOURCE_SHA256})
assert w.resume_manifest(v1, inventory)['source']['sha256'] == w.V1_SOURCE_SHA256
passed('Exact accepted V2 source, current V3 source and legacy V1 source are accepted with matching name variants')

bad = prior('prefetch_v2_20260909T000003Z_23456789', {**old_record, 'bytes': old_record['bytes'] - 1})
rejects(lambda: w.resume_manifest(bad, inventory))
bad = prior('prefetch_v3_20260909T000004Z_34567890', {**old_record, 'sha256': '0' * 64})
rejects(lambda: w.resume_manifest(bad, inventory))
bad = prior('prefetch_v4_20260909T000005Z_45678901', old_record)
rejects(lambda: w.resume_manifest(bad, inventory))
bad = prior('not_prefetch_v3_20260909T000006Z_56789012', record(source))
rejects(lambda: w.resume_manifest(bad, inventory))
passed('Wrong source byte count/hash and nonmatching V4 or prefixed operation names are refused')

actual_inventory = w.expected_inventory
w.expected_inventory = lambda: inventory
output = streamio.StringIO()
with contextlib.redirect_stdout(output):
    assert w.main(['describe']) == 0
description = json.loads(output.getvalue())
assert description['status_timeout_seconds'] == 45 and description['publication_status_timeout_seconds'] == 15
assert description['overall_seconds_per_file'] == 3600 and description['maximum_attempts_per_file'] == 5
assert description['execution_performed'] is False
w.expected_inventory = actual_inventory; w.LEASE_ROOT = real_lease_root
passed('Synthetic describe exposes45/15 timeout bounds while retaining max5 and3600 seconds and performing no execution')

assert record(old_path) == old_record
diff_path = root / 'SOURCE_DIFF.patch'
diff_path.write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(old_path), tofile=str(source))))
proof = {'status': 'passed_local_synthetic_fixtures_only', 'checks_count': len(checks), 'checks': checks,
         'helper': {'path': str(source), **record(source)}, 'accepted_v2': old_record,
         'proof_source': record(Path(__file__)), 'source_diff': {'path': str(diff_path), **record(diff_path)},
         'remote_connections': 0, 'actual_binary_reads': 0, 'actual_prefetch_executed': False,
         'fixture_directory': str(root)}
proof_path = root / 'PROOF.json'; proof_path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({'status': proof['status'], 'checks': len(checks), 'helper': proof['helper'], 'proof': str(proof_path)}, indent=2))
