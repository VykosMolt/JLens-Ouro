"""Focused future-patch proof; every service and transport boundary is synthetic."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
BASE = Path('/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07/deployment/lease.py')
PATCH = Path('/tmp/lease_setup_latch.patch')
CANDIDATE = Path('/tmp/lease_setup_latch_candidate.py')
OUTPUT = Path(tempfile.mkdtemp(prefix='lease-setup-latch-proof-'))
EXPECTED_BASE = 'a7f5d92cf62f4e1d40773c6edb16c07811bb4568f2dc23018d91d9852b340992'

def record(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

assert record(BASE)['sha256'] == EXPECTED_BASE
base_before = record(BASE)
candidate_before = record(CANDIDATE)
patch_before = record(PATCH)
# Apply the actual proposed patch to an isolated copy, never the pinned source.
apply_root = OUTPUT / 'patch-application'
apply_target = apply_root / 'research/refit_round_2026-09-07/deployment/lease.py'
apply_target.parent.mkdir(parents=True)
apply_target.write_bytes(BASE.read_bytes())
application = subprocess.run(['patch', '-p1', '--batch', '--input', str(PATCH)], cwd=apply_root,
                             capture_output=True, text=True, check=True, timeout=5)
assert apply_target.read_bytes() == CANDIDATE.read_bytes()

base_tree, candidate_tree = ast.parse(BASE.read_text()), ast.parse(CANDIDATE.read_text())
base_functions = {n.name: n for n in base_tree.body if isinstance(n, ast.FunctionDef)}
candidate_functions = {n.name: n for n in candidate_tree.body if isinstance(n, ast.FunctionDef)}
assert set(candidate_functions) - set(base_functions) == {'setup_completed_for_lease', 'remember_worker_status'}
assert not (set(base_functions) - set(candidate_functions))
unchanged = []
for name, node in base_functions.items():
    if name != 'watch':
        assert ast.dump(node) == ast.dump(candidate_functions[name]), name
        unchanged.append(name)
class RestoreFrozenWatch(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'remember_worker_status':
            return ast.Call(func=ast.Name(id='update', ctx=ast.Load()), args=[ast.Name(id='root', ctx=ast.Load())],
                            keywords=[ast.keyword(arg='last_worker_status', value=ast.Name(id='status', ctx=ast.Load()))])
        if isinstance(node.func, ast.Name) and node.func.id == 'setup_completed_for_lease':
            return ast.BoolOp(op=ast.And(), values=[ast.Name(id='status', ctx=ast.Load()),
                ast.Call(func=ast.Attribute(value=ast.Name(id='status', ctx=ast.Load()), attr='get', ctx=ast.Load()),
                         args=[ast.Constant(value='setup_complete')], keywords=[])])
        return node
assert ast.dump(RestoreFrozenWatch().visit(copy.deepcopy(candidate_functions['watch']))) == ast.dump(base_functions['watch'])
assert 'Future-use correction: pre-latch controllers can terminate a completed worker' in ast.get_docstring(candidate_tree)

sys.path.insert(0, str(BASE.parent))
spec = importlib.util.spec_from_file_location('future_latched_lease', CANDIDATE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

class CyclePaused(BaseException):
    pass

class Fixture:
    def __init__(self, label):
        self.label = label
        self.directory = OUTPUT / label
        self.ledger = self.directory / 'ledger'
        self.root = self.ledger / 'attempt'
        self.root.mkdir(parents=True)
        self.clock = [1800000000.0]
        self.key = self.directory / 'synthetic-key-path'
        self.present = True
        self.calls = []
        self.ssh_value = None
        self.ssh_returncode = 255
        self.started_collectors = 0
        self.terminations = []
        bundle = self.directory / 'synthetic-bundle.tar'
        bundle.write_bytes(b'opaque synthetic bundle\n')
        created = self.clock[0] - 7205
        self.state = M.make_plan(bundle, M.stamp(created), balance=27.0, rate=.69,
                                 prior_spend=.25, now=created)
        self.state.update(status='running', mutation_phase='observed', pod_id='synthetic-pod',
                          machine_id='synthetic-machine', account_id='synthetic-account',
                          api_key_file=str(self.key), ledger=str(self.ledger),
                          ssh={'host': '127.0.0.1', 'port': 2222},
                          ssh_identity=str(self.directory / 'synthetic-no-private-key'))
        dump(self.root / 'LEASE.json', self.state)
        self.pod = {'id': self.state['pod_id'], 'name': self.state['name'],
                    'machineId': self.state['machine_id'], 'imageName': M.IMAGE,
                    'gpuCount': 1, 'containerDiskInGb': M.DISK_GB, 'volumeInGb': 0,
                    'costPerHr': .69, 'machine': {'gpuDisplayName': M.GPU},
                    'runtime': {'ports': [{'privatePort':22, 'publicPort':2222,
                                         'ip':'127.0.0.1', 'isIpPublic':True}]}}

    def persisted(self):
        return json.loads((self.root / 'LEASE.json').read_text())

    def set_status(self, setup=True, phase='complete', lease_name=None):
        self.ssh_value = {'schema_version':1, 'lease_name':lease_name or self.state['name'],
                          'phase':phase, 'setup_complete':setup, 'updated_utc':M.stamp(self.clock[0])}
        self.ssh_returncode = 0

    def missing_status(self):
        self.ssh_value = None
        self.ssh_returncode = 255

    def checked_account(self, state):
        assert state['account_id'] == self.state['account_id']
        self.calls.append('checked_account')
        return {'id':self.state['account_id'], 'clientBalance':26.0, 'currentSpendPerHr':.69 if self.present else 0.0}

    def pods(self, account):
        assert account == self.state['account_id']
        self.calls.append('pods')
        return [copy.deepcopy(self.pod)] if self.present else []

    def terminate(self, pod_id):
        assert pod_id == self.pod['id']
        self.terminations.append({'pod_id':pod_id, 'reason':self.persisted()['termination_reason']})
        self.present = False

    def run(self, command, **kwargs):
        assert command[0] == 'ssh' and command[-1] == 'test -f /workspace/jlens/status.json && cat /workspace/jlens/status.json'
        assert command[-2] == 'root@127.0.0.1'
        assert kwargs == {'capture_output':True, 'text':True, 'timeout':30}
        self.calls.append({'ssh_returncode':self.ssh_returncode})
        return subprocess.CompletedProcess(command, self.ssh_returncode,
                    json.dumps(self.ssh_value) if self.ssh_returncode == 0 else '',
                    '' if self.ssh_returncode == 0 else 'synthetic connection failure')

    def popen(self, command, **kwargs):
        assert command == [sys.executable, str(CANDIDATE), 'sync', '--root', str(self.root)]
        assert kwargs['start_new_session'] is True and kwargs['stderr'] == subprocess.STDOUT
        self.started_collectors += 1
        self.calls.append('collector_spawn')
        return SimpleNamespace(pid=-1)

    def sleep(self, seconds):
        self.clock[0] += seconds
        if seconds == 15:
            raise CyclePaused()
        assert seconds == 5, seconds

    @contextlib.contextmanager
    def bound(self):
        with contextlib.ExitStack() as stack:
            for target, name, value in [(M,'LEDGER',self.ledger), (M.api,'KEY_FILE',self.key),
                     (M,'checked_account',self.checked_account), (M,'pods',self.pods),
                     (M.api,'terminate',self.terminate), (M.time,'time',lambda:self.clock[0]),
                     (M.time,'sleep',self.sleep), (M.subprocess,'run',self.run),
                     (M.subprocess,'Popen',self.popen)]:
                stack.enter_context(mock.patch.object(target,name,value))
            yield

    def cycle(self):
        # Actual candidate watch, remote_status, state transactions, latch,
        # ensure_sync, halt, terminate_owned and confirm_absence all execute.
        with self.bound():
            try:
                M.watch(SimpleNamespace(root=self.root))
            except CyclePaused:
                return 'paused_after_cycle'
        return 'returned'

    def latch(self):
        return {'lease_name':self.state['name'], 'pod_id':self.state['pod_id']}

    def summary(self):
        s=self.persisted()
        return {'case':self.label, 'status':s['status'], 'setup_completion':s.get('setup_completion'),
                'termination_reason':s.get('termination_reason'), 'termination_calls':self.terminations,
                'collector_spawns':self.started_collectors,
                'retrieval_marker_exists':(self.root/'RETRIEVAL_VERIFIED.json').exists(),
                'watch_last_error_type':s.get('watch_last_error_type'),
                'absence_confirmations':s.get('absence_confirmations'),
                'calls':self.calls}

CASES=[]
f=Fixture('success_restart_missing_then_verified')
f.set_status(True)
assert f.cycle() == 'paused_after_cycle'
first=f.persisted()
assert first['setup_completion'] == f.latch() and first['last_worker_status']['setup_complete'] is True
assert not f.terminations and f.started_collectors == 1
assert not (f.root/'RETRIEVAL_VERIFIED.json').exists()
# New watch invocation has no in-memory latch; the state must survive restart.
f.missing_status()
assert f.cycle() == 'paused_after_cycle'
missing=f.persisted()
assert missing['status']=='running' and missing['setup_completion']==first['setup_completion']
assert missing['last_worker_status']==first['last_worker_status']
assert not f.terminations and not (f.root/'RETRIEVAL_VERIFIED.json').exists()
# A later real status with false setup cannot erase the first validated success.
f.set_status(False, phase='failed')
assert f.cycle() == 'paused_after_cycle'
assert f.persisted()['setup_completion']==first['setup_completion'] and not f.terminations
assert f.persisted()['last_worker_status']['setup_complete'] is False
f.missing_status()
dump(f.root/'RETRIEVAL_VERIFIED.json',{'status':'passed','lease_name':f.state['name']})
assert f.cycle() == 'returned'
last=f.persisted()
assert last['status']=='terminated' and last['termination_reason']=='artifacts retrieved and verified'
assert last['setup_completion']==first['setup_completion'] and last['absence_confirmations']==3
assert len(f.terminations)==1
CASES.append(f.summary())

for kind in ('never_setup', 'mismatched_lease_latch', 'mismatched_pod_latch'):
    f=Fixture(kind)
    if kind!='never_setup':
        s=f.persisted(); s['setup_completion']=f.latch()
        s['setup_completion']['lease_name' if kind=='mismatched_lease_latch' else 'pod_id']='wrong-binding'
        dump(f.root/'LEASE.json',s)
    assert f.cycle()=='returned'
    s=f.persisted()
    assert s['status']=='terminated' and s['termination_reason']=='setup did not finish within the bounded window'
    assert len(f.terminations)==1 and not (f.root/'RETRIEVAL_VERIFIED.json').exists()
    if kind=='never_setup': assert 'setup_completion' not in s
    else: assert not M.setup_completed_for_lease(s)
    CASES.append(f.summary())

f=Fixture('unvalidated_status_cannot_latch')
f.set_status(True,lease_name='another-lease')
assert f.cycle()=='paused_after_cycle'
s=f.persisted()
assert 'setup_completion' not in s and 'last_worker_status' not in s
assert s['watch_last_error_type']=='ValueError' and s['watch_consecutive_failures']==1
assert not f.terminations
CASES.append(f.summary())

f=Fixture('truthy_non_boolean_cannot_latch')
f.set_status(1)
assert f.cycle()=='returned'
s=f.persisted()
assert 'setup_completion' not in s and s['termination_reason']=='setup did not finish within the bounded window'
CASES.append(f.summary())

f=Fixture('stale_observer_identity_cannot_persist')
f.set_status(True)
for key in ('name','pod_id'):
    observed=copy.deepcopy(f.persisted());observed[key]='different-observer-binding'
    prior=(f.root/'LEASE.json').read_bytes()
    with f.bound():
        try:M.remember_worker_status(f.root,observed,f.ssh_value)
        except ValueError:pass
        else:raise AssertionError('stale observer accepted')
    assert (f.root/'LEASE.json').read_bytes()==prior
assert 'setup_completion' not in f.persisted()
CASES.append(f.summary())

assert record(BASE)==base_before and record(CANDIDATE)==candidate_before and record(PATCH)==patch_before
receipt={'status':'passed','base':{'path':str(BASE),'record':base_before},
         'candidate':{'path':str(CANDIDATE),'record':candidate_before},
         'patch':{'path':str(PATCH),'record':patch_before},
         'proof_source':{'path':str(Path(__file__)),'record':record(Path(__file__))},
         'actual_patch_application':{'directory':str(apply_root),'returncode':application.returncode,
                                     'matches_candidate_byte_exactly':True},
         'static_scope':{'unchanged_functions':unchanged,'watch_changes_only_two_frozen_call_sites':True,
                         'new_functions':['setup_completed_for_lease','remember_worker_status'],
                         'future_use_note_present':True},
         'cases':CASES,
         'execution_scope':'Actual candidate watcher loop, remote_status, state persistence, async-collector dispatch, termination and absence-confirmation logic; only account/provider/SSH/process-spawn/time boundaries are synthetic. Real temporary files and locks are used. No cloud, SSH, provider mutation, pinned source edit, broad self-test, or actual experiment outcome read.'}
dump(OUTPUT/'PROOF.json',receipt)
print(json.dumps({'proof':str(OUTPUT/'PROOF.json'),'cases':len(CASES),'candidate':candidate_before,
                  'patch':patch_before},indent=2))
