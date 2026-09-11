from pathlib import Path
import datetime, hashlib, io, json, os, shlex, stat, subprocess, sys, tarfile, time

ROUND = Path('/home/moloch/jacobian-lens/research/refit_round_2026-09-07')
DEPLOYMENT = ROUND / 'deployment'
sys.path.insert(0, str(DEPLOYMENT))
import lease
sys.path.insert(0, str(ROUND / 'monitoring'))
import collect_ouro_handoff06 as handoff

NAME = 'jlens-refit-14ca9417eb6145c8979fb87ea9fde68c'
POD = 'vzwx0cj43g2uw5'
ARCHIVE_SHA = '3c5c93ce1bfdc6323b8d260afae0b3a85caf14af4188411759f11679ad19c2a4'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
destination = ROUND / 'monitoring/attempt_06' / ('probe_extension_' + stamp)
lease.io._no_links(destination).mkdir(exist_ok=False)
print(json.dumps({'event': 'local_operation_created', 'directory': str(destination)}), flush=True)
lease_root = lease.LEDGER / 'attempt_06'
state = lease.io._json(lease_root / 'LEASE.json')
assert state['name'] == NAME and state['pod_id'] == POD
gate = lease.io._json(ROUND / 'monitoring/attempt_06/P4_GATE.json')
assert gate['status'] == 'passed' and gate['p4_maximum_total_gpu_seconds'] == 900
previous = lease.io._json(ROUND / 'monitoring/attempt_06/ouro_handoff_20260909T110503Z/COPY_VERIFIED.json')
assert previous['status'] == 'passed' and previous['lease_name'] == NAME and previous['pod_id'] == POD
payload = ROUND / 'probe_extension'
manifest = lease.io._json(payload / 'PAYLOAD_MANIFEST.json')
proof = lease.io._json(payload / 'PREPARATION_PROOF.json')
lease.io._verify_record(payload / 'PAYLOAD_MANIFEST.json', {'bytes': proof['payload_manifest']['size'], 'sha256': proof['payload_manifest']['sha256']})
blob = (payload / 'payload.tar.gz').read_bytes()
assert len(blob) == 54178 and hashlib.sha256(blob).hexdigest() == ARCHIVE_SHA
files = {row['relative']: {'bytes': row['size'], 'sha256': row['sha256']} for row in manifest['files']}
assert set(files) == {'workflow.py', 'CONTRACT.md', 'prepared/plan.json', 'prepared/prompts.json', 'prepared/model_manifest.json', 'prepared/historical_provenance.json'}
for name, record in files.items():
    lease.io._verify_record(payload / name, record)
with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as packed:
    members = packed.getmembers()
    assert len(members) == 6 and {m.name for m in members} == set(files)
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        body = packed.extractfile(member).read()
        assert {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} == files[member.name]
plan = lease.io._json(payload / 'prepared/plan.json')
for name, sha in plan['source_pins'].items():
    assert lease.io._file_hash(Path(name)) == sha

config = {'lease_name': NAME, 'pod_id': POD, 'bindings': previous['bindings'], 'files': files,
          'archive_sha256': ARCHIVE_SHA, 'archive_bytes': 54178, 'gate': gate}
ssh = [*lease.ssh_options(lease_root, state), 'root@' + state['ssh']['host']]

COMMON = r'''
from pathlib import Path
import datetime, hashlib, json, os, signal, stat, sys, time
config = CONFIG
workspace = Path('/workspace/jlens')
p4 = workspace / 'p4'
out = workspace / 'results/probe_extension'
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def regular(path, directory=False):
    for parent in (path, *path.parents):
        if parent.is_symlink(): raise ValueError('symlink in P4 path')
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('nonregular P4 path')
def record(path):
    regular(path)
    h = hashlib.sha256(); size = 0
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024): h.update(block); size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}
def new_json(path, value):
    for parent in (path.parent, *path.parent.parents):
        if parent.is_symlink(): raise ValueError('linked receipt parent')
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
def current():
    regular(workspace / 'status.json')
    value = json.loads((workspace / 'status.json').read_text())
    assert value['lease_name'] == config['lease_name']
    assert (workspace / 'provider_lease_name').read_text().strip() == config['lease_name']
    actual_pod = os.environ.get('RUNPOD_POD_ID')
    if actual_pod is not None: assert actual_pod == config['pod_id']
    assert value['phase'] == 'awaiting_ouro_interpretation' and value.get('stop_requested', False) is False
    assert value['required_bindings'] == config['bindings']
    for name in ('STOP', 'OURO_INTERPRETATION.json'):
        assert not (workspace / name).exists() and not (workspace / name).is_symlink()
    return value
def processes():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if str(p4 / 'workflow.py').encode() in args:
            found.append({'pid': int(proc.name), 'argv': [x.decode(errors='replace') for x in args if x]})
    return found
'''.replace('CONFIG', repr(config))

STAGE = COMMON + r'''
import io, tarfile
before = current()
assert not processes()
assert not p4.exists() and not p4.is_symlink() and not out.exists() and not out.is_symlink()
for path in (workspace, workspace / 'results'): regular(path, directory=True)
assert record(workspace / 'results/ouro_evaluation/COMPLETE.json') == config['bindings']['main_complete']
assert record(workspace / 'results/controls_evaluation/COMPLETE.json') == config['bindings']['controls_complete']
deployment = workspace / 'bundle/jacobian-lens/research/refit_round_2026-09-07/deployment'
for name, key in [('run_spec.json', 'run_spec_sha256'), ('combined_contract.json', 'combined_contract_sha256')]:
    assert record(deployment / name)['sha256'] == config['bindings'][key]
blob = sys.stdin.buffer.read(config['archive_bytes'] + 1)
assert len(blob) == config['archive_bytes'] and hashlib.sha256(blob).hexdigest() == config['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as packed:
    members = packed.getmembers()
    assert len(members) == 6 and {m.name for m in members} == set(config['files'])
    bodies = {}
    for member in members:
        relative = Path(member.name)
        assert member.isfile() and not member.issym() and not member.islnk()
        assert not relative.is_absolute() and '..' not in relative.parts
        body = packed.extractfile(member).read()
        assert {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} == config['files'][member.name]
        bodies[member.name] = body
out.mkdir(); p4.mkdir()
with (out / 'payload.tar.gz').open('xb') as stream: stream.write(blob)
for name, body in bodies.items():
    target = p4 / name; target.parent.mkdir(exist_ok=True)
    with target.open('xb') as stream: stream.write(body)
staged = {name: record(p4 / name) for name in sorted(config['files'])}
assert staged == config['files']
after = current()
receipt = {'status': 'passed', 'created_utc': utc(), 'lease_name': config['lease_name'], 'pod_id': config['pod_id'],
           'status_before': before, 'status_after': after, 'payload_files': staged,
           'archive': record(out / 'payload.tar.gz'), 'gate': config['gate'],
           'scope': 'Six reviewed regular payload files staged; existing worker, watcher, models and scientific sources unchanged.'}
new_json(out / 'staging.json', receipt)
print(json.dumps({'event': 'staged', 'receipt': receipt}), flush=True)
'''
lease.io._new_json(destination / 'OPERATION.json', {'lease_name': NAME, 'pod_id': POD, 'gate': gate,
    'bindings': previous['bindings'], 'payload_files': files, 'payload_archive_sha256': ARCHIVE_SHA,
    'operator_source': lease.io._record(Path(__file__)), 'scope': 'Authorized P4 extraction and retrieval only; no interpretation receipt or CPU analysis.'})
result = subprocess.run([*ssh, 'python -c ' + shlex.quote(STAGE)], input=blob, capture_output=True, check=True, timeout=60)
stage = json.loads(result.stdout)
lease.io._new_json(destination / 'STAGED.json', stage)
print(json.dumps({'event': 'staged', 'directory': str(destination), 'remote_files': 6}), flush=True)

RUN = COMMON + r'''
import subprocess
before = current()
assert not processes()
for name, expected in config['files'].items(): assert record(p4 / name) == expected
for name in ('cache', 'deadline.json', 'supervisor.json', 'extraction.log', 'launch.json', 'outer_command.json'):
    assert not (out / name).exists() and not (out / name).is_symlink()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_ALLOC_CONF='expandable_segments:True',
    HF_HOME='/workspace/jlens/runtime_cache/huggingface', HF_HUB_DISABLE_TELEMETRY='1',
    TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
command = ['timeout', '--signal=TERM', '--kill-after=5s', '895s', '/workspace/jlens/venv/bin/python', '-B',
    str(p4 / 'workflow.py'), 'supervise-extract', '--python', '/workspace/jlens/venv/bin/python',
    '--snapshot', '/workspace/jlens/models/ouro', '--ouro-src', '/workspace/jlens/bundle/ouro_project/src',
    '--jlens-root', '/workspace/jlens/bundle/jacobian-lens', '--plan', str(p4 / 'prepared/plan.json'),
    '--output', str(out / 'cache'), '--deadline', str(out / 'deadline.json'), '--receipt', str(out / 'supervisor.json')]
started_utc = utc(); started = time.monotonic()
with (out / 'extraction.log').open('xb') as log:
    child = subprocess.Popen(command, cwd='/workspace/jlens/bundle/jacobian-lens', env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    def cancel(signum, frame):
        if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT): signal.signal(signum, cancel)
    launched = {'status': 'running', 'created_utc': started_utc, 'timeout_pid': child.pid, 'wrapper_pid': os.getpid(),
        'command': command, 'cwd': '/workspace/jlens/bundle/jacobian-lens', 'status_before': before,
        'environment': {key: env[key] for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'CUBLAS_WORKSPACE_CONFIG','PYTORCH_ALLOC_CONF','HF_HOME','HF_HUB_DISABLE_TELEMETRY',
            'TOKENIZERS_PARALLELISM','PYTHONDONTWRITEBYTECODE','PYTHONUNBUFFERED')}}
    new_json(out / 'launch.json', launched)
    print(json.dumps({'event': 'launched', **launched}), flush=True)
    returncode = child.wait()
    log.flush(); os.fsync(log.fileno())
elapsed = time.monotonic() - started
remaining = processes()
receipt = {'status': 'passed' if returncode == 0 and elapsed <= 900 and not remaining else 'incomplete',
    'started_utc': started_utc, 'finished_utc': utc(), 'elapsed_seconds': elapsed, 'returncode': returncode,
    'timeout_pid': child.pid, 'command': command, 'remaining_p4_processes': remaining, 'quiescent': not remaining,
    'status_after': current(), 'external_supervisor_receipt_present': (out / 'supervisor.json').is_file()}
new_json(out / 'outer_command.json', receipt)
print(json.dumps({'event': 'outer_finished', 'receipt': receipt}), flush=True)
assert not remaining, 'P4 process still active after timeout command ended'
'''
with (destination / 'REMOTE_RUN.log').open('xb') as captured:
    proc = subprocess.Popen([*ssh, 'python -u -c ' + shlex.quote(RUN)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline, b''):
        captured.write(line); captured.flush()
        print(line.decode(errors='replace').rstrip(), flush=True)
    code = proc.wait(timeout=940)
lease.io._new_json(destination / 'TRANSPORT.json', {'returncode': code, 'remote_run_log': lease.io._record(destination / 'REMOTE_RUN.log')})
if code != 0: print(json.dumps({'event': 'remote_transport_or_wrapper_failed', 'returncode': code}), flush=True)

SNAPSHOT = COMMON + r'''
before = current()
regular(out, directory=True)
assert not processes(), 'P4 must be quiescent before retrieval'
files = {}
for directory, dirs, names in os.walk(out, followlinks=False):
    for name in dirs: regular(Path(directory) / name, directory=True)
    for name in names:
        path = Path(directory) / name
        files[path.relative_to(out).as_posix()] = record(path)
print(json.dumps({'status_before': before, 'status_after': current(), 'files': files, 'quiescent': True}), flush=True)
'''
def snapshot():
    result = subprocess.run([*ssh, 'python -c ' + shlex.quote(SNAPSHOT)], capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)
before = snapshot()
lease.io._new_json(destination / 'REMOTE_BEFORE.json', before)
copied = destination / 'results/probe_extension'
copied.mkdir(parents=True)
with (destination / 'RSYNC.log').open('xb') as log:
    subprocess.run(['rsync', '-rlt', '--partial', '--timeout=60', '-e', shlex.join(lease.ssh_options(lease_root, state)),
        'root@' + state['ssh']['host'] + ':/workspace/jlens/results/probe_extension/', str(copied) + '/'],
        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
after = snapshot()
lease.io._new_json(destination / 'REMOTE_AFTER.json', after)
local = handoff.local_inventory(copied)
assert before['files'] == after['files'] == local
lease.io._new_json(destination / 'COPY_VERIFIED.json', {'status': 'passed', 'lease_name': NAME, 'pod_id': POD,
    'bindings': previous['bindings'], 'files': local, 'quiescent': True,
    'scope': 'Entire P4 result subtree, every regular file hashed remotely before and after retrieval and locally; no analysis.'})
print(json.dumps({'event': 'retrieved', 'directory': str(destination), 'files': len(local), 'bytes': sum(x['bytes'] for x in local.values())}), flush=True)
command = ['/home/moloch/ouro_project/venv/bin/python', '-B', str(payload / 'workflow.py'), 'validate',
    '--plan', str(payload / 'prepared/plan.json'), '--cache', str(copied / 'cache'), '--receipt', str(copied / 'supervisor.json')]
result = subprocess.run(command, capture_output=True, text=True)
lease.io._new_json(destination / 'LOCAL_VALIDATION.json', {'status': 'passed' if result.returncode == 0 else 'incomplete',
    'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
    'scope': 'Existing workflow validate only; no classifier fitting or scientific analysis.'})
print(json.dumps({'event': 'local_validation', 'directory': str(destination), 'returncode': result.returncode,
    'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
sys.exit(0 if code == 0 and result.returncode == 0 else 1)
