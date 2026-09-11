"""One supervised RunPod lease within the user's combined $25 ceiling.

Plan and self-test are offline. Create requires a recent notice timestamp,
checks live funding/price, writes its intent before the mutation, and installs
a detached systemd watcher first. The provider also receives a fixed deletion
deadline. An uncertain create is reconciled by the unique name, never retried.
Credentials stay in the local account client and are never sent to the GPU.

Future-use correction: pre-latch controllers can terminate a completed worker
when a later status read is missing after the setup deadline. Apply this
revision only before a new lease; preserve existing pinned controllers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid

import run_refits as io
import runpod_api as api
import artifact_handoff as handoff

HERE = Path(__file__).resolve().parent
IMAGE = "python@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f"
GPU = "NVIDIA GeForce RTX 5090"
MAX_GPU_RATE = 0.69
DISK_GB = 120
STORAGE_RATE = DISK_GB * 0.10 / 730.0
BUDGET = 25.0
# Allow bounded time for both account-identity reads before watcher readiness.
WATCHER_STARTUP_SECONDS = 180
LEDGER = Path("/home/moloch/jacobian-lens/research/confirmation_2026-09-09/cloud_leases")
TERMINATION_RESERVE_SECONDS = 600
CONTROLLER_FILES = tuple(sorted(handoff.SOURCE_NAMES))
# Every new deadline includes the selected GPU's full permitted price and disk.
HARD_RATE = MAX_GPU_RATE + STORAGE_RATE
REMOTE = "/workspace/jlens"
POD_FIELDS = """id name machineId costPerHr desiredStatus imageName gpuCount
 containerDiskInGb volumeInGb createdAt machine { gpuDisplayName }
 runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } }"""
SUPPLY_MESSAGE = ("There are no longer any instances available with the requested specifications. "
                  "Please refresh and try again.")


def stamp(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(value):
    return io._deadline(value).timestamp()


def number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid {name}")
    return float(value)


def account(*, write_key=False):
    value = api.gql("{ myself { id clientBalance currentSpendPerHr } }", write=write_key)["myself"]
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("account identity is missing")
    number(value["clientBalance"], "balance")
    number(value["currentSpendPerHr"], "current rate")
    return value


def checked_account(state):
    live, writable = account(), account(write_key=True)
    if live["id"] != state["account_id"] or writable["id"] != state["account_id"]:
        raise ValueError("lease credentials no longer identify the expected account")
    return live


def pod_identity(pod):
    if (not isinstance(pod, dict) or not isinstance(pod.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", pod["id"])
            or not isinstance(pod.get("name"), str) or not pod["name"]):
        raise ValueError("invalid pod identity")
    return pod["id"]


def pods(expected_account=None):
    myself = api.gql("{ myself { id pods { " + POD_FIELDS + " } } }")["myself"]
    if (not isinstance(myself, dict) or not isinstance(myself.get("id"), str) or not myself["id"]
            or (expected_account is not None and myself["id"] != expected_account)):
        raise ValueError("pod listing does not identify the expected account")
    value = myself["pods"]
    if not isinstance(value, list):
        raise ValueError("invalid pod listing")
    identifiers = [pod_identity(row) for row in value]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate pod identifiers in listing")
    return value


def offer():
    query = """query Quote($id: String!) { gpuTypes(input:{id:$id}) {
      id memoryInGb lowestPrice(input:{gpuCount:1,secureCloud:false,totalDisk:120,
        minMemoryInGb:64,minVcpuCount:8,supportPublicIp:true}) {
        stockStatus uninterruptablePrice minMemory minVcpu } } }"""
    values = api.gql(query, {"id": GPU})["gpuTypes"]
    if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict)
            or values[0].get("id") != GPU
            or number(values[0].get("memoryInGb"), "GPU memory") != 32):
        raise ValueError("requested GPU offer is missing")
    result = values[0].get("lowestPrice")
    if not isinstance(result, dict):
        raise ValueError("requested GPU price is missing")
    if str(result.get("stockStatus") or "").lower() not in ("low", "medium", "high"):
        raise ValueError("requested GPU is not available")
    if number(result["uninterruptablePrice"], "GPU rate", minimum=0.001) > MAX_GPU_RATE:
        raise ValueError("live GPU rate exceeds the notified limit")
    number(result.get("minMemory"), "quoted host RAM", minimum=64)
    number(result.get("minVcpu"), "quoted CPU count", minimum=8)
    return {**result, "recorded_utc": stamp(), "gpu_type_id": GPU}


@contextmanager
def locked(root, name):
    io._mkdir(root)
    with io._no_links(root / name).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def change(root, action):
    with locked(root, ".state.lock"):
        state = io._json(root / "LEASE.json")
        previous_rate = number(state["all_in_rate"], "previous all-in rate", minimum=0.001)
        lease_floor = number(state["max_gpu_rate"], "lease GPU price ceiling", minimum=0.001) + STORAGE_RATE
        immutable = {key: json.loads(io._canonical(state[key])) for key in (
            'name', 'run_id', 'account_id', 'run_spec_path', 'run_spec_record', 'run_spec_sha256',
            'output_contract_path', 'output_contract_record', 'output_contract_sha256',
            'run_config_path', 'run_config_record', 'semantic_verifier', 'controller_sources',
            'prior_debit_path', 'prior_debit_record', 'prior_spend_upper_usd', 'job_incremental_cap_usd',
            'billing_start_utc', 'max_gpu_rate', 'preserve_balance') if key in state}
        owned = {key: state[key] for key in ('pod_id', 'machine_id') if state.get(key) is not None}
        deadlines = {key: state[key] for key in ('provider_deadline_utc', 'watch_deadline_utc',
                                               'work_deadline_utc', 'setup_deadline_utc') if key in state}
        action(state)
        if any(state.get(key) != value for key, value in {**immutable, **owned}.items()):
            raise ValueError('durable lease/run identity or accounting cannot change')
        if any(epoch(state[key]) > epoch(value) for key, value in deadlines.items()):
            raise ValueError('fixed lease deadlines cannot be extended')
        state["all_in_rate"] = max(previous_rate, lease_floor, number(state["all_in_rate"], "all-in rate", minimum=0.001))
        io._atomic_json(root / "LEASE.json", state)
        return state


def update(root, **fields):
    return change(root, lambda state: state.update(fields))


def lease_root(value):
    root = io._no_links(value)
    if root.parent != LEDGER or root.name.startswith("."):
        raise ValueError("lease roots must be direct children of the canonical experiment ledger")
    return root


def load_run_config(path):
    path = io._no_links(path)
    config = io._json(path)
    expected = {'schema', 'run_id', 'run_spec_path', 'output_contract_path', 'semantic_verifier',
                'setup_budget_seconds', 'compute_budget_seconds', 'preservation_reserve_seconds', 'transfer_timeout_seconds'}
    if (not isinstance(config, dict) or set(config) != expected
            or config['schema'] != 'confirmation_run_config.v1'
            or not isinstance(config['run_id'], str) or not handoff.SAFE_ID.fullmatch(config['run_id'])):
        raise ValueError('invalid confirmation run config')
    fields = dict(config)
    for key in ('run_spec_path', 'output_contract_path'):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError('run configuration paths must be absolute')
        record = handoff.checked_record(config[key])
        prefix = key.removesuffix('_path')
        fields[prefix + '_record'] = record
        fields[prefix + '_sha256'] = record['sha256']
    for key in ('setup_budget_seconds', 'compute_budget_seconds', 'preservation_reserve_seconds', 'transfer_timeout_seconds'):
        fields[key] = number(config[key], key, minimum=1)
    if (fields['preservation_reserve_seconds'] < 600
            or fields['transfer_timeout_seconds'] > fields['preservation_reserve_seconds']):
        raise ValueError('preservation reserve does not cover bounded transfer')
    verifier = config['semantic_verifier']
    if (not isinstance(verifier, dict) or set(verifier) != {'path', 'record', 'callable'}
            or verifier['callable'] != 'validate_outputs' or not Path(verifier['path']).is_absolute()):
        raise ValueError('invalid pinned local semantic verifier')
    handoff.verify_record(verifier['path'], verifier['record'])
    fields.update(run_config_path=str(path), run_config_record=handoff.checked_record(path))
    handoff.load_contract(fields)
    return fields


def prior_debit(path):
    path = io._no_links(path)
    record, value = handoff.checked_record(path), io._json(path)
    if (not isinstance(value, dict) or value.get('schema') != 'confirmation_prior_debit.v1'
            or value.get('combined_cap_usd') != BUDGET
            or not isinstance(value.get('account_id'), str) or not value['account_id']):
        raise ValueError('invalid prior debit snapshot')
    base = number(value.get('prior_spend_upper_usd'), 'prior debit')
    if (base >= BUDGET or not math.isclose(number(value.get('remaining_authorized_upper_usd'), 'remaining budget'),
                                          BUDGET - base, rel_tol=0, abs_tol=1e-9)):
        raise ValueError('prior debit remaining budget differs')
    previous, historical_names = [], set()
    for row in value.get('historical_leases', []):
        if row['name'] in historical_names:
            raise ValueError('duplicate historical lease debit')
        historical_names.add(row['name'])
        pin = row['record']
        handoff.verify_record(pin['path'], {'bytes': pin['bytes'], 'sha256': pin['sha256']})
        old = io._json(pin['path'])
        if (old.get('name') != row['name'] or old.get('status') != 'terminated'
                or old.get('pod_id') != row.get('pod_id')
                or old.get('lease_spend_upper_usd') != row['lease_spend_upper_usd']):
            raise ValueError('historical lease is not reconciled to the debit')
    if (not historical_names or base + 1e-9 < sum(number(row['lease_spend_upper_usd'], 'historical lease debit')
                                                for row in value['historical_leases'])
            or base + 1e-9 < max(number(row['combined_spend_upper_usd'], 'historical combined debit')
                                 for row in value['historical_leases'])):
        raise ValueError('prior debit omits historical spending')
    handoff.verify_record(path, record)
    names = set(historical_names)
    for path_new in sorted(LEDGER.glob('*/LEASE.json')):
        state = io._json(io._no_links(path_new))
        if state['name'] in names or state.get('status') != 'terminated':
            raise ValueError('duplicate or unreconciled new lease debit')
        if state.get('prior_debit_record') != record or state.get('account_id') != value['account_id']:
            raise ValueError('new lease has another prior debit/account')
        names.add(state['name'])
        previous.append({'name': state['name'], 'record': handoff.checked_record(path_new),
                         'lease_spend_upper_usd': number(state['lease_spend_upper_usd'], 'new lease debit')})
    total = base + sum(row['lease_spend_upper_usd'] for row in previous)
    return value, record, total, previous


def make_plan(bundle, notice, *, balance, rate, run_config, debit_path, job_cap, now=None):
    now = time.time() if now is None else now
    config = load_run_config(run_config)
    debit, debit_record, prior_spend, previous = prior_debit(debit_path)
    rate = number(rate, "GPU rate", minimum=0.001)
    balance = number(balance, "available funds")
    job_cap = number(job_cap, 'incremental job cap', minimum=0.01)
    if job_cap > number(debit['initial_ouro_job_incremental_cap_usd'], 'prior job ceiling'):
        raise ValueError('job cap exceeds the frozen initial ceiling')
    if rate > MAX_GPU_RATE or prior_spend >= BUDGET:
        raise ValueError("rate or remaining combined budget is invalid")
    remaining = BUDGET - prior_spend
    if balance < job_cap + number(debit['minimum_balance_to_preserve_usd'], 'preserved balance'):
        raise ValueError("the account lacks the job funds above its preserved balance")
    # Reserve $0.15 beyond the provider deadline for polling/billing latency.
    # The provider deadline remains safe when the assigned price exceeds the quote.
    all_in = HARD_RATE
    seconds = math.ceil(config['setup_budget_seconds'] + config['compute_budget_seconds']
                        + config['preservation_reserve_seconds'] + TERMINATION_RESERVE_SECONDS)
    if seconds / 3600 * all_in + 0.15 > min(remaining, job_cap):
        raise ValueError('declared computation and preservation do not fit the combined/job budget')
    return {
        **config, "schema_version": 2, "name": "jlens-confirm-" + uuid.uuid4().hex,
        "image": IMAGE, "gpu_type_id": GPU, "cloud": "COMMUNITY", "gpu_count": 1,
        "container_disk_gb": DISK_GB, "volume_gb": 0, "max_gpu_rate": MAX_GPU_RATE,
        "quoted_gpu_rate": rate, "storage_rate": STORAGE_RATE, "all_in_rate": all_in,
        "combined_cap_usd": BUDGET, "prior_spend_upper_usd": prior_spend,
        "starting_balance": balance, "preserve_balance": max(debit['minimum_balance_to_preserve_usd'], balance - job_cap),
        "billing_start_utc": stamp(now), "provider_deadline_utc": stamp(now + seconds),
        "watch_deadline_utc": stamp(now + seconds - TERMINATION_RESERVE_SECONDS),
        "work_deadline_utc": stamp(now + config['setup_budget_seconds'] + config['compute_budget_seconds']),
        "setup_deadline_utc": stamp(now + config['setup_budget_seconds']),
        "job_incremental_cap_usd": job_cap, "prior_debit_path": str(io._no_links(debit_path)),
        "prior_debit_record": debit_record, "new_lease_debits": previous,
        "account_id": debit['account_id'], "known_manifests": {}, "stage_acceptances": {},
        "user_notice_utc": notice, "bundle": str(bundle.absolute()), "bundle_record": io._record(bundle),
        "status": "planned", "mutation_phase": "not_started", "halt_requested": False,
        "pod_id": None, "machine_id": None, "ledger": str(LEDGER),
    }


def validate_notice(value, now=None):
    age = (time.time() if now is None else now) - epoch(value)
    if not 0 <= age <= 300:
        raise ValueError("creation requires the user's notification within the preceding five minutes")


def bind_pod(state, pod):
    pod_identity(pod)
    selected_gpu = state.get("gpu_type_id")
    if selected_gpu not in ("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"):
        raise ValueError("lease GPU is outside the explicitly supported selections")
    gpu_aliases = (selected_gpu, selected_gpu.removeprefix("NVIDIA GeForce "))
    if (pod.get("name") != state["name"]
            or not isinstance(pod.get("machineId"), str) or not pod["machineId"]
            or (state.get("pod_id") is not None and pod["id"] != state["pod_id"])
            or (state.get("machine_id") is not None and pod.get("machineId") != state["machine_id"])):
        raise ValueError("pod is not the uniquely owned lease")
    if (pod.get("imageName") != state["image"] or type(pod.get("gpuCount")) is not int or pod["gpuCount"] != 1
            or type(pod.get("containerDiskInGb")) is not int or pod["containerDiskInGb"] != DISK_GB
            or type(pod.get("volumeInGb")) is not int or pod["volumeInGb"] != 0
            or not isinstance(pod.get("machine"), dict)
            or pod["machine"].get("gpuDisplayName") not in gpu_aliases):
        raise ValueError("assigned pod geometry/image differs from the lease")
    rate = number(pod.get("costPerHr"), "assigned GPU rate", minimum=0.001)
    maximum_rate = number(state["max_gpu_rate"], "lease GPU price ceiling", minimum=0.001)
    if rate > maximum_rate:
        raise ValueError("assigned GPU price exceeds the notice and budget")
    return {"pod_id": pod["id"], "machine_id": pod.get("machineId"),
            "actual_gpu_rate": rate, "all_in_rate": max(
                number(state["all_in_rate"], "previous all-in rate", minimum=0.001),
                maximum_rate + STORAGE_RATE, rate + STORAGE_RATE)}


def endpoint(pod):
    for port in (pod.get("runtime") or {}).get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic") is True:
            ipaddress.ip_address(port["ip"])
            if type(port.get("publicPort")) is int and 1 <= port["publicPort"] <= 65535:
                return {"host": port["ip"], "port": port["publicPort"]}
    return None


def ssh_options(root, state):
    remote = state["ssh"]
    ipaddress.ip_address(remote["host"])
    return ["ssh", "-i", state["ssh_identity"], "-p", str(remote["port"]),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={root / 'known_hosts'}"]


def remote_status(root, state):
    if not state.get("ssh"):
        return None
    result = subprocess.run([*ssh_options(root, state), "root@" + state["ssh"]["host"],
                             f"test -f {REMOTE}/status.json && cat {REMOTE}/status.json"],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        return None
    value = json.loads(result.stdout)
    if value.get("lease_name") != state["name"]:
        raise ValueError("remote status belongs to another run")
    return value


def setup_completed_for_lease(state):
    try:
        marker = state.get('setup_completion')
        return (isinstance(marker, dict) and set(marker) == {'binding', 'observed_utc'}
                and marker['binding'] == handoff.binding(state))
    except (ValueError, KeyError, TypeError):
        return False


def same_run(current, observed):
    if handoff.binding(current) != handoff.binding(observed):
        raise ValueError('lease/pod/run changed before persistence')


def remember_worker_status(root, observed_state, status):
    def apply(state):
        same_run(state, observed_state)
        if (not isinstance(status, dict) or status.get('lease_name') != state['name']
                or status.get('binding') != handoff.binding(state)
                or type(status.get('setup_complete')) is not bool):
            raise ValueError('worker status lacks the exact current identity')
        state['last_worker_status'] = status
        if status['setup_complete'] and not setup_completed_for_lease(state):
            state['setup_completion'] = {'binding': handoff.binding(state), 'observed_utc': stamp()}
    return change(root, apply)


def remember_manifest(root, observed_state, manifest, ref):
    handoff.validate_manifest(observed_state, manifest)
    manifest_sha = handoff.digest(manifest)
    def apply(state):
        same_run(state, observed_state)
        known = state.setdefault('known_manifests', {})
        for previous in known.values():
            other = previous['manifest']
            if other['kind'] == 'final' and manifest['kind'] == 'final' and other != manifest:
                raise ValueError('terminal manifest changed')
            if previous['ref']['path'] == ref['path'] and previous['ref'] != ref:
                raise ValueError('immutable manifest path changed')
            for name in set(other['files']) & set(manifest['files']):
                if other['files'][name] != manifest['files'][name]:
                    raise ValueError('committed artifact changed across stages')
        known[manifest_sha] = {'manifest': manifest, 'ref': ref}
        if manifest['kind'] == 'final':
            terminal = {'binding': handoff.binding(state), 'outcome': manifest['outcome'],
                        'manifest_sha256': manifest_sha, 'manifest': manifest, 'observed_utc': stamp()}
            previous = state.get('computation_finished')
            if previous and previous['manifest_sha256'] != manifest_sha:
                raise ValueError('latched terminal manifest cannot be replaced')
            state.setdefault('computation_finished', terminal)
    return change(root, apply)


def request_worker_stop(root, state):
    if state.get('ssh'):
        from transport import run_bounded
        run_bounded([*ssh_options(root, state), 'root@' + state['ssh']['host'],
                     'touch ' + REMOTE + '/STOP'], timeout=30, operation='worker_stop')
    update(root, stop_requested_utc=stamp())


def local_inventory(root):
    return handoff.payload_inventory(root)


def sync_results(root, *, transport=None):
    with io._no_links(root / '.sync.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        state = io._json(root / 'LEASE.json')
        handoff.verify_local_sources(state)
        transport = transport or handoff.SSHTransport(root, state, ssh_options(root, state))
        try:
            manifests = handoff.read_index(state, transport)
            if manifests:
                if not set(state.get('known_manifests', {})) <= {handoff.digest(m) for m in manifests}:
                    raise ValueError('artifact index removed a committed manifest')
                for manifest in manifests:
                    state = remember_manifest(root, state, manifest, transport.manifest_refs[handoff.digest(manifest)])
            for sha, entry in state.get('known_manifests', {}).items():
                transport.manifest_refs[sha] = entry['ref']
            results = []
            entries = sorted(state.get('known_manifests', {}).values(), key=lambda e: e['manifest']['kind'] == 'final')
            for entry in entries:
                manifest = entry['manifest']
                def progress(**fields):
                    def apply(current):
                        same_run(current, state)
                        current['retrieval_in_progress'] = {'binding': handoff.binding(current),
                                                            'updated_utc': stamp(), **fields}
                    change(root, apply)
                pointer = handoff.collect_manifest(root, state, manifest, transport, progress=progress)
                if pointer is not None:
                    def accept(current):
                        same_run(current, state)
                        if manifest['kind'] == 'final':
                            if current.get('computation_finished', {}).get('manifest_sha256') != handoff.digest(manifest):
                                raise ValueError('acceptance final manifest changed')
                            current['retrieval_accepted'] = pointer
                        else:
                            current.setdefault('stage_acceptances', {})[handoff.digest(manifest)] = pointer
                        current['retrieval_in_progress'] = {'binding': handoff.binding(current),
                            'manifest_sha256': handoff.digest(manifest), 'phase': 'accepted', 'updated_utc': stamp()}
                    state = change(root, accept)
                    results.append(pointer)
            return results
        except BaseException as error:
            def failed(current):
                same_run(current, state)
                current['retrieval_last_error'] = {'type': type(error).__name__, 'message': str(error), 'utc': stamp()}
            change(root, failed)
            raise


def ensure_sync(root, state):
    if not state.get('ssh') or not state.get('pod_id'):
        return
    # The collector itself validates receipts. Never suppress it because a marker exists.
    with io._no_links(root / '.sync.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    with (root / 'retrieval.log').open('ab') as log:
        subprocess.Popen([sys.executable, str(Path(__file__)), 'sync', '--root', str(root)],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def owned_candidates(state, listing):
    if not isinstance(listing, list):
        raise ValueError("invalid pod listing")
    for pod in listing:
        pod_identity(pod)
    candidates = [pod for pod in listing if pod.get("name") == state["name"]
                  or (state.get("pod_id") and pod.get("id") == state["pod_id"])]
    if len(candidates) > 1:
        raise ValueError("multiple resources match the owned lease")
    if candidates and candidates[0].get("name") != state["name"]:
        raise ValueError("owned pod was renamed; cannot report it absent")
    return candidates


def observe_cost(root, state, pod):
    """Account for an assigned rate even when the subsequent price gate fails."""
    if isinstance(pod, dict) and pod.get("name") == state["name"]:
        rate = pod.get("costPerHr")
        if not isinstance(rate, bool) and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
            return update(root, all_in_rate=rate + STORAGE_RATE)
    return state


def observe_identity(root, pod):
    """Resolve the single deployment before checking price/geometry, so it can be deleted."""
    pod_identity(pod)
    def apply(state):
        if (pod["name"] != state["name"]
                or (state.get("pod_id") is not None and pod["id"] != state["pod_id"])
                or not isinstance(pod.get("machineId"), str) or not pod["machineId"]
                or (state.get("machine_id") is not None and pod["machineId"] != state["machine_id"])):
            raise ValueError("pod identity changed during the lease")
        state.update(pod_id=pod["id"], machine_id=pod["machineId"], mutation_phase="observed")
    return change(root, apply)


def begin_create(root):
    def apply(state):
        handoff.verify_record(state['prior_debit_path'], state['prior_debit_record'])
        handoff.verify_record(state['run_config_path'], state['run_config_record'])
        handoff.verify_local_sources(state)
        handoff.load_contract(state)
        if (state["status"] != "pending" or state["mutation_phase"] != "not_started"
                or state.get("halt_requested") is not False
                or not state.get("watch_ready_utc")
                or state.get("watch_account_id") != state["account_id"]
                or not 0 <= time.time() - epoch(state["watch_ready_utc"]) <= 30
                or time.time() >= epoch(state["setup_deadline_utc"])):
            raise ValueError("creation was halted or lacks a current verified watcher")
        state.update(status="creating", mutation_phase="in_flight", mutation_started_utc=stamp())
    return change(root, apply)


def finish_create(root, fields):
    def apply(state):
        state.update(fields, deployment_response_utc=stamp())
        if not state.get("halt_requested") and state["status"] == "creating":
            state["status"] = "running"
    return change(root, apply)


def fail_create(root):
    def apply(state):
        state["halt_requested"] = True
        if state["mutation_phase"] == "in_flight":
            state["mutation_phase"] = "uncertain"
        if state["status"] not in ("terminated", "terminating"):
            state["status"] = "create_rejected" if state["mutation_phase"] == "rejected" else "create_uncertain"
    return change(root, apply)


def exact_supply_rejection(body, account_id):
    """Recognize only the observed provider allocation refusal, never null alone."""
    if (not isinstance(body, dict) or set(body) != {"errors", "data"}
            or body["data"] != {"podFindAndDeployOnDemand": None}
            or not isinstance(body["errors"], list) or len(body["errors"]) != 1):
        return False
    error = body["errors"][0]
    return (isinstance(error, dict) and error.get("path") == ["podFindAndDeployOnDemand"]
            and error.get("message") == SUPPLY_MESSAGE
            and isinstance(error.get("extensions"), dict)
            and error["extensions"].get("code") == "SUPPLY_CONSTRAINT"
            and isinstance(account_id, str) and bool(account_id)
            and error["extensions"].get("userId") == account_id)


def completed_supply_rejection(error, account_id):
    return (isinstance(error, api.APIError) and error.response_completed is True
            and type(error.status) is int and 200 <= error.status < 300
            and exact_supply_rejection(error.graphql_body, account_id))


def record_rejection(root, body, provenance):
    """Called with the mutation lock held; atomically seal evidence and halt."""
    body = json.loads(json.dumps(body, allow_nan=False))
    proof = {"schema_version": 1, "code": "SUPPLY_CONSTRAINT", "recorded_utc": stamp(),
             "response_body": body, "canonical_json_sha256": io._digest(body),
             "hash_scope": "canonical decoded JSON, not raw HTTP-response bytes", "provenance": provenance}
    def apply(state):
        if (state.get("mutation_phase") not in ("in_flight", "uncertain")
                or state.get("pod_id") is not None or state.get("machine_id") is not None
                or state.get("deployment_response_utc") is not None
                or not exact_supply_rejection(body, state["account_id"])):
            raise ValueError("supply rejection cannot replace an observed or unstarted deployment")
        state.update(mutation_phase="rejected", halt_requested=True, create_rejection=proof)
        if state["status"] not in ("terminating", "terminated"):
            state["status"] = "create_rejected"
    return change(root, apply)


def verified_rejection(state):
    proof = state.get("create_rejection")
    return (state.get("pod_id") is None and state.get("machine_id") is None
            and state.get("deployment_response_utc") is None
            and isinstance(proof, dict) and proof.get("schema_version") == 1
            and proof.get("code") == "SUPPLY_CONSTRAINT"
            and exact_supply_rejection(proof.get("response_body"), state.get("account_id"))
            and proof.get("canonical_json_sha256") == io._digest(proof["response_body"]))


def reconcile_rejection(args):
    """Register a complete historical exception capture, then verify absence.

    The capture's provenance must state that the single create process finished;
    it is normalized JSON evidence and makes no claim about original HTTP bytes.
    """
    root = lease_root(args.root)
    capture_path = io._no_links(args.capture)
    capture = io._json(capture_path)
    if (not isinstance(capture, dict) or capture.get("schema_version") != 1
            or type(capture.get("deployment_call_count")) is not int or capture["deployment_call_count"] != 1
            or type(capture.get("process_exit_code")) is not int or capture["process_exit_code"] == 0
            or type(capture.get("create_exec_session_id")) is not int or capture["create_exec_session_id"] <= 0
            or not isinstance(capture.get("provenance"), str) or not capture["provenance"].strip()):
        raise ValueError("historical rejection needs a complete one-call process capture and provenance")
    with locked(LEDGER, ".lease_creation.lock"):
        with locked(root, ".mutation.lock"):
            state = io._json(root / "LEASE.json")
            if (state.get("mutation_phase") != "uncertain" or state.get("halt_requested") is not True
                    or str(io._no_links(api.KEY_FILE)) != state["api_key_file"]
                    or not exact_supply_rejection(capture.get("response_body"), state["account_id"])):
                raise ValueError("historical supply capture does not identify the halted lease and credential path")
            checked_account(state)
            provenance = {"kind": "normalized_capture_from_completed_create", "description": capture["provenance"],
                          "capture_path": str(capture_path), "capture_record": io._record(capture_path),
                          "create_exec_session_id": capture["create_exec_session_id"],
                          "process_exit_code": capture["process_exit_code"], "deployment_call_count": 1,
                          "reconciliation_sources": {name: io._record(HERE / name) for name in ("lease.py", "runpod_api.py")}}
            record_rejection(root, capture["response_body"], provenance)
        if not reconcile_absence(root, "completed provider supply rejection; verify no resource exists"):
            raise RuntimeError("supply rejection registered but fresh account-verified absence remains unresolved")
    return io._json(root / "LEASE.json")


def halt(root, reason):
    def apply(state):
        state.update(halt_requested=True, termination_reason=reason)
        if state["status"] != "terminated":
            state["status"] = "terminating"
    return change(root, apply)


def absence_can_finish(state):
    phase = state.get("mutation_phase")
    if not state.get("halt_requested"):
        return False
    if phase == "not_started":
        return True  # The durable halt barrier prevents a later deployment mutation.
    if phase == "rejected":
        return verified_rejection(state)
    if phase == "observed" and isinstance(state.get("pod_id"), str) and state["pod_id"]:
        return True
    return phase in ("in_flight", "uncertain") and time.time() >= epoch(state["provider_deadline_utc"])


def confirm_absence(root, absent, account_id):
    # A suspended creator must not resume its mutation after a final absence receipt.
    with io._no_links(root / ".mutation.lock").open("a+b") as mutation:
        try:
            fcntl.flock(mutation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        def apply(state):
            if account_id != state["account_id"] or absent < 3 or not absence_can_finish(state):
                return
            elapsed = max(0.0, time.time() - epoch(state["billing_start_utc"]))
            charge = 0.0 if state["mutation_phase"] == "not_started" else elapsed / 3600 * state["all_in_rate"]
            state.update(status="terminated", absence_confirmations=absent,
                         absence_account_id=account_id, terminated_verified_utc=stamp(),
                         lease_spend_upper_usd=charge,
                         combined_spend_upper_usd=state["prior_spend_upper_usd"] + charge)
        return change(root, apply)["status"] == "terminated"


def emergency_reason(state, live=None):
    elapsed = max(0.0, time.time() - epoch(state['billing_start_utc']))
    spend = state['prior_spend_upper_usd'] + elapsed / 3600 * state['all_in_rate']
    limit = min(BUDGET, state['prior_spend_upper_usd'] + state['job_incremental_cap_usd'])
    if time.time() >= epoch(state['watch_deadline_utc']):
        return 'fixed job spending deadline'
    if spend >= limit - 0.15:
        return 'combined or incremental spending ceiling'
    if live is not None and live['clientBalance'] <= state['preserve_balance'] + 0.15:
        return 'account balance reaches the reserved job allocation'
    return None


def explicit_teardown(state, path):
    path = io._no_links(path)
    document = io._json(path)
    if (not isinstance(document, dict) or set(document) != {'schema', 'binding', 'scope', 'authorized_by'}
            or document['schema'] != 'confirmation_explicit_teardown.v1'
            or document['binding'] != handoff.binding(state) or document['authorized_by'] != 'root'
            or not isinstance(document['scope'], str) or not document['scope'].strip()):
        raise ValueError('explicit teardown requires exact local root authorization')
    return {'path': str(path), 'record': handoff.checked_record(path), 'scope': document['scope']}


def reconcile_absence(root, reason):
    # This path never invokes a deletion API. halt still blocks a late create.
    halt(root, reason)
    absent = 0
    for _ in range(24):
        state = io._json(root / 'LEASE.json')
        live = checked_account(state)
        if owned_candidates(state, pods(state['account_id'])):
            return False
        absent += 1
        if absent >= 3 and confirm_absence(root, absent, live['id']):
            return True
        time.sleep(5)
    return False


def terminate_owned(root, authorization):
    if not isinstance(authorization, dict) or authorization.get('class') not in {
            'accepted_artifacts', 'explicit_teardown', 'budget_emergency'}:
        raise ValueError('deletion needs a structured authorization class')
    state = io._json(root / 'LEASE.json')
    if state['status'] == 'terminated':
        return True
    live = checked_account(state)
    emergency = emergency_reason(state, live)
    kind = 'budget_emergency' if emergency else authorization['class']
    # An emergency must never wait for a long collector lock.
    lock = handoff.handoff_lock(root, nonblocking=True) if kind == 'accepted_artifacts' else nullcontext()
    with lock:
        state = io._json(root / 'LEASE.json')
        evidence = {}
        if kind == 'accepted_artifacts':
            if not handoff.accepted_for_current_run(root, state, rehash=True):
                update(root, acceptance_rejected_utc=stamp())
                return False
            evidence['accepted_receipt'] = state['retrieval_accepted']
            reason = 'artifacts accepted locally for the exact final manifest'
        elif kind == 'explicit_teardown':
            evidence['explicit_authorization'] = explicit_teardown(state, authorization['authorization_path'])
            reason = evidence['explicit_authorization']['scope']
        else:
            emergency = emergency_reason(state, live)
            if not emergency:
                raise ValueError('budget emergency is not established')
            reason = emergency
        # Recheck time/cost after potentially long local hashing before classifying the mutation.
        late_emergency = emergency_reason(state, live)
        if late_emergency:
            kind, reason = 'budget_emergency', late_emergency
        preserved = set()
        for sha in state.get('stage_acceptances', {}):
            preserved.update(state.get('known_manifests', {}).get(sha, {}).get('manifest', {}).get('files', {}))
        if state.get('retrieval_accepted'):
            preserved.update(state.get('computation_finished', {}).get('manifest', {}).get('files', {}))
        try:
            contracted = set(handoff.load_contract(state)['files'])
        except (OSError, ValueError, KeyError, TypeError):
            contracted = set()
        intent = {'schema': 'confirmation_termination_intent.v1', 'class': kind, 'reason': reason,
                  'lease_name': state['name'], 'pod_id': state.get('pod_id'), 'run_id': state['run_id'],
                  'binding': handoff.binding(state) if state.get('pod_id') else None,
                  'recorded_utc': stamp(), 'observed_balance': live['clientBalance'],
                  'combined_spend_upper_usd': state['prior_spend_upper_usd'] + max(0.0, time.time() - epoch(state['billing_start_utc'])) / 3600 * state['all_in_rate'],
                  'previously_accepted_files': sorted(preserved), 'not_previously_accepted_files': sorted(contracted - preserved),
                  'preservation_scope': 'Recorded acceptance history; no new remote hashing at the emergency boundary.',
                  'retrieval_progress': state.get('retrieval_in_progress'), **evidence}
        intent_path = root / ('TERMINATION_INTENT_' + uuid.uuid4().hex + '.json')
        io._new_json(intent_path, intent)
        update(root, termination_class=kind, termination_authorization={'path': str(intent_path), 'record': handoff.checked_record(intent_path)})
        state = halt(root, reason)
        absent = 0
        for _ in range(24):
            try:
                state = io._json(root / 'LEASE.json')
                live = checked_account(state)
                matching = owned_candidates(state, pods(state['account_id']))
                if matching:
                    pod = matching[0]
                    state = observe_cost(root, state, pod)
                    state = observe_identity(root, pod)
                    if kind != 'budget_emergency' and handoff.binding(state) != intent['binding']:
                        raise ValueError('deletion authorization identity changed')
                    if kind == 'accepted_artifacts' and state.get('retrieval_accepted') != evidence['accepted_receipt']:
                        raise ValueError('accepted receipt changed before mutation')
                    api.terminate(pod['id'])
                    absent = 0
                else:
                    absent += 1
                    if absent >= 3 and confirm_absence(root, absent, live['id']):
                        return True
            except (api.APIError, OSError, TimeoutError, ValueError, KeyError, TypeError) as error:
                absent = 0
                update(root, termination_last_error_type=type(error).__name__)
            time.sleep(5)
        update(root, termination_unresolved_utc=stamp())
        return False


def arm_watcher(root, state):
    monitor = io._mkdir(root / "monitor")
    for name in CONTROLLER_FILES:
        source = HERE / name
        destination = monitor / name
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
        io._verify_record(destination, state["controller_sources"][name])
    unit = state["name"]
    subprocess.run(["systemd-run", "--user", "--unit", unit, "--collect",
                    "--property=Restart=on-failure", "--property=RestartSec=10",
                    sys.executable, str(monitor / "lease.py"), "watch", "--root", str(root),
                    "--key-file", state["api_key_file"]], check=True)
    for _ in range(WATCHER_STARTUP_SECONDS):
        state = io._json(root / "LEASE.json")
        if (state.get("watch_ready_utc") and state.get("watch_account_id") == state["account_id"]
                and 0 <= time.time() - epoch(state["watch_ready_utc"]) < 30):
            active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
            if active.returncode == 0 and active.stdout.strip() == "active":
                return
        time.sleep(1)
    raise RuntimeError("detached lease watcher did not acknowledge readiness")


def create(args):
    root = lease_root(args.root)
    with locked(LEDGER, ".lease_creation.lock"):
        # A separate attempt must explicitly account for any previous lease.
        for sibling in LEDGER.glob("*/LEASE.json"):
            old = io._json(io._no_links(sibling))
            if old.get("status") != "terminated":
                raise ValueError("an existing lease attempt must be reconciled first")
        if (root / "LEASE.json").exists():
            raise ValueError("this lease directory already has an intent")
        validate_notice(args.notice_utc)
        live, writable, quoted = account(), account(write_key=True), offer()
        if live["id"] != writable["id"]:
            raise ValueError("read and write credentials refer to different accounts")
        if pods(live["id"]) or number(live["currentSpendPerHr"], "account rate") != 0:
            raise ValueError("the account already has resources or other active spending")
        state = make_plan(args.bundle, args.notice_utc, balance=live["clientBalance"],
                          rate=quoted["uninterruptablePrice"], run_config=args.run_config,
                          debit_path=args.prior_debit, job_cap=args.job_cap_usd)
        if state['account_id'] != live['id']:
            raise ValueError('current account differs from the immutable prior debit')
        public = args.public_key.read_text().strip()
        if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,3}(?: [^\r\n]*)?", public):
            raise ValueError("an explicit ed25519 SSH public key is required")
        io._no_links(args.ssh_identity)
        if not args.ssh_identity.is_file():
            raise ValueError("SSH private identity is missing")
        derived = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", str(args.ssh_identity)],
                                 capture_output=True, text=True, check=True).stdout.strip()
        if derived.split()[:2] != public.split()[:2]:
            raise ValueError("public key does not match the SSH private identity")
        state.update(account_id=live["id"], quote=quoted,
                     api_key_file=str(io._no_links(api.KEY_FILE)),
                     ssh_identity=str(args.ssh_identity.absolute()), status="pending",
                     controller_sources={name: handoff.checked_record(HERE / name) for name in CONTROLLER_FILES})
        io._mkdir(root)
        io._new_json(root / "LEASE.json", state)
        arm_watcher(root, state)
        validate_notice(args.notice_utc)
        io._verify_record(args.bundle, state["bundle_record"])
        active = subprocess.run(["systemctl", "--user", "is-active", state["name"]], capture_output=True, text=True)
        if active.returncode or active.stdout.strip() != "active":
            raise RuntimeError("watcher stopped before the deployment mutation")
        environment = {"PUBLIC_KEY": public, "NVIDIA_VISIBLE_DEVICES": "all",
                       "NVIDIA_DRIVER_CAPABILITIES": "compute,utility", "JLENS_LEASE_NAME": state["name"]}
        request = {"cloudType": "COMMUNITY", "gpuCount": 1, "gpuTypeId": GPU,
                   "name": state["name"], "imageName": IMAGE, "containerDiskInGb": DISK_GB,
                   "volumeInGb": 0, "minVcpuCount": 8, "minMemoryInGb": 64,
                   "ports": "22/tcp", "supportPublicIp": True,
                   "terminateAfter": state["provider_deadline_utc"],
                   "dockerArgs": "/bin/bash -lc " + shlex.quote((HERE / "pod_entry.sh").read_text()),
                   "env": [{"key": key, "value": value} for key, value in environment.items()]}
        query = "mutation Deploy($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input:$input) { " + POD_FIELDS + " } }"
        try:
            with locked(root, ".mutation.lock"):
                checked_account(state)
                state = begin_create(root)
                try:
                    pod = api.gql(query, {"input": request}, write=True)["podFindAndDeployOnDemand"]
                except api.APIError as error:
                    if completed_supply_rejection(error, state["account_id"]):
                        record_rejection(root, error.graphql_body,
                                         {"kind": "completed_graphql_error_response", "http_status": error.status})
                    raise
                state = observe_cost(root, state, pod)
                state = observe_identity(root, pod)
                fields = bind_pod(state, pod)
                current = finish_create(root, fields)
            if current.get("halt_requested"):
                request_worker_stop(root, current)
        except BaseException:
            fail_create(root)
            # No second deploy call. The watcher reconciles/terminates by name.
            raise
        print(json.dumps({"root": str(root), "pod_id": pod["id"], "gpu": GPU,
                          "all_in_rate": fields["all_in_rate"], "work_deadline_utc": state["work_deadline_utc"]}))


def watch(args):
    root = lease_root(args.root)
    with locked(root, '.watch.lock'):
        state = io._json(root / 'LEASE.json')
        if state['status'] == 'terminated':
            return
        if str(io._no_links(api.KEY_FILE)) != state['api_key_file']:
            raise ValueError('watcher key-file path differs from the creator')
        checked_account(state)
        update(root, watch_ready_utc=stamp(), watch_account_id=state['account_id'])
        failures = 0
        while True:
            state = io._json(root / 'LEASE.json')
            if state['status'] == 'terminated':
                return
            try:
                update(root, watch_heartbeat_utc=stamp())
                live = checked_account(state)
                if emergency_reason(state, live):
                    if terminate_owned(root, {'class': 'budget_emergency'}):
                        return
                    continue
                candidates = owned_candidates(state, pods(state['account_id']))
                if candidates:
                    state = observe_cost(root, state, candidates[0])
                    state = observe_identity(root, candidates[0])
                    fields = bind_pod(state, candidates[0])
                    connection = endpoint(candidates[0])
                    state = update(root, **fields, **({'ssh': connection} if connection else {}))
                elif state.get('pod_id') or state.get('halt_requested'):
                    if reconcile_absence(root, 'provider no longer lists the owned pod'):
                        return
                    continue
                state = io._json(root / 'LEASE.json')
                # Collection discovers closed stages while later computation runs.
                ensure_sync(root, state)
                if state.get('ssh'):
                    try:
                        status = remote_status(root, state)
                        if status is not None:
                            state = remember_worker_status(root, state, status)
                        else:
                            state = update(root, worker_status_missing_utc=stamp())
                    except (OSError, TimeoutError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
                        # A failing status read cannot skip STOP, local acceptance,
                        # or the independent budget decision below.
                        state = update(root, worker_status_error_type=type(error).__name__,
                                       worker_status_error_utc=stamp(),
                                       worker_status_failures=state.get('worker_status_failures', 0) + 1)
                stop_reason = None
                if state.get('halt_requested'):
                    stop_reason = 'creation halt or explicit stop barrier'
                elif time.time() >= epoch(state['work_deadline_utc']):
                    stop_reason = 'early preservation reserve reached'
                elif time.time() >= epoch(state['setup_deadline_utc']) and not setup_completed_for_lease(state):
                    stop_reason = 'setup did not complete within its admission window'
                elif not candidates and time.time() - epoch(state['billing_start_utc']) > 600:
                    stop_reason = 'creation or provisioning remained unresolved'
                if stop_reason:
                    update(root, computation_stop_reason=stop_reason)
                    try:
                        request_worker_stop(root, state)
                    except (OSError, TimeoutError, subprocess.SubprocessError) as error:
                        update(root, stop_last_error_type=type(error).__name__, stop_last_error_utc=stamp())
                state = io._json(root / 'LEASE.json')
                elapsed = max(0.0, time.time() - epoch(state['billing_start_utc']))
                update(root, observed_combined_spend_upper_usd=state['prior_spend_upper_usd'] + elapsed / 3600 * state['all_in_rate'],
                       observed_balance=live['clientBalance'], last_poll_utc=stamp())
                if emergency_reason(state, live):
                    if terminate_owned(root, {'class': 'budget_emergency'}):
                        return
                elif handoff.accepted_for_current_run(root, state, rehash=False):
                    if terminate_owned(root, {'class': 'accepted_artifacts'}):
                        return
                failures = 0
            except (api.APIError, OSError, TimeoutError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
                failures += 1
                update(root, watch_last_error_type=type(error).__name__, watch_last_error_utc=stamp(),
                       watch_consecutive_failures=failures)
                # Transport/validation errors never grant deletion. The fixed hard
                # deadline is checked independently even when worker status fails.
                state = io._json(root / 'LEASE.json')
                if emergency_reason(state):
                    if terminate_owned(root, {'class': 'budget_emergency'}):
                        return
            time.sleep(15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    plan = sub.add_parser('plan')
    plan.add_argument('--bundle', type=Path, required=True)
    plan.add_argument('--balance', type=float, required=True)
    create_parser = sub.add_parser('create')
    create_parser.add_argument('--bundle', type=Path, required=True)
    create_parser.add_argument('--root', type=Path, required=True)
    create_parser.add_argument('--notice-utc', required=True)
    create_parser.add_argument('--public-key', type=Path, required=True)
    create_parser.add_argument('--ssh-identity', type=Path, required=True)
    create_parser.add_argument('--key-file', type=Path, default=api.KEY_FILE)
    for child in (plan, create_parser):
        child.add_argument('--run-config', type=Path, required=True)
        child.add_argument('--prior-debit', type=Path, required=True)
        child.add_argument('--job-cap-usd', type=float, required=True)
    rejection = sub.add_parser('reconcile-rejection')
    rejection.add_argument('--root', type=Path, required=True)
    rejection.add_argument('--capture', type=Path, required=True)
    rejection.add_argument('--key-file', type=Path, default=api.KEY_FILE)
    for name in ('watch', 'status', 'terminate', 'sync'):
        child = sub.add_parser(name)
        child.add_argument('--root', type=Path, required=True)
        if name in ('watch', 'terminate'):
            child.add_argument('--key-file', type=Path, default=api.KEY_FILE)
        if name == 'terminate':
            child.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, 'root'):
        args.root = lease_root(args.root)
    if hasattr(args, 'key_file'):
        api.KEY_FILE = io._no_links(args.key_file)
    if args.command == 'self-test':
        import test_lifecycle
        test_lifecycle.main([])
    elif args.command == 'plan':
        print(json.dumps(make_plan(args.bundle, None, balance=args.balance, rate=MAX_GPU_RATE,
            run_config=args.run_config, debit_path=args.prior_debit, job_cap=args.job_cap_usd), indent=2))
    elif args.command == 'create':
        create(args)
    elif args.command == 'reconcile-rejection':
        print(json.dumps(reconcile_rejection(args), indent=2))
    elif args.command == 'watch':
        watch(args)
    elif args.command == 'terminate':
        if not terminate_owned(args.root, {'class': 'explicit_teardown', 'authorization_path': args.authorization}):
            raise RuntimeError('termination remains unresolved; watcher retains the hard deadline')
    elif args.command == 'sync':
        print(json.dumps(sync_results(args.root), indent=2))
    else:
        print(json.dumps(io._json(args.root / 'LEASE.json'), indent=2))


if __name__ == '__main__':
    main()
