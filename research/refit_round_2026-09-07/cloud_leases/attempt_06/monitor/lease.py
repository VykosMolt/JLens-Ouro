"""One supervised RunPod lease within the user's combined $25 ceiling.

Plan and self-test are offline. Create requires a recent notice timestamp,
checks live funding/price, writes its intent before the mutation, and installs
a detached systemd watcher first. The provider also receives a fixed deletion
deadline. An uncertain create is reconciled by the unique name, never retried.
Credentials stay in the local account client and are never sent to the GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

HERE = Path(__file__).resolve().parent
IMAGE = "python@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f"
GPU = "NVIDIA GeForce RTX 5090"
MAX_GPU_RATE = 0.69
DISK_GB = 120
STORAGE_RATE = DISK_GB * 0.10 / 730.0
BUDGET = 25.0
# Allow bounded time for both account-identity reads before watcher readiness.
WATCHER_STARTUP_SECONDS = 180
LEDGER = Path("/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07/cloud_leases")
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
        action(state)
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


def make_plan(bundle, notice, *, balance, rate, prior_spend=0.0, now=None):
    now = time.time() if now is None else now
    prior_spend = number(prior_spend, "prior experiment spending")
    rate = number(rate, "GPU rate", minimum=0.001)
    balance = number(balance, "available funds")
    if rate > MAX_GPU_RATE or prior_spend >= BUDGET:
        raise ValueError("rate or remaining combined budget is invalid")
    remaining = BUDGET - prior_spend
    if balance < remaining:
        raise ValueError("the account lacks the remaining authorized funds")
    # Reserve $0.15 beyond the provider deadline for polling/billing latency.
    # The provider deadline remains safe when the assigned price exceeds the quote.
    all_in = HARD_RATE
    seconds = int((remaining - 0.15) / all_in * 3600)
    if seconds <= 7200:
        raise ValueError("remaining budget lacks a useful work and retrieval window")
    return {
        "schema_version": 1, "name": "jlens-refit-" + uuid.uuid4().hex,
        "image": IMAGE, "gpu_type_id": GPU, "cloud": "COMMUNITY", "gpu_count": 1,
        "container_disk_gb": DISK_GB, "volume_gb": 0, "max_gpu_rate": MAX_GPU_RATE,
        "quoted_gpu_rate": rate, "storage_rate": STORAGE_RATE, "all_in_rate": all_in,
        "combined_cap_usd": BUDGET, "prior_spend_upper_usd": prior_spend,
        "starting_balance": balance, "preserve_balance": max(0.0, balance - remaining),
        "billing_start_utc": stamp(now), "provider_deadline_utc": stamp(now + seconds),
        "watch_deadline_utc": stamp(now + seconds - 600),
        "work_deadline_utc": stamp(now + seconds - 3600),
        "setup_deadline_utc": stamp(now + min(7200, seconds - 7200)),
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


def request_worker_stop(root, state):
    if state.get("ssh"):
        subprocess.run([*ssh_options(root, state), "root@" + state["ssh"]["host"],
                        f"touch {REMOTE}/STOP"], capture_output=True, timeout=30, check=True)


def local_inventory(root):
    root = io._no_links(root)
    if not root.is_dir():
        raise ValueError("results directory is missing")
    files = {}
    def walk_error(error):
        raise error
    for directory, directories, names in os.walk(root, followlinks=False, onerror=walk_error):
        directories[:] = [name for name in directories if name != "runtime_cache"]
        for name in directories:
            io._no_links(Path(directory) / name)
        for name in names:
            if name.endswith(".lock"):
                continue
            path = io._no_links(Path(directory) / name)
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("nonregular retrieved results file")
            files[path.relative_to(root).as_posix()] = io._record(path)
    return files


def _sync_results(root):
    """Retrieve a quiescent results tree and compare every byte to its inventory."""
    state = io._json(root / "LEASE.json")
    status = remote_status(root, state)
    if not status or status.get("phase") not in ("complete", "stopped", "failed"):
        raise ValueError("artifact retrieval requires a stopped or completed worker")
    script = r'''
import hashlib,json,os,pathlib,stat
root=pathlib.Path('/workspace/jlens/results')
if root.is_symlink() or not root.is_dir(): raise RuntimeError('missing regular results directory')
files={}
def walk_error(error): raise error
for directory,dirs,names in os.walk(root,followlinks=False,onerror=walk_error):
    dirs[:]=[name for name in dirs if name!='runtime_cache']
    for name in dirs:
        if (pathlib.Path(directory)/name).is_symlink(): raise RuntimeError('linked results directory')
    for name in names:
        if name.endswith('.lock'): continue
        p=pathlib.Path(directory)/name
        if not stat.S_ISREG(p.lstat().st_mode): raise RuntimeError('nonregular results file')
        h=hashlib.sha256()
        with p.open('rb') as f:
            while block:=f.read(8*1024*1024): h.update(block)
        files[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(files,sort_keys=True))
'''
    command = [*ssh_options(root, state), "root@" + state["ssh"]["host"], "python -c " + shlex.quote(script)]
    inventory = json.loads(subprocess.run(command, capture_output=True, text=True, timeout=900, check=True).stdout)
    destination = io._mkdir(root / "retrieved")
    transport = shlex.join(ssh_options(root, state))
    subprocess.run(["rsync", "-a", "--partial", "--timeout=60", "--exclude=runtime_cache/",
                    "--exclude=*.lock", "-e", transport,
                    f"root@{state['ssh']['host']}:{REMOTE}/results/", str(destination) + "/"], check=True)
    after = json.loads(subprocess.run(command, capture_output=True, text=True, timeout=900, check=True).stdout)
    if inventory != after:
        raise ValueError("remote artifact inventory changed during retrieval")
    if not isinstance(inventory, dict) or local_inventory(destination) != inventory:
        raise ValueError("local artifact inventory does not exactly match the remote files and bytes")
    io._atomic_json(root / "RETRIEVAL_VERIFIED.json", {
        "status": "passed", "lease_name": state["name"], "verified_utc": stamp(),
        "worker_status": status, "files": inventory,
        "scope": "all bytes of the stopped worker's results; scientific completeness is recorded separately",
    })


def sync_results(root):
    with io._no_links(root / ".sync.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        _sync_results(root)


def ensure_sync(root):
    if (root / "RETRIEVAL_VERIFIED.json").exists():
        return
    with io._no_links(root / ".sync.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    with (root / "retrieval.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(Path(__file__)), "sync", "--root", str(root)],
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
        if not terminate_owned(root, "completed provider supply rejection; verify no resource exists"):
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


def terminate_owned(root, reason):
    state = halt(root, reason)
    if state["status"] == "terminated":
        return True
    # When create was ambiguous, only the durable unique name can be adopted.
    absent = 0
    for attempt in range(24):
        try:
            state = io._json(root / "LEASE.json")
            live = checked_account(state)
            matching = owned_candidates(state, pods(state["account_id"]))
            if matching:
                pod = matching[0]
                state = observe_cost(root, state, pod)
                state = observe_identity(root, pod)
                api.terminate(pod["id"])
                absent = 0
            else:
                absent += 1
                if absent >= 3 and confirm_absence(root, absent, live["id"]):
                    return True
        except (api.APIError, OSError, TimeoutError, ValueError, KeyError, TypeError) as error:
            absent = 0
            update(root, termination_last_error_type=type(error).__name__)
        time.sleep(5)
    update(root, termination_unresolved_utc=stamp())
    return False


def arm_watcher(root, state):
    monitor = io._mkdir(root / "monitor")
    for name in ("lease.py", "run_refits.py", "runpod_api.py", "pod_entry.sh"):
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
        previous = [io._json(io._no_links(path)) for path in LEDGER.glob("*/LEASE.json")]
        prior = sum(number(value["lease_spend_upper_usd"], "past lease spending") for value in previous)
        state = make_plan(args.bundle, args.notice_utc, balance=live["clientBalance"],
                          rate=quoted["uninterruptablePrice"], prior_spend=prior)
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
                     controller_sources={name: io._record(HERE / name) for name in
                                         ("lease.py", "run_refits.py", "runpod_api.py", "pod_entry.sh")})
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
                terminate_owned(root, "watcher requested termination during creation")
        except BaseException:
            fail_create(root)
            # No second deploy call. The watcher reconciles/terminates by name.
            raise
        print(json.dumps({"root": str(root), "pod_id": pod["id"], "gpu": GPU,
                          "all_in_rate": fields["all_in_rate"], "work_deadline_utc": state["work_deadline_utc"]}))


def watch(args):
    root = lease_root(args.root)
    with locked(root, ".watch.lock"):
        state = io._json(root / "LEASE.json")
        if state["status"] == "terminated":
            return
        if str(io._no_links(api.KEY_FILE)) != state["api_key_file"]:
            raise ValueError("watcher key-file path differs from the creator's local path")
        checked_account(state)
        update(root, watch_ready_utc=stamp(), watch_account_id=state["account_id"])
        failures = 0
        while True:
            state = io._json(root / "LEASE.json")
            if state["status"] == "terminated":
                return
            try:
                update(root, watch_heartbeat_utc=stamp())
                elapsed = time.time() - epoch(state["billing_start_utc"])
                live = checked_account(state)
                candidates = owned_candidates(state, pods(state["account_id"]))
                if candidates:
                    state = observe_cost(root, state, candidates[0])
                    state = observe_identity(root, candidates[0])
                    fields = bind_pod(state, candidates[0])
                    connection = endpoint(candidates[0])
                    state = update(root, **fields, **({"ssh": connection} if connection else {}))
                elif state.get("pod_id"):
                    if terminate_owned(root, "provider no longer lists the owned pod"):
                        return
                    continue
                # Re-read halt and rate updates made concurrently by the creator.
                state = io._json(root / "LEASE.json")
                spend = state["prior_spend_upper_usd"] + max(0.0, elapsed) / 3600 * state["all_in_rate"]
                status = remote_status(root, state)
                if status:
                    state = update(root, last_worker_status=status)
                if status and status.get("phase") in ("complete", "stopped", "failed"):
                    ensure_sync(root)
                elif time.time() >= epoch(state["work_deadline_utc"]):
                    request_worker_stop(root, state)
                reason = None
                if state.get("halt_requested"):
                    reason = "explicit stop or failed creation"
                elif time.time() >= epoch(state["watch_deadline_utc"]) or spend >= BUDGET - 0.15:
                    reason = "combined spending deadline"
                elif live["clientBalance"] <= state["preserve_balance"] + 0.15:
                    reason = "account balance reaches the experiment allocation"
                elif time.time() >= epoch(state["setup_deadline_utc"]) and not (status and status.get("setup_complete")):
                    reason = "setup did not finish within the bounded window"
                elif not candidates and elapsed > 600:
                    reason = "creation or provisioning remained unresolved"
                elif (root / "RETRIEVAL_VERIFIED.json").exists():
                    receipt = io._json(root / "RETRIEVAL_VERIFIED.json")
                    if receipt.get("lease_name") == state["name"] and receipt.get("status") == "passed":
                        reason = "artifacts retrieved and verified"
                update(root, observed_combined_spend_upper_usd=spend,
                       observed_balance=live["clientBalance"], last_poll_utc=stamp())
                if reason:
                    if terminate_owned(root, reason):
                        return
                failures = 0
            except (api.APIError, OSError, TimeoutError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
                failures += 1
                update(root, watch_last_error_type=type(error).__name__, watch_consecutive_failures=failures)
                if failures >= 3:
                    if terminate_owned(root, "three consecutive supervision failures"):
                        return
            time.sleep(15)


def self_test():
    """Exercise lease races and receipts using only temporary files and CPU fakes."""
    from contextlib import ExitStack, redirect_stdout
    import io as text_io
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    cases = []
    transport = api._request
    clock = [1800000000.0]
    def passed(name, condition=True):
        if not condition:
            raise AssertionError(name)
        cases.append(name)
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, RuntimeError, api.APIError):
            passed(name)
        else:
            raise AssertionError(name + " was accepted")
    def forbidden(*args, **kwargs):
        raise AssertionError("offline self-test reached an external operation")
    def sleep(seconds):
        clock[0] += seconds

    with tempfile.TemporaryDirectory(prefix="jlens-lease-test-") as temporary, ExitStack() as stack:
        base = Path(temporary)
        ledger = base / "ledger"
        bundle = base / "bundle.tar"
        bundle.write_bytes(b"offline bundle fixture")
        public = base / "identity.pub"
        public.write_text("ssh-ed25519 " + "A" * 32 + " offline-test\n")
        private = base / "identity"
        private.write_text("offline fake; ssh-keygen is mocked\n")
        key_path = base / "local-credential-path"
        for target, name, value in ((sys.modules[__name__], "LEDGER", ledger),
                                    (time, "time", lambda: clock[0]), (time, "sleep", sleep),
                                    (api, "_request", forbidden), (api, "_key", forbidden),
                                    (api, "KEY_FILE", key_path), (subprocess, "run", forbidden),
                                    (subprocess, "Popen", forbidden)):
            stack.enter_context(patch.object(target, name, value))
        serial = [0]
        discount_rate = MAX_GPU_RATE / 2
        def fixture(*, phase="not_started", ready=True):
            serial[0] += 1
            root = io._mkdir(ledger / str(serial[0]))
            state = make_plan(bundle, stamp(), balance=27.0, rate=discount_rate, now=clock[0])
            state.update(status="pending" if phase == "not_started" else "creating",
                         mutation_phase=phase, account_id="account-A", api_key_file=str(key_path),
                         ssh_identity=str(private), ssh={"host": "192.0.2.1", "port": 2200})
            if ready:
                state.update(watch_ready_utc=stamp(), watch_account_id="account-A")
            io._new_json(root / "LEASE.json", state)
            return root, state
        def sample_pod(state, **fields):
            return {"id": "pod-A", "name": state["name"], "machineId": "machine-A",
                    "imageName": IMAGE, "gpuCount": 1, "containerDiskInGb": DISK_GB,
                    "volumeInGb": 0, "costPerHr": MAX_GPU_RATE,
                    "machine": {"gpuDisplayName": state.get("gpu_type_id", GPU).removeprefix("NVIDIA GeForce ")}, **fields}
        live = {"id": "account-A", "clientBalance": 27.0, "currentSpendPerHr": 0.0}

        supply_body = {"errors": [{"message": SUPPLY_MESSAGE, "path": ["podFindAndDeployOnDemand"],
                                  "extensions": {"code": "SUPPLY_CONSTRAINT", "userId": "account-A"}}],
                       "data": {"podFindAndDeployOnDemand": None}}
        def supply_error(body=None, **kwargs):
            return api.APIError("offline completed supply rejection", status=kwargs.get("status", 200),
                                graphql_body=supply_body if body is None else body,
                                response_completed=kwargs.get("completed", True))
        with patch.object(api, "_request", return_value=(200, supply_body)):
            try:
                api.gql("mutation Deploy { podFindAndDeployOnDemand { id } }", write=True)
            except api.APIError as error:
                passed("completed GraphQL response preserves structured rejection",
                       completed_supply_rejection(error, "account-A") and error.graphql_body == supply_body)
                error.graphql_body["errors"][0]["message"] = "mutated diagnostic copy"
                passed("structured API error does not alias the original decoded response",
                       supply_body["errors"][0]["message"] == SUPPLY_MESSAGE)
            else:
                raise AssertionError("GraphQL rejection was accepted")
        mutations = (
            lambda b: b["errors"].append(b["errors"][0].copy()),
            lambda b: b["errors"][0].update(path=["podFindAndDeployOnDemand", "id"]),
            lambda b: b["errors"][0].update(message="temporarily unavailable"),
            lambda b: b["errors"][0]["extensions"].update(code="INTERNAL_SERVER_ERROR"),
            lambda b: b["errors"][0]["extensions"].update(userId="account-B"),
            lambda b: b["errors"][0]["extensions"].pop("userId"),
            lambda b: b.update(data=None),
            lambda b: b["data"].update(podFindAndDeployOnDemand={"id": "pod-A"}),
            lambda b: b["data"].update(otherMutation=None),
        )
        for index, mutate in enumerate(mutations):
            body = json.loads(json.dumps(supply_body))
            mutate(body)
            passed("nonexact supply rejection remains uncertain " + str(index),
                   not completed_supply_rejection(supply_error(body), "account-A"))
        for label, error in (("network exception", api.APIError("lost response")),
                             ("incomplete response", supply_error(completed=False)),
                             ("server HTTP error", supply_error(status=500))):
            passed(label + " cannot classify as rejected", not completed_supply_rejection(error, "account-A"))

        root, state = fixture(phase="in_flight")
        with locked(root, ".mutation.lock"):
            record_rejection(root, supply_body, {"kind": "completed_graphql_error_response", "http_status": 200})
            passed("rejected create still respects the suspended creator lock", not confirm_absence(root, 3, "account-A"))
        fail_create(root)
        rejected = io._json(root / "LEASE.json")
        passed("failure cleanup preserves the durable exact rejection", rejected["mutation_phase"] == "rejected"
               and rejected["halt_requested"] and verified_rejection(rejected))
        passed("two rejection absences cannot close a lease", not confirm_absence(root, 2, "account-A"))
        passed("wrong-account rejection absences cannot close a lease", not confirm_absence(root, 3, "account-B"))
        damaged = json.loads(json.dumps(rejected))
        damaged["create_rejection"]["canonical_json_sha256"] = "0" * 64
        passed("a rejection with altered evidence cannot close by absence", not absence_can_finish(damaged))
        damaged.pop("create_rejection")
        passed("a rejected phase without evidence cannot close by absence", not absence_can_finish(damaged))
        root, state = fixture(phase="observed")
        update(root, pod_id="pod-A", machine_id="machine-A")
        rejects("supply body cannot erase a previously bound resource", lambda: record_rejection(
            root, supply_body, {"kind": "completed_graphql_error_response", "http_status": 200}))

        root, state = fixture(phase="uncertain")
        halt(root, "historical captured rejection")
        capture = {"schema_version": 1, "response_body": supply_body, "deployment_call_count": 1,
                   "process_exit_code": 1, "create_exec_session_id": 123,
                   "provenance": "Complete normalized GraphQL exception capture; not raw HTTP bytes."}
        capture_path = root / "CAPTURED_PROVIDER_REJECTION.json"
        io._new_json(capture_path, capture)
        clock[0] += 10
        with patch.object(sys.modules[__name__], "checked_account", return_value=live) as accounts, \
                patch.object(sys.modules[__name__], "pods", return_value=[]) as listings, \
                patch.object(api, "terminate", forbidden):
            result = reconcile_rejection(SimpleNamespace(root=root, capture=capture_path))
        passed("historical capture closes after three fresh verified account absences",
               result["status"] == "terminated" and result["mutation_phase"] == "rejected"
               and result["absence_confirmations"] == 3 and accounts.call_count == 4 and listings.call_count == 3)
        passed("historical rejection retains conservative elapsed spending",
               result["lease_spend_upper_usd"] > 0
               and result["lease_spend_upper_usd"] == (clock[0] - epoch(state["billing_start_utc"])) / 3600 * result["all_in_rate"])
        passed("historical rejection labels its canonical hash and capture provenance",
               result["create_rejection"]["canonical_json_sha256"] == io._digest(supply_body)
               and result["create_rejection"]["provenance"]["capture_record"] == io._record(capture_path)
               and "not raw HTTP" in result["create_rejection"]["hash_scope"])

        quote = {"gpuTypes": [{"id": GPU, "memoryInGb": 32, "lowestPrice": {
            "stockStatus": "low", "uninterruptablePrice": discount_rate, "minMemory": 91, "minVcpu": 12}}]}
        with patch.object(api, "gql", return_value=quote) as quote_request:
            offer()
        quoted_query = "".join(quote_request.call_args.args[0].split())
        passed("offer quote requests the deployment's disk RAM CPU and public IP constraints",
               all(value in quoted_query for value in ("totalDisk:120", "minMemoryInGb:64", "minVcpuCount:8", "supportPublicIp:true"))
               and "minDisk:" not in quoted_query)

        plan = make_plan(bundle, stamp(), balance=27.0, rate=discount_rate, prior_spend=2.0)
        duration = epoch(plan["provider_deadline_utc"]) - epoch(plan["billing_start_utc"])
        passed("quoted discount never extends the hard-rate provider deadline",
               plan["all_in_rate"] == HARD_RATE and 2.0 + duration / 3600 * HARD_RATE <= BUDGET - 0.15)
        passed("5090 deadline uses its 0.69 ceiling and is shorter than the old 0.34 basis",
               GPU == "NVIDIA GeForce RTX 5090" and MAX_GPU_RATE == 0.69
               and HARD_RATE == 0.69 + STORAGE_RATE
               and duration == int((BUDGET - 2.0 - 0.15) / (0.69 + STORAGE_RATE) * 3600)
               and duration < int((BUDGET - 2.0 - 0.15) / (0.34 + STORAGE_RATE) * 3600))
        full_price_plan = make_plan(bundle, stamp(), balance=27.0, rate=MAX_GPU_RATE, prior_spend=2.0)
        passed("discount changes none of the fixed provider watcher work or setup deadlines",
               all(plan[field] == full_price_plan[field] for field in (
                   "provider_deadline_utc", "watch_deadline_utc", "work_deadline_utc", "setup_deadline_utc")))
        passed("work and watcher reserves precede the same provider deadline",
               epoch(plan["work_deadline_utc"]) == epoch(plan["provider_deadline_utc"]) - 3600
               and epoch(plan["watch_deadline_utc"]) == epoch(plan["provider_deadline_utc"]) - 600)
        for rate in (None, True, float("nan"), float("inf"), -1, 0, 0.70):
            rejects("invalid quote " + repr(rate), lambda rate=rate: make_plan(bundle, stamp(), balance=27, rate=rate))
        rejects("outside ledger root", lambda: lease_root(base / "elsewhere" / "lease"))
        rejects("nested ledger root", lambda: lease_root(ledger / "nested" / "lease"))
        passed("canonical direct child", lease_root(ledger / "allowed") == ledger / "allowed")

        root, state = fixture()
        passed("selected 5090 binds at the authorized 0.69 price",
               bind_pod(state, sample_pod(state))["actual_gpu_rate"] == 0.69)
        selections = (("NVIDIA GeForce RTX 3090", 0.22), ("NVIDIA GeForce RTX 4090", 0.34), (GPU, MAX_GPU_RATE))
        for selected, ceiling in selections:
            recorded_rate = max(0.34, ceiling) + STORAGE_RATE
            selected_state = {**state, "gpu_type_id": selected, "max_gpu_rate": ceiling, "all_in_rate": recorded_rate}
            for label in (selected, selected.removeprefix("NVIDIA GeForce ")):
                bound = bind_pod(selected_state, sample_pod(selected_state, costPerHr=ceiling,
                                 machine={"gpuDisplayName": label}))
                passed("explicit GPU alias retains its own price ceiling and recorded floor " + label,
                       bound["actual_gpu_rate"] == ceiling and bound["all_in_rate"] == recorded_rate)
            rejects("assigned price above this lease's own ceiling " + selected, lambda: bind_pod(
                selected_state, sample_pod(selected_state, costPerHr=ceiling + 0.01)))
            for other, _ in selections:
                if other != selected:
                    rejects("wrong assignment " + other + " for " + selected, lambda: bind_pod(
                        selected_state, sample_pod(selected_state, costPerHr=ceiling,
                                       machine={"gpuDisplayName": other.removeprefix("NVIDIA GeForce ")})))
            historical_root, historical_state = fixture()
            historical_state.update(gpu_type_id=selected, max_gpu_rate=ceiling, all_in_rate=recorded_rate)
            io._atomic_json(historical_root / "LEASE.json", historical_state)
            lowered = update(historical_root, all_in_rate=ceiling / 2 + STORAGE_RATE)
            passed("discounted observation preserves this lease's recorded floor " + selected,
                   lowered["all_in_rate"] == recorded_rate)
            raised = update(historical_root, all_in_rate=2.0 + STORAGE_RATE)
            updated = update(historical_root, **bind_pod(historical_state, sample_pod(historical_state, costPerHr=ceiling)))
            passed("stale binding preserves this lease's historical maximum " + selected,
                   updated["all_in_rate"] == raised["all_in_rate"] == 2.0 + STORAGE_RATE)
        rejects("GPU outside the explicit old and current selections", lambda: bind_pod(
            {**state, "gpu_type_id": "NVIDIA GeForce RTX 3090 Ti"}, sample_pod(state)))
        rejects("assigned 0.70 rate exceeds the selected GPU ceiling", lambda: bind_pod(
            state, sample_pod(state, costPerHr=0.70)))
        too_expensive = json.loads(json.dumps(quote))
        too_expensive["gpuTypes"][0]["lowestPrice"]["uninterruptablePrice"] = 0.70
        with patch.object(api, "gql", return_value=too_expensive):
            rejects("quoted 0.70 rate exceeds the selected GPU ceiling", offer)
        exact_quote = json.loads(json.dumps(quote))
        exact_quote["gpuTypes"][0]["lowestPrice"]["uninterruptablePrice"] = 0.69
        with patch.object(api, "gql", return_value=exact_quote):
            passed("quoted 0.69 5090 with sufficient host RAM and CPU is accepted", offer()["uninterruptablePrice"] == 0.69)
        for field, value in (("id", "NVIDIA GeForce RTX 4090"), ("id", "NVIDIA GeForce RTX 5090 Ti"),
                             ("memoryInGb", 24), ("memoryInGb", 31), ("memoryInGb", True),
                             ("memoryInGb", "32"), ("memoryInGb", None)):
            bad_quote = json.loads(json.dumps(quote))
            bad_quote["gpuTypes"][0][field] = value
            with patch.object(api, "gql", return_value=bad_quote):
                rejects("invalid quoted GPU " + field + " " + repr(value), offer)
        for field, value in (("minMemory", 63), ("minMemory", None), ("minMemory", True),
                             ("minMemory", "91"), ("minVcpu", 7), ("minVcpu", None)):
            bad_quote = json.loads(json.dumps(quote))
            bad_quote["gpuTypes"][0]["lowestPrice"][field] = value
            with patch.object(api, "gql", return_value=bad_quote):
                rejects("invalid quoted host " + field + " " + repr(value), offer)
        for field, value in (("id", None), ("id", 12), ("id", ""), ("id", "../pod"),
                             ("machineId", None), ("machineId", ""), ("gpuCount", True),
                             ("containerDiskInGb", 120.0), ("volumeInGb", False),
                             ("costPerHr", True), ("costPerHr", 0.70), ("machine", None)):
            rejects("invalid assigned " + field + " " + repr(value),
                    lambda field=field, value=value: bind_pod(state, sample_pod(state, **{field: value})))
        for listing in (None, {}, [{}], [{"id": None, "name": "x"}],
                        [sample_pod(state), sample_pod(state)]):
            with patch.object(api, "gql", return_value={"myself": {"id": "account-A", "pods": listing}}):
                rejects("malformed listing " + str(len(cases)), lambda: pods("account-A"))
        with patch.object(api, "gql", return_value={"myself": {"id": "account-B", "pods": []}}):
            rejects("wrong-account empty listing", lambda: pods("account-A"))
        for value in (None, False, float("nan")):
            with patch.object(api, "gql", return_value={"myself": {**live, "currentSpendPerHr": value}}):
                rejects("invalid account rate " + repr(value), account)

        stale = state.copy()
        observe_cost(root, state, sample_pod(state, costPerHr=2.0))
        observe_cost(root, stale, sample_pod(state, costPerHr=discount_rate))
        update(root, **bind_pod(stale, sample_pod(state)))
        passed("stale observations and binding cannot lower accrued rate",
               io._json(root / "LEASE.json")["all_in_rate"] == 2.0 + STORAGE_RATE)
        halt(root, "stop before mutation")
        rejects("halt barrier forbids first mutation", lambda: begin_create(root))
        for fields in ({"watch_account_id": "account-B"}, {"watch_ready_utc": stamp(clock[0] - 31)},
                       {"watch_ready_utc": stamp(clock[0] + 1)}, {"watch_ready_utc": None}):
            candidate, _ = fixture()
            update(candidate, **fields)
            rejects("unverified watcher " + str(fields), lambda: begin_create(candidate))
        root, state = fixture()
        begin_create(root)
        halt(root, "stop during mutation")
        state = observe_identity(root, sample_pod(state))
        state = finish_create(root, bind_pod(state, sample_pod(state)))
        passed("create response cannot overwrite concurrent halt",
               state["status"] == "terminating" and state["halt_requested"])
        fail_create(root)
        passed("post-response exception preserves resolved identity",
               io._json(root / "LEASE.json")["mutation_phase"] == "observed")

        root, state = fixture(phase="uncertain")
        with patch.object(sys.modules[__name__], "checked_account", return_value=live), \
                patch.object(sys.modules[__name__], "pods", return_value=[]), \
                patch.object(api, "terminate", forbidden):
            passed("unknown create remains unresolved despite repeated empty lists", terminate_owned(root, "uncertain") is False)
            passed("unknown create has no premature termination receipt", io._json(root / "LEASE.json")["status"] != "terminated")
            clock[0] = epoch(state["provider_deadline_utc"]) + 1
            with locked(root, ".mutation.lock"):
                passed("suspended creator prevents a final absence receipt", confirm_absence(root, 3, "account-A") is False)
            passed("unknown create closes only after TTL and verified absences", terminate_owned(root, "TTL passed") is True)
            fail_create(root)
            passed("late exception preserves terminated status", io._json(root / "LEASE.json")["status"] == "terminated")

        root, state = fixture(phase="uncertain")
        deleted = []
        late = sample_pod(state, costPerHr=0.70)
        with patch.object(sys.modules[__name__], "checked_account", return_value=live), \
                patch.object(sys.modules[__name__], "pods", side_effect=[[], [], [], [late], [], [], []]) as listings, \
                patch.object(api, "terminate", side_effect=lambda pod_id: deleted.append(pod_id)):
            passed("late visible creation is adopted and deleted", terminate_owned(root, "late create") is True and deleted == ["pod-A"])
            passed("three previsibility absences cannot hide late creation", listings.call_count == 7)
            passed("rejected higher price is still charged in upper bound", io._json(root / "LEASE.json")["all_in_rate"] == 0.70 + STORAGE_RATE)

        root, state = fixture()
        with patch.object(sys.modules[__name__], "account", side_effect=lambda write_key=False: {**live, "id": "account-B" if write_key else "account-A"}), \
                patch.object(sys.modules[__name__], "pods", forbidden), patch.object(api, "terminate", forbidden):
            passed("wrong write account never verifies absence or deletes", terminate_owned(root, "wrong account") is False)
        root, state = fixture()
        sequence = [live, live, ValueError("account changed"), live, live, live]
        with patch.object(sys.modules[__name__], "checked_account", side_effect=sequence) as accounts, \
                patch.object(sys.modules[__name__], "pods", return_value=[]), patch.object(api, "terminate", forbidden):
            passed("failed account check resets consecutive absences", terminate_owned(root, "reset") is True and accounts.call_count == 6)
        root, state = fixture()
        valid_empty = {"myself": {"id": "account-A", "pods": []}}
        malformed = {"myself": {"id": "account-A", "pods": [{}]}}
        with patch.object(sys.modules[__name__], "checked_account", return_value=live), \
                patch.object(api, "gql", side_effect=[valid_empty, valid_empty, malformed, valid_empty, valid_empty, valid_empty]) as reads:
            passed("malformed listing resets consecutive absences", terminate_owned(root, "malformed listing") is True and reads.call_count == 6)
        root, state = fixture(phase="unknown")
        update(root, halt_requested=True, provider_deadline_utc=stamp(clock[0] - 1))
        passed("unrecognized mutation phase cannot produce an absence receipt", confirm_absence(root, 3, "account-A") is False)

        root, state = fixture(ready=False)
        with patch.object(sys.modules[__name__], "checked_account", side_effect=ValueError("wrong account")):
            rejects("watcher cannot advertise readiness before account proof", lambda: watch(SimpleNamespace(root=root)))
        passed("failed readiness leaves no ready timestamp", "watch_ready_utc" not in io._json(root / "LEASE.json"))
        with patch.object(api, "KEY_FILE", base / "different-local-path"):
            rejects("watcher rejects a different credential path before account access", lambda: watch(SimpleNamespace(root=root)))

        def startup_case(label, *, delay=None, succeeds=False, limit=WATCHER_STARTUP_SECONDS,
                         ready_account="account-A", ready_age=0, active_code=0, active_text="active\n"):
            candidate, initial = fixture(ready=False)
            initial = update(candidate, controller_sources={name: io._record(HERE / name) for name in
                             ("lease.py", "run_refits.py", "runpod_api.py", "pod_entry.sh")})
            started = clock[0]
            calls = {"starts": 0, "active_checks": 0, "ready": False}
            def startup_run(command, **kwargs):
                if command[0] == "systemd-run":
                    calls["starts"] += 1
                    return SimpleNamespace(returncode=0, stdout="")
                if command[0] == "systemctl":
                    calls["active_checks"] += 1
                    return SimpleNamespace(returncode=active_code, stdout=active_text)
                return forbidden()
            def startup_sleep(seconds):
                sleep(seconds)
                if delay is not None and clock[0] - started >= delay and not calls["ready"]:
                    update(candidate, watch_ready_utc=stamp(clock[0] - ready_age), watch_account_id=ready_account)
                    calls["ready"] = True
            with patch.object(sys.modules[__name__], "WATCHER_STARTUP_SECONDS", limit), \
                    patch.object(subprocess, "run", startup_run), patch.object(time, "sleep", startup_sleep):
                if succeeds:
                    arm_watcher(candidate, initial)
                    passed(label + " accepts delayed verified readiness", clock[0] - started == delay)
                else:
                    rejects(label + " cannot arm the creator", lambda: arm_watcher(candidate, initial))
                    passed(label + " exits at the bounded startup wait", clock[0] - started == limit)
            final = io._json(candidate / "LEASE.json")
            passed(label + " starts one watcher and leaves deployment unstarted",
                   calls["starts"] == 1 and final["mutation_phase"] == "not_started" and final["pod_id"] is None)
            return candidate, started, calls

        startup_case("readiness after 30 seconds", delay=30, succeeds=True)
        _, started, _ = startup_case("observed readiness after 123 seconds", delay=123, succeeds=True)
        rejects("delayed readiness cannot extend the 300-second user notice", lambda: validate_notice(stamp(started - 178)))
        startup_case("old 20-second wait with a 123-second acknowledgement", delay=123, limit=20)
        startup_case("no acknowledgement for 180 seconds")
        for label, options in (("stale acknowledgement", {"ready_age": 30}),
                               ("wrong-account acknowledgement", {"ready_account": "account-B"}),
                               ("inactive unit exit code", {"active_code": 3}),
                               ("inactive unit status text", {"active_text": "inactive\n"})):
            _, _, calls = startup_case(label, delay=1, **options)
            passed(label + " retains account freshness and active-unit gates",
                   calls["active_checks"] == (30 if label.startswith("inactive") else 0))

        root, state = fixture(ready=False)
        halt(root, "creator startup failed; operator closed the unstarted intent")
        def close_during_account_read(stale):
            sleep(123)
            passed("halted unstarted intent closes while a late watcher verifies its account",
                   confirm_absence(root, 3, "account-A"))
            return live
        with patch.object(sys.modules[__name__], "checked_account", close_during_account_read), \
                patch.object(sys.modules[__name__], "pods", forbidden), \
                patch.object(api, "terminate", forbidden):
            watch(SimpleNamespace(root=root))
        closed = io._json(root / "LEASE.json")
        passed("late readiness preserves termination halt zero spending and an unstarted deployment",
               closed["status"] == "terminated" and closed["halt_requested"]
               and closed["mutation_phase"] == "not_started" and closed["pod_id"] is None
               and closed["lease_spend_upper_usd"] == 0.0 and closed["absence_confirmations"] == 3)
        rejects("late watcher readiness cannot reopen the closed creator", lambda: begin_create(root))
        closed_record = io._record(root / "LEASE.json")
        with patch.object(sys.modules[__name__], "checked_account", forbidden):
            watch(SimpleNamespace(root=root))
        passed("restarted watcher leaves the closed intent untouched", io._record(root / "LEASE.json") == closed_record)
        class StopWatch(BaseException):
            pass
        root, state = fixture(phase="uncertain")
        halt(root, "watch reconciliation fixture")
        iterations = [0]
        def interrupt_sleep(seconds):
            sleep(seconds)
            if seconds == 15:
                iterations[0] += 1
                if iterations[0] == 2:
                    raise StopWatch()
        with patch.object(sys.modules[__name__], "checked_account", return_value=live), \
                patch.object(sys.modules[__name__], "pods", return_value=[]), \
                patch.object(sys.modules[__name__], "remote_status", return_value=None), \
                patch.object(time, "sleep", interrupt_sleep):
            try:
                watch(SimpleNamespace(root=root))
            except StopWatch:
                passed("watch continues after unresolved termination", iterations[0] == 2)
            else:
                raise AssertionError("watch exited while creation remained unresolved")

        # Exercise the real create/arm code with one wholly fake account and provider.
        class Provider:
            def __init__(self, halt_on_arm=False, uncertain=False, rejected=False):
                self.requests, self.commands, self.live_pod = [], [], None
                self.halt_on_arm, self.uncertain = halt_on_arm, uncertain
                self.rejected = rejected
            def gql(self, query, variables=None, write=False):
                if "gpuTypes" in query:
                    return json.loads(json.dumps(quote))
                if "podFindAndDeployOnDemand" in query:
                    self.requests.append(variables["input"])
                    if self.rejected:
                        raise supply_error()
                    self.live_pod = sample_pod({"name": variables["input"]["name"]})
                    if self.uncertain:
                        raise api.APIError("fake lost deployment response")
                    return {"podFindAndDeployOnDemand": self.live_pod}
                if "podTerminate" in query:
                    self.live_pod = None
                    return {"podTerminate": None}
                if "pods {" in query:
                    return {"myself": {"id": "account-A", "pods": [self.live_pod] if self.live_pod else []}}
                return {"myself": live.copy()}
            def run(self, command, **kwargs):
                self.commands.append(command)
                if command[0] == "ssh-keygen":
                    return SimpleNamespace(returncode=0, stdout=public.read_text())
                if command[0] == "systemd-run":
                    candidate = Path(command[command.index("--root") + 1])
                    update(candidate, watch_ready_utc=stamp(), watch_account_id="account-A")
                    if self.halt_on_arm:
                        halt(candidate, "fake pre-mutation halt")
                    return SimpleNamespace(returncode=0, stdout="")
                if command[0] == "systemctl":
                    return SimpleNamespace(returncode=0, stdout="active\n")
                return forbidden()
        # A separate temporary canonical ledger makes each independent attempt eligible.
        for label, provider in (("normal", Provider()), ("halted", Provider(halt_on_arm=True)),
                                ("uncertain", Provider(uncertain=True)), ("rejected", Provider(rejected=True))):
            with patch.object(sys.modules[__name__], "LEDGER", base / label), \
                    patch.object(api, "gql", provider.gql), patch.object(subprocess, "run", provider.run), \
                    redirect_stdout(text_io.StringIO()):
                args = SimpleNamespace(root=base / label / "lease", bundle=bundle, notice_utc=stamp(),
                                       public_key=public, ssh_identity=private)
                if label == "normal":
                    previous_records = {}
                    for gpu_name, spent in (("3090", 0.75), ("4090", 1.25)):
                        previous = io._mkdir(base / label / ("previous_" + gpu_name)) / "LEASE.json"
                        io._new_json(previous, {"status": "terminated", "gpu_type_id": "NVIDIA GeForce RTX " + gpu_name,
                                               "lease_spend_upper_usd": spent})
                        previous_records[previous] = io._record(previous)
                    create(args)
                    created = io._json(args.root / "LEASE.json")
                    passed("full creation uses exactly one mutation", len(provider.requests) == 1 and created["status"] == "running")
                    passed("canonical ledger charges all previous attempts", created["prior_spend_upper_usd"] == 2.0)
                    passed("selected 5090 keeps the same combined 25 cap after both historical GPU attempts",
                           created["gpu_type_id"] == GPU and created["combined_cap_usd"] == 25.0
                           and created["all_in_rate"] == 0.69 + STORAGE_RATE
                           and created["prior_spend_upper_usd"] + (
                               epoch(created["provider_deadline_utc"]) - epoch(created["billing_start_utc"])
                           ) / 3600 * created["all_in_rate"] <= 25.0 - 0.15)
                    passed("creating a 5090 lease never rewrites closed historical ledger files",
                           all(io._record(path) == record for path, record in previous_records.items()))
                    second = SimpleNamespace(**{**vars(args), "root": base / label / "second"})
                    rejects("a second root cannot bypass an unresolved lease", lambda: create(second))
                    passed("unresolved sibling prevents another mutation", len(provider.requests) == 1)
                    request = provider.requests[0]
                    passed("provider request selects exactly one 5090 with the quoted RAM CPU and disk floors",
                           request["gpuTypeId"] == "NVIDIA GeForce RTX 5090" and request["gpuCount"] == 1
                           and request["minMemoryInGb"] == 64 and request["minVcpuCount"] == 8
                           and request["containerDiskInGb"] == 120 and request["volumeInGb"] == 0)
                    passed("provider request binds the hard-rate UTC deletion deadline", request["terminateAfter"] == created["provider_deadline_utc"])
                    passed("Pod receives only the SSH public key and approved runtime environment",
                           {item["key"] for item in request["env"]} == {"PUBLIC_KEY", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "JLENS_LEASE_NAME"})
                    unit = next(command for command in provider.commands if command[0] == "systemd-run")
                    passed("detached watcher receives the creator's explicit local key-file path",
                           unit[unit.index("--key-file") + 1] == str(key_path))
                else:
                    rejects("full create " + label, lambda: create(args))
                    passed("halt and uncertain creation are never retried", len(provider.requests) == (0 if label == "halted" else 1))
                    if label == "rejected":
                        rejected = io._json(args.root / "LEASE.json")
                        passed("automatic create catch seals rejection without ever binding a pod",
                               rejected["mutation_phase"] == "rejected" and rejected["pod_id"] is None
                               and rejected["machine_id"] is None and verified_rejection(rejected)
                               and provider.live_pod is None)
                passed("full lifecycle verifies deletion or cancelled intent", terminate_owned(args.root, "offline cleanup") is True)

        for label in ("exact", "extra", "changed", "missing", "linked", "remote_changed"):
            root, state = fixture()
            destination = io._mkdir(root / "retrieved")
            payload = destination / "artifact.json"
            payload.write_text("frozen bytes\n")
            inventory = local_inventory(destination)
            if label == "extra":
                (destination / "stale.partial").write_text("extra")
            elif label == "changed":
                payload.write_text("changed bytes\n")
            elif label == "missing":
                payload.unlink()
            elif label == "linked":
                (destination / "linked-directory").symlink_to(base, target_is_directory=True)
            inventories = [inventory, {} if label == "remote_changed" else inventory]
            def retrieve(command, **kwargs):
                if command[0] == "rsync":
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=0, stdout=json.dumps(inventories.pop(0)))
            with patch.object(sys.modules[__name__], "remote_status", return_value={"phase": "complete", "lease_name": state["name"]}), \
                    patch.object(subprocess, "run", retrieve):
                if label == "exact":
                    _sync_results(root)
                    passed("exact retrieval produces a bound receipt", io._json(root / "RETRIEVAL_VERIFIED.json")["files"] == inventory)
                else:
                    rejects("retrieval rejects " + label, lambda: _sync_results(root))
                    passed("invalid retrieval has no receipt " + label, not (root / "RETRIEVAL_VERIFIED.json").exists())

        # Keep transport tests independent of the globally blocked network function.
        import urllib.request
        for url in ("http://api.runpod.io/graphql", "https://evil.example/graphql",
                    "https://api.runpod.io@evil.example/graphql", "https://api.runpod.io/other"):
            rejects("credential origin " + url, lambda url=url: transport("GET", url))
        rejects("all API redirects refused", lambda: api._NoRedirect().redirect_request(
            urllib.request.Request(api.GQL), None, 302, "redirect", {}, "https://evil.example/"))

    return {"schema_version": 1, "status": "passed", "cases": cases, "case_count": len(cases),
            "execution": "offline CPU; fake clocks, provider, credentials, subprocesses, and temporary ledgers",
            "controller_sources": {name: io._record(HERE / name) for name in ("lease.py", "runpod_api.py", "run_refits.py", "pod_entry.sh")},
            "provider_ttl_field": {"field": "terminateAfter", "type": "DateTime",
                                   "source": "https://graphql-spec.runpod.io/#definition-PodFindAndDeployOnDemandInput",
                                   "checked_date": "2026-09-08"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    plan = sub.add_parser("plan")
    plan.add_argument("--bundle", type=Path, required=True)
    plan.add_argument("--balance", type=float, default=BUDGET)
    plan.add_argument("--prior-spend", type=float, default=0.0)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--bundle", type=Path, required=True)
    create_parser.add_argument("--root", type=Path, required=True)
    create_parser.add_argument("--notice-utc", required=True)
    create_parser.add_argument("--public-key", type=Path, required=True)
    create_parser.add_argument("--ssh-identity", type=Path, required=True)
    create_parser.add_argument("--key-file", type=Path, default=api.KEY_FILE)
    rejection = sub.add_parser("reconcile-rejection")
    rejection.add_argument("--root", type=Path, required=True)
    rejection.add_argument("--capture", type=Path, required=True)
    rejection.add_argument("--key-file", type=Path, default=api.KEY_FILE)
    for name in ("watch", "status", "terminate", "sync"):
        child = sub.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name in ("watch", "terminate"):
            child.add_argument("--key-file", type=Path, default=api.KEY_FILE)
    args = parser.parse_args(argv)
    if hasattr(args, "root"):
        args.root = lease_root(args.root)
    if hasattr(args, "key_file"):
        api.KEY_FILE = io._no_links(args.key_file)
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
    elif args.command == "plan":
        print(json.dumps(make_plan(args.bundle, None, balance=args.balance, rate=MAX_GPU_RATE,
                                   prior_spend=args.prior_spend), indent=2))
    elif args.command == "create":
        create(args)
    elif args.command == "reconcile-rejection":
        state = reconcile_rejection(args)
        print(json.dumps({key: state[key] for key in ("name", "status", "mutation_phase", "absence_confirmations",
                                                     "lease_spend_upper_usd", "combined_spend_upper_usd")}, indent=2))
    elif args.command == "watch":
        watch(args)
    elif args.command == "terminate":
        if not terminate_owned(args.root, "operator requested termination"):
            raise RuntimeError("termination remains unresolved; the detached watcher must keep reconciling")
    elif args.command == "sync":
        sync_results(args.root)
    else:
        print(json.dumps(io._json(args.root / "LEASE.json"), indent=2))


if __name__ == "__main__":
    main()
