"""Upload the leased source bundle once and acknowledge a detached worker.

``--plan`` reads local evidence only; ``--self-test`` uses temporary files and
mocked SSH/rsync. Execution requires ``--run``. No provider credential is read
or sent by this program unless ``--check-account`` explicitly requests the
local account identity check. A failed or uncertain launch retains its intent
and is never retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

import lease
import run_refits as io


REMOTE = "/workspace/jlens"
WORKER = "jacobian-lens/research/refit_round_2026-09-07/deployment/cloud_worker.py"
SPEC = "jacobian-lens/research/refit_round_2026-09-07/deployment/run_spec.json"
WORKER_KEYS = ("name", "billing_start_utc", "work_deadline_utc", "all_in_rate", "prior_spend_upper_usd")
IDENTITY_KEYS = (
    "name", "pod_id", "machine_id", "account_id", "watch_account_id", "ssh", "ssh_identity", "api_key_file",
    "bundle", "bundle_record", "billing_start_utc", "setup_deadline_utc", "work_deadline_utc",
    "watch_deadline_utc", "provider_deadline_utc", "all_in_rate", "prior_spend_upper_usd",
)


def validate_archive(path, expected):
    """Read a bounded, sealed, regular-file source archive without extracting.

    This function is self-contained so the identical validator can be sent to
    the fresh Python image before any source from the archive is executed.
    """
    import gzip
    import hashlib
    import json
    from pathlib import Path, PurePosixPath
    import re
    import stat
    import tarfile

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def decode(data):
        return json.loads(data, object_pairs_hook=object_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))

    def canonical_name(name):
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
                or relative.is_absolute() or relative.as_posix() != name
                or any(part in (".", "..", "") for part in name.split("/"))):
            raise ValueError("noncanonical archive member")
        if name != "MANIFEST.json" and not name.startswith(("jacobian-lens/", "ouro_project/src/")):
            raise ValueError("archive member is outside the source roots")
        lowered = [part.lower() for part in relative.parts]
        if (any(part in (".git", ".ssh", ".env", "credentials", "credential", "secrets") for part in lowered)
                or any(part.startswith(("id_rsa", "id_ed25519")) for part in lowered)
                or relative.suffix.lower() in (".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pem", ".key")):
            raise ValueError("source archive contains a model, credential, or private-key path")
        return relative

    path = Path(path)
    if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int or not 1 <= expected["bytes"] <= 64 * 1024 * 1024
            or not isinstance(expected["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
        raise ValueError("invalid or oversized archive record")
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("linked archive path")
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size != expected["bytes"]:
        raise ValueError("archive is not a regular file")
    with path.open("rb") as handle:
        raw = handle.read(expected["bytes"] + 1)
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("archive bytes differ from the lease")
    import io
    # Bound gzip expansion before tarfile interprets PAX/long-name metadata,
    # whose allocations otherwise precede the per-member checks below.
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as compressed:
        expanded = compressed.read(272 * 1024 * 1024 + 1)
    if len(expanded) > 272 * 1024 * 1024:
        raise ValueError("archive expands beyond the complete tar limit")
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:", ignore_zeros=True) as archive:
        for member in archive:
            canonical_name(member.name)
            if (not member.isfile() or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or member.name in files or len(files) >= 4096
                    or not 0 <= member.size <= 16 * 1024 * 1024):
                raise ValueError("nonregular, duplicate, excessive, or oversized archive member")
            total += member.size
            if total > 256 * 1024 * 1024:
                raise ValueError("archive expands beyond the source limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("archive member has no payload")
            data = stream.read(member.size + 1)
            if len(data) != member.size:
                raise ValueError("archive member is truncated")
            files[member.name] = data
    if "MANIFEST.json" not in files:
        raise ValueError("archive manifest is missing")
    manifest = decode(files["MANIFEST.json"])
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("files"), dict)
            or set(files) != set(manifest["files"]) | {"MANIFEST.json"}
            or "MANIFEST.json" in manifest["files"]):
        raise ValueError("archive members do not exactly match the manifest")
    for name, record in manifest["files"].items():
        canonical_name(name)
        if (not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int or record["bytes"] != len(files[name])
                or record["sha256"] != hashlib.sha256(files[name]).hexdigest()):
            raise ValueError("archive member differs from its manifest")
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise ValueError("archive has a file/directory name collision")
    worker = "jacobian-lens/research/refit_round_2026-09-07/deployment/cloud_worker.py"
    specification = "jacobian-lens/research/refit_round_2026-09-07/deployment/run_spec.json"
    if (worker not in files or specification not in files
            or manifest.get("run_spec_sha256") != hashlib.sha256(files[specification]).hexdigest()):
        raise ValueError("worker or sealed run specification is missing")
    return manifest, files


def remote_stage(request, *, remote_root="/workspace/jlens", popen=None):
    """Fresh-image, standard-library staging endpoint; override only in tests."""
    from datetime import datetime, timezone
    import json
    import math
    import os
    from pathlib import Path
    import re
    import stat
    import subprocess
    import sys
    import time

    def no_links(path):
        path = Path(os.path.abspath(path))
        cursor = Path(path.anchor)
        for part in path.parts[1:]:
            cursor /= part
            try:
                mode = cursor.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise ValueError("linked remote staging path")
        return path

    def sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_new(path, data):
        no_links(path)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)
        sync_directory(path.parent)

    def encoded(value):
        return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()

    def epoch(value):
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
            raise ValueError("invalid UTC deadline")
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()

    if (not isinstance(request, dict) or set(request) != {"schema_version", "action", "stage_id", "lease", "bundle_record"}
            or request["schema_version"] != 1 or request["action"] not in ("prepare", "launch")
            or not isinstance(request["stage_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", request["stage_id"])):
        raise ValueError("invalid staging request")
    config = request["lease"]
    expected = request["bundle_record"]
    if (not isinstance(config, dict)
            or set(config) != {"name", "billing_start_utc", "work_deadline_utc", "all_in_rate", "prior_spend_upper_usd"}
            or not isinstance(config["name"], str) or not re.fullmatch(r"jlens-refit-[0-9a-f]{32}", config["name"])):
        raise ValueError("invalid worker lease configuration")
    for key in ("all_in_rate", "prior_spend_upper_usd"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("invalid worker budget")
    now = time.time()
    if not epoch(config["billing_start_utc"]) <= now < epoch(config["work_deadline_utc"]):
        raise ValueError("lease is outside its work interval")
    if config["all_in_rate"] <= 0 or config["prior_spend_upper_usd"] >= 25:
        raise ValueError("worker budget is exhausted")
    if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int or not 1 <= expected["bytes"] <= 64 * 1024 * 1024
            or not isinstance(expected["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
        raise ValueError("invalid archive record")
    root = no_links(remote_root)
    marker = no_links(root / "provider_lease_name")
    if not stat.S_ISREG(marker.lstat().st_mode) or marker.read_text().strip() != config["name"]:
        raise ValueError("remote host does not carry the expected provider lease marker")
    for name in ("STOP", "status.json", "bundle", "WORKER_LAUNCH.json", "WORKER_STARTED.json", "worker.log", "worker_lease.json"):
        path = no_links(root / name)
        if path.exists():
            raise FileExistsError("remote staging or execution already exists")
    incoming = no_links(root / "incoming")
    intent = no_links(root / "STAGE_INTENT.json")
    claim = {key: value for key, value in request.items() if key != "action"}
    archive_path = incoming / ("bundle-" + expected["sha256"] + ".tar.gz")
    if request["action"] == "prepare":
        if incoming.exists() or intent.exists():
            raise FileExistsError("remote staging was already claimed")
        write_new(intent, encoded(claim))
        incoming.mkdir(mode=0o700)
        sync_directory(root)
        return {"schema_version": 1, "status": "prepared", "stage_id": request["stage_id"],
                "lease_name": config["name"], "archive_path": str(archive_path), "bundle_record": expected}
    if (not incoming.is_dir() or not stat.S_ISREG(intent.lstat().st_mode)
            or json.loads(intent.read_bytes()) != claim
            or sorted(path.name for path in incoming.iterdir()) != [archive_path.name]):
        raise ValueError("remote upload does not match the staging claim")
    manifest, files = validate_archive(archive_path, expected)
    destination = no_links(root / "bundle")
    destination.mkdir(mode=0o700)
    for name, data in sorted(files.items()):
        output = no_links(destination / name)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_new(output, data)
        output.chmod(0o644)
    for directory, _, _ in os.walk(destination, topdown=False):
        sync_directory(directory)
    configuration = root / "worker_lease.json"
    write_new(configuration, encoded(config))
    if (no_links(root / "STOP").exists() or no_links(root / "status.json").exists()
            or time.time() >= epoch(config["work_deadline_utc"])):
        raise ValueError("worker stop, status, or deadline arrived during extraction")
    launch = {"schema_version": 1, "status": "launch_intent", "stage_id": request["stage_id"],
              "lease_name": config["name"], "bundle_record": expected,
              "run_spec_sha256": manifest["run_spec_sha256"], "source_files": len(manifest["files"])}
    # Write before Popen: interruption in this interval must not launch twice.
    write_new(root / "WORKER_LAUNCH.json", encoded(launch))
    worker = destination / "jacobian-lens/research/refit_round_2026-09-07/deployment/cloud_worker.py"
    environment = dict(os.environ)
    for key in ("RUNPOD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PUBLIC_KEY", "SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        environment.pop(key, None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       PYTHONPATH=os.pathsep.join((str(destination / "jacobian-lens"),
                                                 str(destination / "ouro_project/src"), str(worker.parent.parent))))
    with (root / "worker.log").open("xb") as log:
        process = (subprocess.Popen if popen is None else popen)(
            [sys.executable, str(worker), "--root", str(root), "--lease", str(configuration)],
            cwd=str(worker.parent), env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    if type(process.pid) is not int or process.pid <= 0:
        raise ValueError("worker launch returned no process identifier")
    receipt = {**launch, "status": "launch_acknowledged", "pid": process.pid,
               "acknowledged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_new(root / "WORKER_STARTED.json", encoded(receipt))
    return receipt


def load_state(root, *, now=None):
    now = time.time() if now is None else now
    state = io._json(io._no_links(root / "LEASE.json"))
    if (state.get("schema_version") != 1 or state.get("status") != "running"
            or state.get("mutation_phase") != "observed"
            or state.get("halt_requested") is not False
            or not isinstance(state.get("name"), str) or not re.fullmatch(r"jlens-refit-[0-9a-f]{32}", state["name"])
            or not isinstance(state.get("account_id"), str) or not state["account_id"]
            or state.get("watch_account_id") != state["account_id"]):
        raise ValueError("lease is not running with its bound watcher")
    for name in ("pod_id", "machine_id"):
        if not isinstance(state.get(name), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", state[name]):
            raise ValueError("lease has no valid pod and machine identity")
    if not 0 <= now - lease.epoch(state["watch_heartbeat_utc"]) <= 90:
        raise ValueError("lease watcher heartbeat is stale or in the future")
    if not lease.epoch(state["billing_start_utc"]) <= now < lease.epoch(state["setup_deadline_utc"]):
        raise ValueError("lease is outside its setup interval")
    if not (lease.epoch(state["setup_deadline_utc"]) <= lease.epoch(state["work_deadline_utc"])
            < lease.epoch(state["watch_deadline_utc"]) < lease.epoch(state["provider_deadline_utc"])):
        raise ValueError("lease deadlines are not ordered")
    rate = lease.number(state["all_in_rate"], "all-in rate", minimum=0.001)
    prior = lease.number(state["prior_spend_upper_usd"], "previous experiment spending")
    if prior + (now - lease.epoch(state["billing_start_utc"])) / 3600 * rate >= 25:
        raise ValueError("combined experiment budget is exhausted")
    address = state["ssh"]
    if not isinstance(address, dict) or set(address) != {"host", "port"}:
        raise ValueError("SSH endpoint is missing")
    ipaddress.ip_address(address["host"])
    if type(address["port"]) is not int or not 1 <= address["port"] <= 65535:
        raise ValueError("SSH port is invalid")
    private = io._no_links(state["ssh_identity"])
    if not private.is_file() or private.stat().st_mode & 0o077:
        raise ValueError("SSH identity is missing or accessible to other users")
    for key in IDENTITY_KEYS:
        state[key]
    return state


def request_for(state, stage_id, action):
    return {"schema_version": 1, "action": action, "stage_id": stage_id,
            "lease": {key: state[key] for key in WORKER_KEYS}, "bundle_record": state["bundle_record"]}


def checked_local_account(state):
    """Use the creator's explicit credential path for the optional local read."""
    previous = lease.api.KEY_FILE
    try:
        lease.api.KEY_FILE = io._no_links(state["api_key_file"])
        return lease.checked_account(state)
    finally:
        lease.api.KEY_FILE = previous


def remote_call(root, state, request, *, run=subprocess.run):
    program = (inspect.getsource(validate_archive) + "\n" + inspect.getsource(remote_stage)
               + "\nimport json, sys\nrequest = json.loads(sys.stdin.read(65537))\n"
               + "print(json.dumps(remote_stage(request), sort_keys=True, allow_nan=False))\n")
    command = "python -c " + shlex.quote(program)
    result = run([*lease.ssh_options(root, state), "root@" + state["ssh"]["host"], command],
                 input=json.dumps(request, allow_nan=False), text=True, capture_output=True,
                 timeout=180, check=True)
    if len(result.stdout) > 65536:
        raise ValueError("remote acknowledgment is oversized")
    value = json.loads(result.stdout)
    if (not isinstance(value, dict) or value.get("stage_id") != request["stage_id"]
            or value.get("lease_name") != state["name"] or value.get("bundle_record") != state["bundle_record"]):
        raise ValueError("remote acknowledgment does not identify the staging request")
    return value


def stage(root, *, plan=False, check_account=False, run=subprocess.run):
    """All external actions are injected in CPU tests; no create/delete API."""
    root = lease.lease_root(root)
    state = load_state(root)
    archive = io._no_links(state["bundle"])
    manifest, files = validate_archive(archive, state["bundle_record"])
    del files
    expected_path = REMOTE + "/incoming/bundle-" + state["bundle_record"]["sha256"] + ".tar.gz"
    overview = {"schema_version": 1, "lease_name": state["name"], "bundle_record": state["bundle_record"],
                "run_spec_sha256": manifest["run_spec_sha256"], "source_files": len(manifest["files"]),
                "remote_archive": expected_path, "worker_lease_keys": list(WORKER_KEYS)}
    if plan:
        return {**overview, "status": "planned", "external_actions": 0, "local_writes": 0}
    with lease.locked(root, ".stage.lock"):
        initial = {key: state[key] for key in IDENTITY_KEYS}
        state = load_state(root)
        if {key: state[key] for key in IDENTITY_KEYS} != initial:
            raise ValueError("lease identity changed while staging was prepared")
        if check_account:
            checked_local_account(state)
        for name in ("STAGE_INTENT.json", "STAGED.json", "STAGE_FAILURE.json"):
            if io._no_links(root / name).exists():
                raise FileExistsError("staging has an existing intent or receipt; reconcile it before any further action")
        stage_id = uuid.uuid4().hex
        io._new_json(root / "STAGE_INTENT.json", {**overview, "status": "staging_intent", "stage_id": stage_id,
                                                "issued_utc": lease.stamp(), "pod_id": state["pod_id"],
                                                "machine_id": state["machine_id"]})
        phase = "prepare"
        try:
            prepared = remote_call(root, state, request_for(state, stage_id, "prepare"), run=run)
            if prepared.get("status") != "prepared" or prepared.get("archive_path") != expected_path:
                raise ValueError("remote preparation returned an unexpected upload path")
            phase = "upload"
            host = state["ssh"]["host"]
            destination_host = "[" + host + "]" if ":" in host else host
            run(["rsync", "--protect-args", "--partial", "--timeout=60", "-e", shlex.join(lease.ssh_options(root, state)),
                 "--", str(archive), "root@" + destination_host + ":" + expected_path],
                text=True, capture_output=True, timeout=600, check=True)
            # Watcher updates are permitted; identity, cost and deadlines cannot
            # silently change between the initial intent and the launch.
            current = load_state(root)
            if {key: current[key] for key in IDENTITY_KEYS} != initial:
                raise ValueError("lease identity, rate, or deadline changed during upload")
            if check_account:
                checked_local_account(current)
            phase = "launch"
            receipt = remote_call(root, current, request_for(current, stage_id, "launch"), run=run)
            if (receipt.get("status") != "launch_acknowledged" or type(receipt.get("pid")) is not int
                    or receipt["pid"] <= 0 or receipt.get("run_spec_sha256") != manifest["run_spec_sha256"]):
                raise ValueError("remote worker launch was not acknowledged")
            result = {**receipt, "pod_id": state["pod_id"], "machine_id": state["machine_id"],
                      "local_acknowledged_utc": lease.stamp(), "worker_progress_verified": False}
            io._new_json(root / "STAGED.json", result)
            return result
        except BaseException as error:
            io._new_json(root / "STAGE_FAILURE.json", {"schema_version": 1, "stage_id": stage_id,
                         "lease_name": state["name"], "phase": phase, "error_type": type(error).__name__,
                         "recorded_utc": lease.stamp(), "automatic_retry_allowed": False,
                         "launch_may_have_occurred": phase == "launch"})
            raise


def self_test():
    """Exercise real tar/JSON/filesystem logic and mocked transport only."""
    import io as memory_io
    import tarfile
    from types import SimpleNamespace
    from unittest.mock import patch

    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, action, exception=(ValueError, FileExistsError, KeyError)):
        try:
            action()
        except exception:
            checks.append(name)
        else:
            raise AssertionError(name)

    with tempfile.TemporaryDirectory(prefix="jlens-stage-cpu-") as temporary:
        base = Path(temporary)
        now = int(time.time())
        data = {WORKER: b"# CPU fixture; never executed\n", SPEC: b'{"schema_version": 1}\n',
                "jacobian-lens/jlens/__init__.py": b"# fixture\n"}

        def archive_at(path, *, additions=None, mutate_manifest=None, special=None):
            payload = {**data, **(additions or {})}
            manifest = {"schema_version": 1, "run_spec_sha256": hashlib.sha256(payload[SPEC]).hexdigest(),
                        "files": {name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                                  for name, value in payload.items()}}
            if mutate_manifest:
                mutate_manifest(manifest)
            with tarfile.open(path, "w:gz") as archive:
                for name, value in {**payload, "MANIFEST.json": json.dumps(manifest).encode()}.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(value)
                    archive.addfile(member, memory_io.BytesIO(value))
                if special:
                    archive.addfile(special, memory_io.BytesIO(b""))
            return io._record(path)

        source = base / "bundle with spaces.tar.gz"
        record = archive_at(source)
        manifest, extracted = validate_archive(source, record)
        check("valid source archive and exact manifest", len(extracted) == 4 and len(manifest["files"]) == 3)
        rejects("wrong archive digest", lambda: validate_archive(source, {**record, "sha256": "0" * 64}))
        linked = base / "linked.tar.gz"
        linked.symlink_to(source)
        rejects("linked archive rejected", lambda: validate_archive(linked, record))
        variants = (
            ("traversal", {"jacobian-lens/../escape.py": b"x"}, None),
            ("absolute", {"/escape.py": b"x"}, None),
            ("noncanonical", {"jacobian-lens//escape.py": b"x"}, None),
            ("weights", {"jacobian-lens/model.safetensors": b"x"}, None),
            ("private key", {"jacobian-lens/.ssh/id_ed25519": b"x"}, None),
            ("file parent collision", {"jacobian-lens/a": b"x", "jacobian-lens/a/b.py": b"x"}, None),
            ("manifest mismatch", None, lambda m: m["files"][WORKER].update(sha256="0" * 64)),
            ("manifest missing member", None, lambda m: m["files"].pop(WORKER)),
            ("run spec mismatch", None, lambda m: m.update(run_spec_sha256="0" * 64)),
        )
        for index, (name, additions, mutation) in enumerate(variants):
            path = base / f"invalid-{index}.tar.gz"
            expected = archive_at(path, additions=additions, mutate_manifest=mutation)
            rejects(name, lambda p=path, e=expected: validate_archive(p, e))
        for index, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE, tarfile.REGTYPE)):
            member = tarfile.TarInfo(WORKER if kind == tarfile.REGTYPE else "jacobian-lens/special")
            member.type, member.linkname = kind, WORKER
            path = base / f"special-{index}.tar.gz"
            expected = archive_at(path, special=member)
            rejects(f"special or duplicate member {kind!r}", lambda p=path, e=expected: validate_archive(p, e))

        ledger = base / "ledger"
        ledger.mkdir()
        private = base / "identity with spaces"
        private.write_text("CPU PRIVATE KEY SENTINEL, MUST NEVER BE READ OR UPLOADED")
        private.chmod(0o600)
        serial = 0

        def fixture():
            nonlocal serial
            serial += 1
            root = ledger / f"lease-{serial}"
            root.mkdir()
            state = {"schema_version": 1, "name": "jlens-refit-" + f"{serial:032x}",
                     "status": "running", "mutation_phase": "observed", "halt_requested": False,
                     "pod_id": "pod-A", "machine_id": "machine-A",
                     "account_id": "account-A", "watch_account_id": "account-A", "watch_heartbeat_utc": lease.stamp(now),
                     "billing_start_utc": lease.stamp(now - 10), "setup_deadline_utc": lease.stamp(now + 300),
                     "work_deadline_utc": lease.stamp(now + 900), "watch_deadline_utc": lease.stamp(now + 1200),
                     "provider_deadline_utc": lease.stamp(now + 1800), "all_in_rate": 0.36, "prior_spend_upper_usd": 0.0,
                     "ssh": {"host": "192.0.2.1", "port": 2200}, "ssh_identity": str(private),
                     "api_key_file": "/DO-NOT-READ/credential", "bundle": str(source), "bundle_record": record}
            io._new_json(root / "LEASE.json", state)
            remote = base / f"remote-{serial}"
            remote.mkdir()
            (remote / "results").mkdir()  # pod_entry.sh already creates this.
            (remote / "provider_lease_name").write_text(state["name"] + "\n")
            return root, state, remote

        def fake_popen(calls):
            def launch(argv, **kwargs):
                calls.append((argv, kwargs))
                check("detached worker and closed input", kwargs["start_new_session"] is True
                      and kwargs["close_fds"] is True and kwargs["stdin"] == subprocess.DEVNULL)
                check("worker logs outside results", Path(kwargs["stdout"].name).name == "worker.log"
                      and Path(kwargs["stdout"].name).parent.name.startswith("remote-"))
                check("worker environment excludes credentials", not any(key in kwargs["env"] for key in
                      ("RUNPOD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SSH_AUTH_SOCK")))
                return SimpleNamespace(pid=12345)
            return launch

        with patch.object(lease, "LEDGER", ledger):
            root, state, remote = fixture()
            before = sorted(root.iterdir())
            check("plan has no writes or external calls", stage(root, plan=True, run=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))["external_actions"] == 0
                  and sorted(root.iterdir()) == before)
            for name, change in (("halted lease", {"halt_requested": True}), ("terminated lease", {"status": "terminated"}),
                                 ("unobserved creation", {"mutation_phase": "in_flight"}),
                                 ("wrong watcher account", {"watch_account_id": "other"}),
                                 ("stale heartbeat", {"watch_heartbeat_utc": lease.stamp(now - 91)}),
                                 ("future heartbeat", {"watch_heartbeat_utc": lease.stamp(now + 20)}),
                                 ("expired setup", {"setup_deadline_utc": lease.stamp(now - 1)}),
                                 ("bad SSH port", {"ssh": {"host": "192.0.2.1", "port": True}})):
                io._atomic_json(root / "LEASE.json", {**state, **change})
                rejects(name, lambda: load_state(root, now=now))
            io._atomic_json(root / "LEASE.json", state)
            previous_key_path = lease.api.KEY_FILE
            account_paths = []
            with patch.object(lease, "checked_account", lambda _state: account_paths.append(lease.api.KEY_FILE)):
                checked_local_account(state)
            check("optional account check uses creator path and restores ambient path",
                  account_paths == [Path(state["api_key_file"])] and lease.api.KEY_FILE == previous_key_path)
            calls, launches = [], []

            def transport(argv, **kwargs):
                calls.append((argv, kwargs))
                if argv[0] == "ssh":
                    request = json.loads(kwargs["input"])
                    check("only five lease fields cross SSH", set(request["lease"]) == set(WORKER_KEYS)
                          and "api_key_file" not in kwargs["input"] and "ssh_identity" not in kwargs["input"])
                    # Execute the exact quoted bootstrap in a local namespace;
                    # the endpoint uses a temporary root and a mocked Popen.
                    words = shlex.split(argv[-1])
                    check("remote program has one safely quoted Python argument", words[:2] == ["python", "-c"] and len(words) == 3)
                    namespace = {}
                    exec(words[2].split("\nimport json, sys\n", 1)[0], namespace)
                    result = namespace["remote_stage"](request, remote_root=remote, popen=fake_popen(launches))
                    if result.get("archive_path"):
                        result["archive_path"] = REMOTE + "/incoming/" + Path(result["archive_path"]).name
                    return SimpleNamespace(stdout=json.dumps(result), returncode=0)
                check("only the sealed source archive is uploaded", argv[0] == "rsync" and argv[-2] == str(source)
                      and "--" in argv and "--protect-args" in argv and shlex.split(argv[argv.index("-e") + 1])[0] == "ssh")
                target = remote / "incoming" / Path(argv[-1].split(":", 1)[1]).name
                target.write_bytes(source.read_bytes())
                return SimpleNamespace(stdout="", returncode=0)

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "DO-NOT-FORWARD", "HF_TOKEN": "DO-NOT-FORWARD", "SSH_AUTH_SOCK": "/DO-NOT-FORWARD"}):
                result = stage(root, run=transport)
            check("complete mocked stage acknowledges one process", result["status"] == "launch_acknowledged"
                  and len(calls) == 3 and len(launches) == 1 and (root / "STAGED.json").is_file()
                  and result["worker_progress_verified"] is False)
            check("remote lease contains only the worker contract", set(io._json(remote / "worker_lease.json")) == set(WORKER_KEYS))
            check("extracted source bytes equal original", (remote / "bundle" / WORKER).read_bytes() == data[WORKER])
            rejects("local repeat never launches again", lambda: stage(root, run=transport))
            check("repeat made no transport calls", len(calls) == 3)

            root, state, remote = fixture()
            calls, launches = [], []
            normal_transport = transport

            def interrupted_transport(argv, **kwargs):
                response = normal_transport(argv, **kwargs)
                if argv[0] == "rsync":
                    io._atomic_json(root / "LEASE.json", {**state, "halt_requested": True})
                return response

            rejects("halt arriving during upload prevents launch", lambda: stage(root, run=interrupted_transport))
            check("halt preserves intent and records no launch", len(calls) == 2 and not launches
                  and (root / "STAGE_INTENT.json").exists() and io._json(root / "STAGE_FAILURE.json")["phase"] == "upload")

            root, state, remote = fixture()
            calls, launches = [], []

            def uncertain_transport(argv, **kwargs):
                response = normal_transport(argv, **kwargs)
                if argv[0] == "ssh" and json.loads(kwargs["input"])["action"] == "launch":
                    raise subprocess.TimeoutExpired("ssh", 180)
                return response

            rejects("uncertain acknowledgment is retained", lambda: stage(root, run=uncertain_transport), subprocess.TimeoutExpired)
            check("uncertain launch is never called again", len(launches) == 1
                  and io._json(root / "STAGE_FAILURE.json")["launch_may_have_occurred"] is True
                  and not (root / "STAGED.json").exists())
            rejects("uncertain launch blocks a later invocation", lambda: stage(root, run=uncertain_transport))
            check("one process after uncertain repeat", len(launches) == 1)

            root, state, remote = fixture()
            request = request_for(state, "f" * 32, "prepare")
            (remote / "provider_lease_name").write_text("another lease")
            rejects("wrong provider marker rejected", lambda: remote_stage(request, remote_root=remote))
            (remote / "provider_lease_name").write_text(state["name"])
            remote_stage(request, remote_root=remote)
            rejects("remote preparation cannot repeat", lambda: remote_stage(request, remote_root=remote))
            path = remote / "incoming" / ("bundle-" + record["sha256"] + ".tar.gz")
            path.write_bytes(source.read_bytes() + b"corrupt")
            rejects("remote corrupted upload rejected before extraction", lambda: remote_stage({**request, "action": "launch"}, remote_root=remote))
            check("corrupted upload created no bundle or process", not (remote / "bundle").exists()
                  and not (remote / "WORKER_LAUNCH.json").exists())

            root, state, remote = fixture()
            request = request_for(state, "e" * 32, "prepare")
            remote_stage(request, remote_root=remote)
            path = remote / "incoming" / ("bundle-" + record["sha256"] + ".tar.gz")
            path.write_bytes(source.read_bytes())

            def failed_popen(*_args, **_kwargs):
                raise OSError("mocked process launch failure")

            rejects("failed Popen leaves its durable launch intent", lambda: remote_stage(
                {**request, "action": "launch"}, remote_root=remote, popen=failed_popen), OSError)
            check("failed Popen cannot claim successful start", (remote / "WORKER_LAUNCH.json").is_file()
                  and not (remote / "WORKER_STARTED.json").exists())
            rejects("failed Popen cannot be retried", lambda: remote_stage({**request, "action": "launch"}, remote_root=remote))
            check("self-test imports no tensor runtime", "torch" not in sys.modules)

    return {"schema_version": 1, "status": "passed", "checks": len(checks), "passed": checks,
            "source_sha256": io._file_hash(Path(__file__)), "external_calls": 0, "gpu_operations": 0,
            "mocked_ssh_rsync": True, "real_worker_processes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="canonical local lease directory")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-account", action="store_true", help="explicit local read-only account identity check")
    parser.add_argument("--output", type=Path, help="exclusive local CPU-proof JSON output")
    args = parser.parse_args()
    if args.self_test:
        if args.root is not None or args.check_account:
            parser.error("self-test does not accept a lease or account check")
        result = self_test()
    else:
        if args.root is None or args.output is not None or (args.plan and args.check_account):
            parser.error("plan/run require --root; --output is for self-test and account checks require --run")
        result = stage(args.root, plan=args.plan, check_account=args.check_account)
    if args.output is not None:
        io._new_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
