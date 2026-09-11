"""Copy the completed Ouro evaluations during the interpretation wait; read-only remotely."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import stat
import subprocess
import sys


DEPLOYMENT = Path(__file__).resolve().parents[1] / "deployment"
LEASE_NAME = "jlens-refit-14ca9417eb6145c8979fb87ea9fde68c"
POD_ID = "vzwx0cj43g2uw5"
TREES = ("ouro_evaluation", "controls_evaluation")
CONTRACTS = ("run_spec.json", "combined_contract.json", "environment.json", "huginn_eligibility.json")
EXTRAS = (*CONTRACTS, "initial_budget_projection.json", *(
    "logs/" + name + ".log" for name in
    ("ouro_fits", "ouro_evaluation", "control_fits", "controls_evaluation")
))


def expected_paths():
    files = set(EXTRAS)
    for tree in TREES:
        files.update(f"{tree}/{name}.json" for name in ("OWNER", "COMPLETE"))
    files.update("ouro_evaluation/common/" + name for name in (
        "arrays.npz", "cache.pt", "items.json", "task_names.json", "token_forms.json",
        "metadata.json", "SEAL.json"))
    for index in range(1, 6):
        files.update(f"ouro_evaluation/fits/fit_{index:02d}/" + name
                     for name in ("arrays.npz", "metadata.json", "SEAL.json"))
    for section in ("penultimate", "positions/sampled_sum", "positions/diagonal"):
        files.update(f"controls_evaluation/{section}/" + name
                     for name in ("arrays.npz", "metadata.json", "SEAL.json"))
    files.update("controls_evaluation/comparisons/" + name for name in (
        "scores.npz", "paired_ranks.npz", "items.json", "task_names.json", "token_forms.json",
        "summary.json", "metadata.json", "SEAL.json"))
    return files


REMOTE_SCRIPT = r'''
from pathlib import Path
import hashlib, json, os, stat
config = CONFIG
workspace = Path(config['workspace'])
def regular(path, directory=False):
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('linked handoff path')
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('nonregular handoff path')
def status():
    path = workspace / 'status.json'
    regular(path)
    value = json.loads(path.read_text())
    stop = workspace / 'STOP'
    if (value.get('phase') != 'awaiting_ouro_interpretation'
            or value.get('lease_name') != config['lease_name']
            or value.get('stop_requested', False) is not False
            or stop.exists() or stop.is_symlink()):
        raise ValueError('worker is not in the active Ouro interpretation wait')
    return value
before = status()
root = workspace / 'results'
regular(root, directory=True)
paths = {root / name for name in config['extras']}
def walk_error(error): raise error
for name in config['trees']:
    tree = root / name
    regular(tree, directory=True)
    for directory, dirs, names in os.walk(tree, followlinks=False, onerror=walk_error):
        for child in dirs: regular(Path(directory) / child, directory=True)
        for child in names: paths.add(Path(directory) / child)
if {p.relative_to(root).as_posix() for p in paths} != set(config['expected']):
    raise ValueError('unexpected or missing handoff member')
files = {}
for path in sorted(paths):
    regular(path)
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
            size += len(block)
    files[path.relative_to(root).as_posix()] = {'bytes': size, 'sha256': h.hexdigest()}
after = status()
if before['required_bindings'] != after['required_bindings']:
    raise ValueError('interpretation bindings changed while hashing')
print(json.dumps({'status_before': before, 'status_after': after, 'files': files}, sort_keys=True))
'''


def remote_script(workspace="/workspace/jlens"):
    config = {"workspace": workspace, "lease_name": LEASE_NAME, "extras": EXTRAS,
              "trees": TREES, "expected": sorted(expected_paths())}
    return REMOTE_SCRIPT.replace("CONFIG", repr(config))


def verify_snapshot(snapshot, records):
    files = snapshot["files"]
    if set(files) != expected_paths():
        raise ValueError("remote handoff inventory differs from the fixed file list")
    expected = {"main_complete": files["ouro_evaluation/COMPLETE.json"],
                "controls_complete": files["controls_evaluation/COMPLETE.json"],
                "run_spec_sha256": records["run_spec.json"]["sha256"],
                "combined_contract_sha256": records["combined_contract.json"]["sha256"]}
    for key in ("status_before", "status_after"):
        status = snapshot[key]
        if (status.get("phase") != "awaiting_ouro_interpretation"
                or status.get("lease_name") != LEASE_NAME
                or status.get("stop_requested", False) is not False
                or status.get("required_bindings") != expected):
            raise ValueError("handoff status does not bind these results and local contracts")
    if any(files[name] != records[name] for name in CONTRACTS):
        raise ValueError("copied contract bytes differ from the local deployment")


def local_inventory(root):
    files = {}
    for path in root.rglob("*"):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("linked local handoff path")
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("nonregular local handoff file")
        h = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                h.update(block)
                size += len(block)
        files[path.relative_to(root).as_posix()] = {"bytes": size, "sha256": h.hexdigest()}
    return files


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(DEPLOYMENT))
    import lease

    root = lease.LEDGER / "attempt_06"
    state = lease.io._json(root / "LEASE.json")
    if state["name"] != LEASE_NAME or state["pod_id"] != POD_ID:
        raise ValueError("unexpected lease or pod")
    records = {name: lease.io._record(DEPLOYMENT / name) for name in CONTRACTS}
    command = [*lease.ssh_options(root, state), "root@" + state["ssh"]["host"],
               "python -c " + shlex.quote(remote_script())]

    def snapshot():
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        value = json.loads(result.stdout)
        verify_snapshot(value, records)
        return value

    before = snapshot()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = DEPLOYMENT.parent / "monitoring/attempt_06" / ("ouro_handoff_" + stamp)
    lease.io._no_links(destination)
    destination.mkdir(parents=True, exist_ok=False)
    lease.io._new_json(destination / "REMOTE_BEFORE.json", before)
    listing = destination / "FILES_FROM"
    with listing.open("xb") as handle:
        handle.write(b"\0".join(name.encode() for name in sorted(expected_paths())) + b"\0")
    copied = destination / "results"
    copied.mkdir()
    transport = shlex.join(lease.ssh_options(root, state))
    subprocess.run(["rsync", "-rlt", "--partial", "--timeout=60", "--from0",
                    "--files-from=" + str(listing), "-e", transport,
                    f"root@{state['ssh']['host']}:/workspace/jlens/results/", str(copied) + "/"],
                   check=True, timeout=900)
    after = snapshot()
    lease.io._new_json(destination / "REMOTE_AFTER.json", after)
    if before["files"] != after["files"] or local_inventory(copied) != before["files"]:
        raise ValueError("remote or copied handoff bytes changed or do not match")
    lease.io._new_json(destination / "COPY_VERIFIED.json", {
        "status": "passed", "lease_name": LEASE_NAME, "pod_id": POD_ID,
        "files": before["files"], "bindings": before["status_after"]["required_bindings"],
        "scope": "Interim copies of two complete Ouro evaluation trees, four frozen contracts, "
                 "initial budget record and four completed stage logs; every file hashed before "
                 "and after transfer and locally. No binary deserialization, scientific interpretation, "
                 "interpretation receipt publication, or final experiment retrieval.",
    })
    print(json.dumps({"directory": str(destination), "files": len(before["files"]),
                      "bytes": sum(row["bytes"] for row in before["files"].values()),
                      "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
