#!/usr/bin/env python3
"""Supervise the unchanged receiver under the existing PyTorch environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
ROUND = Path(__file__).resolve().parents[1]
PYTHON = '/home/moloch/ouro_project/venv/bin/python'
SOURCE_NAMES = {'lease.py', 'artifact_handoff.py', 'run_refits.py',
                'runpod_api.py', 'transport.py', 'pod_entry.sh'}
TRANSPORT_SHA256 = 'acec7a803383119706ee852d8c0b21c4380df0b91b16aa4c198462c6d7bd199a'
STABLE_KEYS = ('name', 'account_id', 'pod_id', 'machine_id', 'run_id',
               'controller_sources', 'semantic_verifier', 'run_spec_path',
               'run_spec_record', 'run_spec_sha256', 'output_contract_path',
               'output_contract_record', 'output_contract_sha256',
               'run_config_path', 'run_config_record', 'billing_start_utc',
               'setup_deadline_utc', 'work_deadline_utc', 'watch_deadline_utc',
               'provider_deadline_utc')


def no_links(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Linked receiver path: ' + str(part))
    return path


def read_file(path):
    path = no_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Receiver input is not a regular file: ' + str(path))
    return path.read_bytes()


def binding_digest(state):
    binding = {key: state[key] for key in ('account_id', 'pod_id', 'run_id',
               'run_spec_sha256', 'output_contract_sha256', 'controller_sources')}
    binding['lease_name'] = state['name']
    raw = json.dumps(binding, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def checked_state(root, expected_binding, stable=None):
    state = json.loads(read_file(root / 'LEASE.json'))
    if binding_digest(state) != expected_binding:
        raise ValueError('Receiver lease binding changed')
    identity = {key: state[key] for key in STABLE_KEYS}
    if stable is not None and identity != stable:
        raise ValueError('Receiver lease identity, inputs, or deadlines changed')
    sources = state['controller_sources']
    if set(sources) != SOURCE_NAMES:
        raise ValueError('Unexpected monitor source set')
    transport_source = None
    for name, expected in sources.items():
        source = read_file(root / 'monitor' / name)
        actual = {'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()}
        if actual != expected:
            raise ValueError('Monitor source hash changed: ' + name)
        if name == 'transport.py':
            if actual['sha256'] != TRANSPORT_SHA256:
                raise ValueError('Unexpected bounded transport implementation')
            transport_source = source
    return state, identity, transport_source


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc_epoch': time.time(), **fields},
                     sort_keys=True), flush=True)


def stop(signum, frame):
    # Unwind through run_bounded's process-group cleanup on service shutdown.
    raise SystemExit(0)


def supervise(root, expected_binding, *, interval=15, timeout=1200):
    root = no_links(root)
    if root.parent != ROUND / 'cloud_leases' or root.name.startswith('.') or not root.is_dir():
        raise ValueError('Receiver root must be a direct child of this round cloud_leases')
    state, stable, source = checked_state(root, expected_binding)
    # Compile the verified source bytes directly, without consulting cached bytecode.
    transport = types.ModuleType('_receiver_bounded_transport')
    transport.__file__ = str(root / 'monitor' / 'transport.py')
    exec(compile(source, transport.__file__, 'exec'), transport.__dict__)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    emit('receiver_started', lease=state['name'], binding_sha256=expected_binding,
         python=PYTHON, interval_seconds=interval, timeout_seconds=timeout)
    while True:
        state, _, _ = checked_state(root, expected_binding, stable)
        if state['status'] == 'terminated':
            emit('receiver_finished', reason='same_lease_terminated', lease=state['name'])
            return
        if state.get('ssh'):
            processes = []
            try:
                result = transport.run_bounded(
                    [PYTHON, '-B', str(root / 'monitor' / 'lease.py'), 'sync', '--root', str(root)],
                    timeout=timeout, operation='existing_lease_artifact_sync', processes=processes)
                emit('sync_finished', stdout_bytes=len(result.stdout), processes=processes)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                emit('sync_retry', error_type=type(error).__name__, processes=processes)
        # Revalidate immediately after collection; termination never waits another interval.
        state, _, _ = checked_state(root, expected_binding, stable)
        if state['status'] == 'terminated':
            emit('receiver_finished', reason='same_lease_terminated', lease=state['name'])
            return
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--interval', type=int, default=15, choices=range(1, 61), metavar='1..60')
    parser.add_argument('--timeout', type=int, default=1200, choices=range(30, 1801), metavar='30..1800')
    args = parser.parse_args(argv)
    if len(args.binding_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.binding_sha256):
        parser.error('--binding-sha256 must be a lowercase SHA-256 digest')
    supervise(args.lease, args.binding_sha256, interval=args.interval, timeout=args.timeout)


if __name__ == '__main__':
    main()
