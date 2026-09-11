"""Offline lifecycle fixtures and a real tiny CPU save/copy/load/verify smoke.

Run with the pinned local environment, PYTHONDONTWRITEBYTECODE=1. All scientific
payloads here are synthetic 2x2 CPU matrices. Provider/SSH boundaries are fake.
"""
from __future__ import annotations

import argparse
import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import artifact_handoff as handoff
import lease
import run_refits as io
from transport import run_bounded

HERE = Path(__file__).resolve().parent


def validate_outputs(*, root, manifest, contract):
    """Pinned fixture verifier: real ownership, tensor, and conversion readers."""
    import torch
    root = Path(root)
    owner = io._json(root / 'fit_01/OWNER.json')
    identity = owner['identity']
    if owner != {'schema_version': 1, 'fit_identity_sha256': io._digest(identity), 'identity': identity}:
        raise ValueError('fixture owner identity differs')
    if identity['cpu_test'] is not True or identity['d_model'] != 2 or identity['source_layers'] != [0, 1]:
        raise ValueError('fixture geometry changed')
    checkpoint = io._read_checkpoint(torch, root / 'fit_01', identity)
    final = io._read_final(torch, root / 'fit_01', identity, checkpoint)
    if final is None or checkpoint['cursor'] != 100:
        raise ValueError('fixture fit is incomplete')
    return {'schema': 'confirmation_validation.v1', 'status': 'passed', 'run_id': contract['run_id'],
            'manifest_sha256': handoff.digest(manifest),
            'contract_sha256': manifest['binding']['output_contract_sha256'],
            'checked_files': sorted(manifest['files']),
            'checks': {'loadability': True, 'numerical': True, 'owner_geometry': True},
            'details': {'scope': 'synthetic 2x2 CPU tensors; actual sealed checkpoint/final readers'}}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def rejects(operation, message, *, contains=None):
    try:
        operation()
    except Exception as error:
        if isinstance(error, AssertionError):
            raise
        if contains is not None and contains not in str(error):
            raise AssertionError('wrong rejection for ' + message + ': ' + str(error)) from error
        return type(error).__name__
    raise AssertionError('unexpected acceptance: ' + message)


def fixture(parent, name):
    import torch
    base = parent / name
    base.mkdir()
    workspace = io._mkdir(base / 'worker')
    results = io._mkdir(workspace / 'results')
    prompts = ['synthetic paragraph ' + str(i) for i in range(100)]
    identity = io._fit_identity(prompts, [3] * 100, [2] * 100, 1,
                                {'kind': 'local_fixture'}, 2, [0, 1], True)
    diagnostics = [{'index': i, 'prompt_sha256': identity['prompt_sha256'][i],
                    'token_length': 3, 'n_valid': 2} for i in range(100)]
    with io._owned_fit(results, 1, identity) as fit:
        sums = {0: torch.tensor([[100., 25.], [-50., 200.]]),
                1: torch.tensor([[2., 4.], [6., 8.]])}
        pointer, _ = io._commit(torch, fit, identity, sums, 100, diagnostics, 'checkpoint')
        io._commit(torch, fit, identity, sums, 100, diagnostics, 'final', checkpoint_pointer=pointer)
    (fit / '.lock').unlink()
    files = handoff.payload_inventory(results)
    spec_path = base / 'run_spec.json'
    io._new_json(spec_path, {'schema': 'fixture_run.v1', 'run_id': name, 'blind': True})
    contract_path = base / 'output_contract.json'
    contract = {'schema': 'confirmation_output_contract.v1', 'run_id': name,
                'run_spec_sha256': handoff.checked_record(spec_path)['sha256'],
                'files': {p: {'role': 'tiny_sealed_fit'} for p in files},
                'stages': {'smoke': sorted(files)}, 'required_checks': ['loadability', 'numerical', 'owner_geometry']}
    io._new_json(contract_path, contract)
    config_path = base / 'run_config.json'
    config = {'schema': 'confirmation_run_config.v1', 'run_id': name, 'run_spec_path': str(spec_path),
              'output_contract_path': str(contract_path), 'semantic_verifier': {
                  'path': str(Path(__file__).resolve()), 'record': handoff.checked_record(__file__), 'callable': 'validate_outputs'},
              'setup_budget_seconds': 90, 'compute_budget_seconds': 90,
              'preservation_reserve_seconds': 600, 'transfer_timeout_seconds': 30}
    io._new_json(config_path, config)
    historical = base / 'historical.json'
    old = {'name': 'fixture-old', 'pod_id': None, 'status': 'terminated',
           'lease_spend_upper_usd': 1.0, 'combined_spend_upper_usd': 1.0}
    io._new_json(historical, old)
    debit_path = base / 'prior_debit.json'
    debit = {'schema': 'confirmation_prior_debit.v1', 'account_id': 'fixture-account',
             'combined_cap_usd': 25.0, 'prior_spend_upper_usd': 1.0, 'remaining_authorized_upper_usd': 24.0,
             'minimum_balance_to_preserve_usd': 1.0, 'initial_ouro_job_incremental_cap_usd': 4.0,
             'historical_leases': [{**old, 'record': {'path': str(historical), **handoff.checked_record(historical)}}]}
    io._new_json(debit_path, debit)
    bundle = base / 'bundle.bin'
    bundle.write_bytes(b'fixture bundle, no executable or model')
    ledger = io._mkdir(base / 'ledger')
    with patch.object(lease, 'LEDGER', ledger):
        state = lease.make_plan(bundle, None, balance=20.0, rate=lease.MAX_GPU_RATE,
                                run_config=config_path, debit_path=debit_path, job_cap=4.0)
    state.update(status='running', mutation_phase='observed', pod_id='fixture-pod', machine_id='fixture-machine',
                 controller_sources={name: handoff.checked_record(HERE / name) for name in lease.CONTROLLER_FILES},
                 ssh={'host': '127.0.0.1', 'port': 2222}, ssh_identity=str(base / 'unused_ssh'),
                 api_key_file=str(io._no_links(lease.api.KEY_FILE)))
    root = io._mkdir(ledger / 'attempt_01')
    io._new_json(root / 'LEASE.json', state)
    return SimpleNamespace(base=base, root=root, workspace=workspace, results=results, state=state,
                           contract=contract, config=config, config_path=config_path, debit=debit,
                           debit_path=debit_path, bundle=bundle, ledger=ledger)


def publish(f, *, kind='final', outcome='complete'):
    state = io._json(f.root / 'LEASE.json')
    manifest = {'schema': 'confirmation_artifact_manifest.v1', 'binding': handoff.binding(state),
                'kind': kind, 'stage_id': 'smoke' if kind == 'stage' else None, 'outcome': outcome,
                'files': handoff.payload_inventory(f.results)}
    name = 'manifests/' + kind + '.json'
    path = io._no_links(f.workspace / name)
    io._mkdir(path.parent)
    io._new_json(path, manifest)
    index_path = f.workspace / 'artifact_index.json'
    index = io._json(index_path) if index_path.exists() else {
        'schema': 'confirmation_artifact_index.v1', 'binding': handoff.binding(state), 'manifests': []}
    index['manifests'].append({'path': name, 'record': handoff.checked_record(path)})
    io._atomic_json(index_path, index)
    return manifest


def discover(f, transport=None):
    transport = transport or handoff.LocalTransport(f.workspace)
    state = io._json(f.root / 'LEASE.json')
    manifests = handoff.read_index(state, transport)
    for manifest in manifests:
        state = lease.remember_manifest(f.root, state, manifest, transport.manifest_refs[handoff.digest(manifest)])
    return state, manifests, transport


def accept(f):
    return lease.sync_results(f.root, transport=handoff.LocalTransport(f.workspace))


def prepare_state(f, **fields):
    """Construct a different initial fixture state; not a production transition."""
    state = io._json(f.root / 'LEASE.json')
    state.update(fields)
    io._atomic_json(f.root / 'LEASE.json', state)
    return state


def pod_for(state):
    return {'id': state['pod_id'], 'name': state['name'], 'machineId': state['machine_id'],
            'imageName': state['image'], 'gpuCount': 1, 'containerDiskInGb': lease.DISK_GB, 'volumeInGb': 0,
            'machine': {'gpuDisplayName': lease.GPU}, 'costPerHr': lease.MAX_GPU_RATE,
            'runtime': {'ports': [{'privatePort': 22, 'isIpPublic': True, 'ip': '127.0.0.1', 'publicPort': 2222}]}}


class StopWatch(BaseException):
    pass


def watch_fixture(f, *, status=None, polls=3, advance=15, absent=False, termination=None):
    state = io._json(f.root / 'LEASE.json')
    clock = [time.time()]
    alive, sleeps = [not absent], [0]
    deletions, stops, collections = [], [], []
    live = {'id': state['account_id'], 'clientBalance': 20.0, 'currentSpendPerHr': lease.HARD_RATE}
    def status_read(*args):
        if isinstance(status, BaseException):
            raise status
        return status
    def listing(*args):
        return [pod_for(io._json(f.root / 'LEASE.json'))] if alive[0] else []
    def delete(pod):
        current = io._json(f.root / 'LEASE.json')
        authorization = current['termination_authorization']
        handoff.verify_record(authorization['path'], authorization['record'])
        intent = io._json(authorization['path'])
        require(intent['pod_id'] == pod, 'deletion targets another pod')
        deletions.append(intent)
        alive[0] = False
    def sleep(seconds):
        clock[0] += seconds if seconds != 15 else advance
        if seconds == 15:
            sleeps[0] += 1
            if sleeps[0] >= polls:
                raise StopWatch()
    with ExitStack() as stack:
        stack.enter_context(patch.object(lease, 'LEDGER', f.ledger))
        stack.enter_context(patch.object(lease, 'checked_account', return_value=live))
        stack.enter_context(patch.object(lease, 'pods', side_effect=listing))
        stack.enter_context(patch.object(lease.api, 'terminate', side_effect=delete))
        stack.enter_context(patch.object(lease.api, 'gql', side_effect=AssertionError('fixture reached provider')))
        stack.enter_context(patch.object(lease, 'remote_status', side_effect=status_read))
        stack.enter_context(patch.object(lease, 'ensure_sync', side_effect=lambda *a: collections.append(True)))
        stack.enter_context(patch.object(lease, 'request_worker_stop', side_effect=lambda *a: stops.append(True)))
        stack.enter_context(patch.object(lease.time, 'time', side_effect=lambda: clock[0]))
        stack.enter_context(patch.object(lease.time, 'sleep', side_effect=sleep))
        try:
            if termination is None:
                lease.watch(SimpleNamespace(root=f.root))
            else:
                lease.terminate_owned(f.root, termination)
        except StopWatch:
            pass
    return {'deletions': deletions, 'stops': len(stops), 'collections': len(collections),
            'state': io._json(f.root / 'LEASE.json')}


def run_smoke(parent):
    f = fixture(parent, 'smoke')
    manifest = publish(f)
    interrupted = handoff.LocalTransport(f.workspace, interrupt_after_bytes=256)
    require(rejects(lambda: lease.sync_results(f.root, transport=interrupted), 'interrupted copy') == 'TimeoutError',
            'copy did not interrupt at real partial bytes')
    state = io._json(f.root / 'LEASE.json')
    require(not handoff.accepted_for_current_run(f.root, state), 'partial copy was accepted')
    staging = handoff.generation_path(f.root, state, manifest, accepted=False)
    require(any((staging / 'partials').rglob('*')), 'interruption left no resumable bytes')
    resumed = handoff.LocalTransport(f.workspace)
    pointers = lease.sync_results(f.root, transport=resumed)
    require(resumed.resumed_files and len(pointers) == 1, 'resume did not consume the same partial staging')
    state = io._json(f.root / 'LEASE.json')
    require(handoff.accepted_for_current_run(f.root, state), 'real load/conversion acceptance missing')
    watched = watch_fixture(f, status=subprocess.TimeoutExpired('synthetic ssh', 30))
    require(len(watched['deletions']) == 1 and watched['deletions'][0]['class'] == 'accepted_artifacts',
            'accepted artifacts plus status timeout did not permit normal deletion')
    require(watched['state']['absence_confirmations'] == 3, 'deletion lacks three bound absence confirmations')
    return {'case': 'tiny_cpu_save_interrupt_resume_load_convert_accept_delete', 'status': 'passed',
            'root': str(f.root), 'manifest_sha256': handoff.digest(manifest), 'files': manifest['files'],
            'resumed_files': resumed.resumed_files, 'acceptance': pointers[0],
            'termination_class': watched['deletions'][0]['class'], 'absence_confirmations': 3}


def run_cases(parent):
    cases = []
    def passed(name, **details):
        cases.append({'case': name, 'status': 'passed', **details})

    f = fixture(parent, 'setup_restart')
    state = io._json(f.root / 'LEASE.json')
    status = {'lease_name': state['name'], 'binding': handoff.binding(state), 'setup_complete': True, 'phase': 'evaluating'}
    state = lease.remember_worker_status(f.root, state, status)
    lease.update(f.root, setup_deadline_utc=lease.stamp(time.time() - 10))
    for observed in (None, {**status, 'setup_complete': False}, subprocess.TimeoutExpired('ssh', 30)):
        result = watch_fixture(f, status=observed, polls=3)
        require(not result['deletions'] and not result['stops'] and lease.setup_completed_for_lease(result['state']),
                'setup regressed after restart/missing/false/three timeouts')
    passed('setup_restart_missing_false_three_transport_errors')

    f = fixture(parent, 'never_setup')
    lease.update(f.root, setup_deadline_utc=lease.stamp(time.time() - 10))
    result = watch_fixture(f, status=None)
    require(result['stops'] and not result['deletions'] and not result['state'].get('retrieval_accepted'),
            'setup timeout deleted or fabricated acceptance')
    passed('setup_timeout_stops_preserves_until_hard_deadline')

    f = fixture(parent, 'concurrency')
    observed = io._json(f.root / 'LEASE.json')
    status = {'lease_name': observed['name'], 'binding': handoff.binding(observed), 'setup_complete': True, 'phase': 'compute'}
    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(lease.update, f.root, halt_requested=True, all_in_rate=1.5),
                   pool.submit(lease.remember_worker_status, f.root, observed, status)]
        [r.result() for r in results]
    state = io._json(f.root / 'LEASE.json')
    require(state['halt_requested'] and state['all_in_rate'] == 1.5 and lease.setup_completed_for_lease(state),
            'state lock lost concurrent halt/rate/setup fields')
    rejects(lambda: lease.update(f.root, pod_id='another-pod'), 'bound pod changed')
    wrong_observer = {**observed, 'pod_id': 'another-pod'}
    rejects(lambda: lease.remember_worker_status(f.root, wrong_observer, status), 'stale pod status')
    require(not lease.setup_completed_for_lease({**state, 'pod_id': 'another-pod'}), 'old latch crossed pod identity')
    rejects(lambda: lease.update(f.root, run_id='another-run'), 'run ID changed')
    rejects(lambda: lease.update(f.root, provider_deadline_utc=lease.stamp(lease.epoch(state['provider_deadline_utc']) + 1)),
            'provider TTL extended')
    passed('locked_updates_and_stale_observer')

    f = fixture(parent, 'early_stop')
    state = io._json(f.root / 'LEASE.json')
    lease.remember_worker_status(f.root, state, {'lease_name': state['name'], 'binding': handoff.binding(state),
                                              'setup_complete': True, 'phase': 'compute'})
    lease.update(f.root, work_deadline_utc=lease.stamp(time.time() - 1))
    result = watch_fixture(f, status=subprocess.TimeoutExpired('ssh', 30))
    require(result['stops'] and result['collections'] and not result['deletions'], 'status timeout skipped early preservation')
    passed('early_preservation_survives_status_timeout')

    f = fixture(parent, 'emergency')
    manifest = publish(f)
    discover(f)
    lease.update(f.root, watch_deadline_utc=lease.stamp(time.time() - 1), retrieval_in_progress={'phase': 'transferring'})
    with handoff.handoff_lock(f.root):
        result = watch_fixture(f)
    require(len(result['deletions']) == 1 and result['deletions'][0]['class'] == 'budget_emergency'
            and set(result['deletions'][0]['not_previously_accepted_files']) == set(manifest['files'])
            and not result['state'].get('retrieval_accepted'), 'emergency waited on collector or claimed acceptance')
    passed('budget_emergency_precedes_mutation_and_bypasses_collection_lock')

    f = fixture(parent, 'provider_absent')
    result = watch_fixture(f, absent=True)
    require(not result['deletions'] and result['state']['status'] == 'terminated'
            and result['state']['absence_confirmations'] == 3, 'absence issued a deletion or lacked confirmation')
    passed('provider_absence_without_new_deletion')

    f = fixture(parent, 'stage_reuse')
    stage = publish(f, kind='stage')
    accept(f)
    state = io._json(f.root / 'LEASE.json')
    require(state['stage_acceptances'] and not handoff.accepted_for_current_run(f.root, state), 'stage authorized final deletion')
    final = publish(f)
    receiver = handoff.LocalTransport(f.workspace)
    lease.sync_results(f.root, transport=receiver)
    require(receiver.transferred_bytes == 0 and handoff.accepted_for_current_run(f.root, io._json(f.root / 'LEASE.json')),
            'final did not reuse exact accepted stage bytes')
    with handoff.handoff_lock(f.root):
        result = watch_fixture(f, polls=1)
    require(not result['deletions'], 'normal deletion waited for or bypassed handoff lock')
    passed('stage_preservation_reuse_and_nonblocking_normal_delete')

    f = fixture(parent, 'stale_ack')
    manifest = publish(f)
    accept(f)
    state = io._json(f.root / 'LEASE.json')
    for key, value in [('run_id', 'wrong-run'), ('pod_id', 'wrong-pod'), ('output_contract_sha256', '0' * 64),
                       ('controller_sources', {**state['controller_sources'], 'extra.py': {'bytes': 0, 'sha256': '0' * 64}})]:
        bad = {**state, key: value}
        require(not handoff.accepted_for_current_run(f.root, bad), 'stale acknowledgement accepted: ' + key)
    bad = copy.deepcopy(state)
    bad['retrieval_accepted']['manifest_sha256'] = '0' * 64
    require(not handoff.accepted_for_current_run(f.root, bad), 'wrong manifest acknowledgement accepted')
    bad = copy.deepcopy(manifest)
    bad['files'].pop(next(iter(bad['files'])))
    rejects(lambda: handoff.validate_manifest(state, bad), 'required artifact omitted')
    bad = copy.deepcopy(manifest)
    bad['files']['../escape'] = next(iter(bad['files'].values()))
    rejects(lambda: handoff.validate_manifest(state, bad), 'traversal manifest')
    passed('stale_receipt_exact_binding_and_required_pathset')

    for variant in ('payload', 'manifest', 'receiptless_payload'):
        f = fixture(parent, 'repair_' + variant)
        manifest = publish(f)
        accept(f)
        state = io._json(f.root / 'LEASE.json')
        old_pointer = copy.deepcopy(state['retrieval_accepted'])
        generation = handoff.generation_path(f.root, state, manifest, accepted=True)
        if variant == 'manifest':
            (generation / 'MANIFEST.json').unlink()
        else:
            name = next(iter(manifest['files']))
            target = generation / 'results' / name
            data = bytearray(target.read_bytes())
            data[0] ^= 1
            target.write_bytes(data)
            if variant == 'receiptless_payload':
                (generation / 'RECEIPT.json').unlink()
        require(not handoff.accepted_for_current_run(f.root, state), 'damaged published artifact remained accepted')
        accept(f)
        after = io._json(f.root / 'LEASE.json')
        require(handoff.accepted_for_current_run(f.root, after) and after['retrieval_accepted'] != old_pointer,
                'published corruption did not get new verified acceptance')
        require(any((f.root / 'handoff/quarantine' / state['run_id']).iterdir()), 'repair destroyed original evidence')
        passed('quarantine_and_repair_' + variant)

    for stage in ('after_transfer', 'after_validation', 'after_pending_unlink', 'after_publication', 'before_receipt_link', 'after_receipt_link'):
        f = fixture(parent, 'crash_' + stage)
        manifest = publish(f)
        state, _, transport = discover(f)
        def crash(point):
            if point == stage:
                raise RuntimeError('synthetic crash at ' + point)
        rejects(lambda: handoff.collect_manifest(f.root, state, manifest, transport, fault_hook=crash), stage)
        pointer = handoff.collect_manifest(f.root, state, manifest, transport)
        require(pointer['manifest_sha256'] == handoff.digest(manifest), 'crash recovery accepted another manifest')
        passed('atomic_publication_resume_' + stage)

    f = fixture(parent, 'double_collector')
    manifest = publish(f)
    state, _, transport = discover(f)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(handoff.collect_manifest, f.root, state, manifest, transport) for _ in range(2)]
        pointers = [future.result() for future in futures]
    require(pointers[0] == pointers[1], 'two collectors published conflicting receipts')
    passed('double_collector_single_publication')

    f = fixture(parent, 'changed_source')
    manifest = publish(f)
    state, _, transport = discover(f)
    source = f.results / next(iter(manifest['files']))
    source.write_bytes(source.read_bytes() + b'changed')
    rejects(lambda: handoff.collect_manifest(f.root, state, manifest, transport), 'source mutation')
    require(not (handoff.generation_path(f.root, state, manifest, accepted=True) / 'RECEIPT.json').exists(),
            'source mutation published acceptance')
    passed('source_mutation_rejects_transfer')

    for variant in ('manifest', 'pending_source_unlink'):
        f = fixture(parent, 'after_transfer_' + variant)
        manifest = publish(f)
        state, _, transport = discover(f)
        def mutate(stage):
            if stage != 'after_transfer':
                return
            if variant == 'manifest':
                changed = {**manifest, 'outcome': 'failed'}
                io._atomic_json(f.workspace / 'manifests/final.json', changed)
            else:
                (f.results / next(iter(manifest['files']))).unlink()
        rejects(lambda: handoff.collect_manifest(f.root, state, manifest, transport, fault_hook=mutate),
                'source changed after transfer')
        require(not handoff.generation_path(f.root, state, manifest, accepted=True).exists(),
                'post-transfer source mutation reached publication')
        passed('after_transfer_source_change_' + variant)

    import torch
    for variant in ('unloadable', 'wrong_shape', 'wrong_dtype', 'nonfinite', 'wrong_mean'):
        f = fixture(parent, 'invalid_' + variant)
        pointer_path = f.results / 'fit_01/COMPLETE.json'
        pointer = io._json(pointer_path)
        generation = f.results / 'fit_01' / pointer['generation']
        binary = generation / 'lens.pt'
        if variant == 'unloadable':
            binary.write_bytes(b'hash-consistent bytes that are not a tensor archive')
        else:
            payload = torch.load(binary, weights_only=True, map_location='cpu')
            if variant == 'wrong_shape':
                payload['J'][0] = torch.zeros((3, 3), dtype=torch.float16)
            elif variant == 'wrong_dtype':
                payload['J'][0] = payload['J'][0].float()
            elif variant == 'nonfinite':
                payload['J'][0][0, 0] = float('nan')
            else:
                payload['J'][0][0, 0] += 1
            torch.save(payload, binary)
        seal = io._json(generation / 'SEAL.json')
        seal['files']['lens.pt'] = handoff.checked_record(binary)
        io._atomic_json(generation / 'SEAL.json', seal)
        pointer['seal'] = handoff.checked_record(generation / 'SEAL.json')
        io._atomic_json(pointer_path, pointer)
        publish(f)
        expected_error = {'wrong_shape': 'invalid shape, dtype', 'wrong_dtype': 'invalid shape, dtype',
                          'nonfinite': 'nonfinite', 'wrong_mean': 'FP16 N=100 mean'}.get(variant)
        rejects(lambda: accept(f), variant, contains=expected_error)
        require(not io._json(f.root / 'LEASE.json').get('retrieval_accepted'), 'semantic failure was accepted')
        passed('hash_consistent_semantic_failure_' + variant)

    f = fixture(parent, 'truncated')
    manifest = publish(f)
    state, _, transport = discover(f)
    staging = handoff.generation_path(f.root, state, manifest, accepted=False)
    partial = io._mkdir(staging / 'partials')
    name = next(iter(manifest['files']))
    target = io._no_links(partial / name)
    io._mkdir(target.parent)
    target.write_bytes(b'wrong prefix')
    pointer = handoff.collect_manifest(f.root, state, manifest, transport)
    require(pointer and transport.resumed_files, 'corrupt partial did not resume safely')
    passed('truncated_wrong_prefix_resume')

    f = fixture(parent, 'debit')
    state = io._json(f.root / 'LEASE.json')
    lease.update(f.root, status='terminated', lease_spend_upper_usd=0.2, combined_spend_upper_usd=1.2)
    with patch.object(lease, 'LEDGER', f.ledger):
        _, _, debit, previous = lease.prior_debit(f.debit_path)
        require(debit == 1.2 and len(previous) == 1, 'new ledger reset or double counted prior debit')
        require(lease.prior_debit(f.debit_path)[2] == 1.2, 'restart changed debit')
        rejects(lambda: lease.make_plan(f.bundle, None, balance=20, rate=lease.MAX_GPU_RATE,
            run_config=f.config_path, debit_path=f.debit_path, job_cap=0.2), 'insufficient admission reserve')
        rejects(lambda: lease.make_plan(f.bundle, None, balance=20, rate=lease.MAX_GPU_RATE,
            run_config=f.config_path, debit_path=f.debit_path, job_cap=4.1), 'job ceiling increased')
    reduced = {**f.debit, 'prior_spend_upper_usd': 0.1, 'remaining_authorized_upper_usd': 24.9}
    reduced_path = f.base / 'reduced_debit.json'
    io._new_json(reduced_path, reduced)
    with patch.object(lease, 'LEDGER', f.base / 'empty'):
        rejects(lambda: lease.prior_debit(reduced_path), 'historical debit reduced')
    passed('prior_debit_carried_once_and_admission_caps')

    f = fixture(parent, 'create_race')
    prepare_state(f, status='pending', mutation_phase='not_started', pod_id=None, machine_id=None,
                  watch_ready_utc=lease.stamp(), watch_account_id=f.state['account_id'])
    def begin():
        try:
            lease.begin_create(f.root)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(2) as pool:
        outcomes = [v.result() for v in [pool.submit(begin), pool.submit(begin)]]
    require(outcomes.count(True) == 1, 'create intent committed more than once')
    lease.fail_create(f.root)
    require(not begin(), 'uncertain create could be retried')
    require(not lease.confirm_absence(f.root, 3, f.state['account_id']), 'uncertain create finalized absence before TTL')
    current = io._json(f.root / 'LEASE.json')
    with patch.object(lease.time, 'time', return_value=lease.epoch(current['provider_deadline_utc']) + 1):
        require(lease.confirm_absence(f.root, 3, f.state['account_id']), 'uncertain create did not reconcile after fixed TTL')
    passed('create_race_uncertainty_halt_and_absence_barriers')

    f = fixture(parent, 'verifier_changed')
    verifier_copy = f.base / 'validator.py'
    shutil.copyfile(__file__, verifier_copy)
    state = io._json(f.root / 'LEASE.json')
    verifier = {**state['semantic_verifier'], 'path': str(verifier_copy)}
    prepare_state(f, semantic_verifier=verifier, status='pending', mutation_phase='not_started',
                  watch_ready_utc=lease.stamp(), watch_account_id=state['account_id'])
    verifier_copy.write_text(verifier_copy.read_text() + '\n# changed\n')
    rejects(lambda: lease.begin_create(f.root), 'verifier changed before paid creation')
    require(io._json(f.root / 'LEASE.json')['mutation_phase'] == 'not_started', 'bad verifier allowed create intent')
    passed('verifier_pin_rechecked_before_create')

    f = fixture(parent, 'forged_worker_ack')
    state = io._json(f.root / 'LEASE.json')
    status = {'lease_name': state['name'], 'binding': handoff.binding(state), 'phase': 'complete',
              'setup_complete': True, 'retrieval_accepted': {'status': 'passed'}}
    result = watch_fixture(f, status=status)
    require(not result['deletions'] and not result['state'].get('retrieval_accepted')
            and not result['state'].get('computation_finished'), 'worker status supplied local acceptance')
    passed('worker_complete_and_forged_ack_are_not_receiver_acceptance')

    f = fixture(parent, 'explicit_teardown')
    state = io._json(f.root / 'LEASE.json')
    authorization = f.base / 'root_teardown.json'
    document = {'schema': 'confirmation_explicit_teardown.v1', 'binding': handoff.binding(state),
                'scope': 'fixture explicit stop after reviewing partial preservation', 'authorized_by': 'root'}
    io._new_json(authorization, {**document, 'binding': {**document['binding'], 'run_id': 'another-run'}})
    rejects(lambda: watch_fixture(f, termination={'class': 'explicit_teardown', 'authorization_path': authorization}),
            'stale explicit teardown')
    io._atomic_json(authorization, document)
    result = watch_fixture(f, termination={'class': 'explicit_teardown', 'authorization_path': authorization})
    require(len(result['deletions']) == 1 and result['deletions'][0]['class'] == 'explicit_teardown'
            and result['deletions'][0]['reason'] == document['scope'], 'explicit teardown lost its local scope')
    passed('root_explicit_teardown_exact_binding_and_scope')

    f = fixture(parent, 'mutation_guards')
    state = io._json(f.root / 'LEASE.json')
    live = {'id': state['account_id'], 'clientBalance': 20.0, 'currentSpendPerHr': lease.HARD_RATE}
    lease.update(f.root, watch_deadline_utc=lease.stamp(time.time() - 1))
    with patch.object(lease, 'account', return_value={**live, 'id': 'another-account'}), \
            patch.object(lease.api, 'terminate') as deletion:
        rejects(lambda: lease.terminate_owned(f.root, {'class': 'budget_emergency'}), 'account identity changed',
                contains='expected account')
        deletion.assert_not_called()
    for changed in ({**pod_for(state), 'name': 'renamed-pod'}, {**pod_for(state), 'id': 'foreign-pod'}):
        with patch.object(lease, 'checked_account', return_value=live), \
                patch.object(lease, 'pods', return_value=[changed]), \
                patch.object(lease.api, 'terminate') as deletion, patch.object(lease.time, 'sleep'):
            require(lease.terminate_owned(f.root, {'class': 'budget_emergency'}) is False,
                    'changed pod identity was treated as deleted')
            deletion.assert_not_called()
    passed('account_renamed_and_foreign_pod_mutation_guards')

    f = fixture(parent, 'bootstrap')
    script = (HERE / 'pod_entry.sh').read_text().split("python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    bootstrap = io._mkdir(f.base / 'bootstrap')
    for name in ('workspace/jlens', 'root/.ssh', 'etc/ssh/sshd_config.d'):
        io._mkdir(bootstrap / name)
    environment = {'PUBLIC_KEY': 'ssh-ed25519 AAAA fixture', 'JLENS_LEASE_NAME': f.state['name']}
    normal_import = builtins.__import__
    def controlled_import(name, *args, **kwargs):
        if name == 'os':
            return SimpleNamespace(environ=environment)
        if name == 'pathlib':
            return SimpleNamespace(Path=lambda path: bootstrap / path.lstrip('/'))
        return normal_import(name, *args, **kwargs)
    namespace = {'__builtins__': {**vars(builtins), '__import__': controlled_import}}
    exec(compile(script, 'actual_pod_entry_python', 'exec'), namespace)
    require((bootstrap / 'workspace/jlens/provider_lease_name').read_text().strip() == f.state['name'],
            'actual bootstrap rejected or changed the planned confirmation name')
    environment['JLENS_LEASE_NAME'] = f.state['name'].replace('jlens-confirm-', 'jlens-refit-')
    try:
        exec(compile(script, 'actual_pod_entry_python', 'exec'), namespace)
    except SystemExit:
        pass
    else:
        raise AssertionError('new bootstrap accepted the historical name prefix')
    subprocess.run(['bash', '-n', str(HERE / 'pod_entry.sh')], check=True)
    passed('actual_bootstrap_name_and_ssh_setup_snippet_without_system_mutation')

    f = fixture(parent, 'failed_final')
    manifest = publish(f, outcome='failed')
    accept(f)
    state = io._json(f.root / 'LEASE.json')
    require(state['computation_finished']['outcome'] == 'failed' and not state.get('retrieval_accepted'),
            'failed scientific run counted as full acceptance')
    passed('failed_terminal_preserved_without_full_acceptance')

    processes = []
    rejects(lambda: run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'],
                               timeout=0.05, operation='fixture_timeout', processes=processes), 'bounded subprocess timeout')
    require(processes[0]['process_group_quiescent'], 'timed out process group remained alive')
    passed('bounded_process_group_cleanup', processes=processes)

    # Execute the actual generated SSH metadata snippet against a local fixture.
    f = fixture(parent, 'metadata_atime')
    metadata = f.workspace / 'artifact_index.json'
    metadata.write_text('{"unchanged":true}')
    os.utime(metadata, (1, time.time()))
    receiver = handoff.SSHTransport(f.root, f.state, ['unused-ssh'])
    def local_python(script, **kwargs):
        script = script.replace("pathlib.Path('/workspace/jlens')", 'pathlib.Path(' + repr(str(f.workspace)) + ')')
        return subprocess.run([sys.executable, '-c', script], check=True, capture_output=True).stdout
    with patch.object(receiver, '_python', side_effect=local_python):
        require(receiver.read_bytes('artifact_index.json') == b'{"unchanged":true}', 'atime rejected stable metadata')
    passed('metadata_read_stat_guard_ignores_atime_only_change')
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Only the real tiny CPU end-to-end smoke')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    parent = args.output_dir or Path(tempfile.mkdtemp(prefix='confirmation-controller-proof-'))
    parent = io._mkdir(parent)
    started = time.time()
    smoke = run_smoke(parent)
    cases = [] if args.smoke else run_cases(parent)
    proof = {'schema': 'confirmation_controller_proof.v1', 'status': 'passed',
             'scope': 'Real temporary files, locks, tiny CPU serialization, hashes, semantic readers and receiver; fake provider/SSH mutation boundaries.',
             'source_records': {name: handoff.checked_record(HERE / name) for name in (*lease.CONTROLLER_FILES, 'test_lifecycle.py')},
             'python': sys.executable, 'smoke': smoke, 'cases': cases, 'elapsed_seconds': time.time() - started,
             'no_paid_science': True, 'no_cloud_or_ssh': True}
    io._new_json(parent / 'PROOF.json', proof)
    print(json.dumps({'status': 'passed', 'proof': str(parent / 'PROOF.json'), 'cases': len(cases) + 1,
                      'elapsed_seconds': proof['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()

# changed
