#!/usr/bin/env python3
"""A5: lifecycle smoke with the unchanged controller and an isolated ledger.

Publish (the worker's Publisher and save_outputs), retrieve (LocalTransport),
verify (the pinned verifier), accept, then a mock deletion through the
controller's own watch fixture. Payload tensors are the A4 CPU outputs;
provenance, pod, account, SSH and provider are fixtures.

Isolation: the controller runs from a byte-identical temporary copy, lease.LEDGER
is patched in-process to a temporary ledger, and the real ledger is hashed
before and after.
Usage: python -B tests/lifecycle_smoke.py --work EMPTY_DIRECTORY --cpu-results A4_RESULTS_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

from common import BUNDLE, DIAGNOSTIC, LEDGER, ROUND, copy_bundle, empty_directory, require, save_evidence, tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cpu-results', type=Path, required=True)
    args = parser.parse_args()
    work = empty_directory(args.work)
    files = copy_bundle(work / 'bundle')
    controller = work / 'bundle/controller'
    sys.path[:0] = [str(controller), str(work / 'bundle/evaluation')]
    import torch
    import artifact_handoff as handoff
    import lease
    import test_lifecycle
    from artifacts import Publisher, json_save
    import worker
    before = tree(LEDGER)
    ledger = work / 'ledger'
    ledger.mkdir()
    fixture_bundle = work / 'bundle.bin'
    fixture_bundle.write_bytes(b'fixture bundle record; nothing is uploaded')
    with patch.object(lease, 'LEDGER', ledger):
        state = lease.make_plan(fixture_bundle, None, balance=20.0, rate=lease.MAX_GPU_RATE, run_config=DIAGNOSTIC / 'run_config.json',
                                debit_path=ROUND / 'resources/prior_debit.json', job_cap=1.5)
    state.update(status='running', mutation_phase='observed', pod_id='fixture-pod', machine_id='fixture-machine',
                 controller_sources={name: handoff.checked_record(controller / name) for name in lease.CONTROLLER_FILES},
                 ssh={'host': '127.0.0.1', 'port': 2222}, ssh_identity=str(work / 'unused_ssh'),
                 api_key_file=str(lease.io._no_links(lease.api.KEY_FILE)))
    require(state['controller_sources'] == json.loads((LEDGER / 'attempt_05/LEASE.json').read_text())['controller_sources'],
            'controller copy differs from the attempt05 pins')
    root = ledger / 'diagnostic_native_logit_v1'
    root.mkdir()
    lease.io._new_json(root / 'LEASE.json', state)

    publisher = Publisher(work / 'worker', handoff.binding(state), json.loads((BUNDLE / 'frozen/output_contract.json').read_text()))
    spec = json.loads((BUNDLE / 'frozen/run_spec.json').read_text())
    publisher.root.mkdir(parents=True)
    shutil.copyfile(BUNDLE / 'frozen/run_spec.json', publisher.root / 'run_spec.json')
    json_save(publisher.root / 'provenance.json', {
        'binding': handoff.binding(state), 'source_records': spec['source_records'], 'model_record': spec['model_record'],
        'dtype': 'torch.bfloat16', 'geometry': [4, 48, 192, 2048],
        'runtime': {'precision': spec['original_precision'], 'gpu': {'name': 'NVIDIA GeForce RTX 5090'}, 'fixture': True},
        'actual_worker_source': spec['source_records']['evaluation/worker.py'], 'config_record': {'bytes': 0, 'sha256': '0' * 64}})
    def load(name):
        return torch.load(args.cpu_results / name, map_location='cpu', weights_only=True)
    worker.save_outputs(publisher.root, load('development/native.pt'), load('diagnostic/tensors.pt'))
    manifest_sha256 = publisher.publish()
    publisher.status('computation_finished', outcome='complete')

    pointers = lease.sync_results(root, transport=handoff.LocalTransport(publisher.work))
    state = lease.io._json(root / 'LEASE.json')
    require(len(pointers) == 1 and handoff.accepted_for_current_run(root, state), 'the complete final manifest was not accepted')
    manifest = state['computation_finished']['manifest']
    validation = lease.io._json(handoff.generation_path(root, state, manifest, accepted=True) / 'VALIDATION.json')
    status = json.loads((publisher.work / 'status.json').read_text())
    watched = test_lifecycle.watch_fixture(SimpleNamespace(root=root, ledger=ledger), status=status)
    require(len(watched['deletions']) == 1 and watched['deletions'][0]['class'] == 'accepted_artifacts'
            and watched['state']['status'] == 'terminated' and watched['state']['absence_confirmations'] == 3,
            'the accepted run was not deleted with three absence confirmations')
    require(tree(LEDGER) == before, 'the real ledger changed')
    save_evidence('A5_lifecycle_smoke.json', {
        'schema': 'native_logit_lifecycle_smoke.v1', 'status': 'passed',
        'label': 'Lifecycle smoke: A4 CPU tensors with fixture provenance, pod, account, SSH and provider; mock deletion. Not GPU evidence.',
        'bundle_records': files,
        'ledger_isolation': 'Controller executed from a byte-identical temporary copy; lease.LEDGER patched in-process to a temporary ledger; real ledger hashed before and after.',
        'real_ledger_unchanged': True, 'controller_sources': state['controller_sources'],
        'published_manifest_sha256': manifest_sha256, 'manifest_files': manifest['files'],
        'acceptance_pointer': pointers[0], 'validation': validation,
        'termination_class': watched['deletions'][0]['class'], 'deletions': len(watched['deletions']),
        'absence_confirmations': watched['state']['absence_confirmations'], 'final_lease_status': watched['state']['status']})
    print(json.dumps({'status': 'passed', 'manifest_sha256': manifest_sha256}))


if __name__ == '__main__':
    main()
