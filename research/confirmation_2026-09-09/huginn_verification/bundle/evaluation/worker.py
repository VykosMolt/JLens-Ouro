#!/usr/bin/env python3
"""Run the unchanged N100 Huginn chain with externally accepted native evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from artifacts import Publisher, atomic, json_save, record
from export_fit import export_fit, export_checkpoint_evidence

HISTORICAL_PROMPT_BOUND = 130.48383641405962


def fit_admission(remaining, measured):
    if not math.isfinite(remaining) or remaining <= 0 or not math.isfinite(measured) or measured <= 0:
        raise ValueError('Invalid complete preflight prompt timing')
    per_prompt = 1.25 * max(measured, HISTORICAL_PROMPT_BOUND)
    required = 100 * per_prompt + 900
    if remaining < required:
        raise ValueError('Insufficient work time for all100 prompts and readouts')
    return {'n_prompts': 100, 'remaining_work_seconds': remaining,
            'measured_preflight_prompt_seconds': measured,
            'historical_prompt_bound_seconds': HISTORICAL_PROMPT_BOUND,
            'initial_prompt_bound_seconds': per_prompt, 'required_seconds': required,
            'readout_reserve_seconds': 900, 'decision': 'admitted'}


def copy_once(source, destination):
    source, destination = Path(source), Path(destination)
    import shutil
    def write(output):
        with source.open('rb') as incoming:
            shutil.copyfileobj(incoming, output, 8 * 1024 * 1024)
    atomic(destination, write)
    if record(source) != record(destination):
        raise ValueError('Preserved file differs from completed child output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work, bundle = Path(config['work']), Path(config['bundle'])
    specpath = bundle / 'frozen/run_spec.json'
    contractpath = bundle / 'frozen/output_contract.json'
    binding = config['binding']
    if (record(specpath)['sha256'] != binding['run_spec_sha256']
            or record(contractpath)['sha256'] != binding['output_contract_sha256']):
        raise ValueError('Huginn launch binding differs')
    spec, contract = json.loads(specpath.read_text()), json.loads(contractpath.read_text())
    publisher = Publisher(work, binding, contract)
    deadline = datetime.fromisoformat(config['work_deadline_utc'].replace('Z', '+00:00')).timestamp()
    stopped = False
    preservation_errors = []
    children_quiescent = True
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def check():
        if stopped or (work / 'STOP').exists() or time.time() >= deadline:
            raise InterruptedError('Work stopped; preserve committed stages within reserve')
    def verify():
        for rel, expected in spec['source_records'].items():
            if record(bundle / rel) != expected:
                raise ValueError('Huginn frozen source/input differs: ' + rel)
    def run(label, command):
        nonlocal children_quiescent
        check()
        publisher.status(label, updated_utc=datetime.now(timezone.utc).isoformat())
        log = work / 'child_logs' / (label + '.log')
        log.parent.mkdir(exist_ok=True)
        failure = None
        try:
            with log.open('xb') as output:
                child = subprocess.Popen([str(x) for x in command], stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                children_quiescent = False
                try:
                    while child.poll() is None:
                        check()
                        time.sleep(2)
                    if child.returncode:
                        raise RuntimeError(f'{label} exited{child.returncode}; inspect child log')
                finally:
                    stop_process(child)
                    children_quiescent = True
        except BaseException as error:
            failure = error
            raise
        finally:
            if children_quiescent and log.is_file():
                try:
                    copy_once(log, publisher.root / 'logs' / log.name)
                except BaseException as error:
                    preservation_errors.append({'operation': 'copy_child_log', 'label': label,
                                                'error_type': type(error).__name__, 'error': str(error)})
                    if failure is None:
                        raise
    deployment = bundle / 'repo/research/refit_round_2026-09-07/deployment'
    common = ['--run-spec', deployment / 'run_spec.json', '--combined-contract', deployment / 'combined_contract.json',
              '--ouro-src', bundle / 'ouro_project/src']
    calibration = ['--calibration', deployment / 'huginn_calibration.json', '--snapshot', work / 'models/huginn',
                   '--ouro-evaluation-dir', bundle / 'prerequisites/ouro_evaluation',
                   '--controls-evaluation-dir', bundle / 'prerequisites/controls_evaluation']
    try:
        verify()
        sys.path.insert(0, str(bundle / 'controller'))
        from transport import stop_process
        copy_once(specpath, publisher.root / 'run_spec.json')
        json_save(publisher.root / 'provenance.json', {
            'binding': binding, 'category': 'newly_rerun_estimator', 'n_prompts': 100,
            'source_records': spec['source_records'], 'actual_worker_source': record(Path(__file__)),
            'config': config, 'purpose': 'inspectable historical Huginn chain; no new-item model comparison'})
        gate_scratch = work / 'scratch/preflight'
        run('native_gate', [sys.executable, bundle / 'evaluation/preserve_gate.py',
                           '--deployment', deployment, '--evidence', publisher.root / 'gate', '--',
                           '--kind', 'huginn', *common, *calibration, '--output-dir', gate_scratch])
        for name in ('START.json', 'huginn_r8_START.json', 'huginn_r8.json', 'COMPLETE.json'):
            copy_once(gate_scratch / name, publisher.root / 'gate' / name)
        gate = json.loads((gate_scratch / 'COMPLETE.json').read_text())
        if gate['status'] != 'passed' or len(gate['profiles']) != 1:
            raise ValueError('Native preflight did not complete')
        identifier = publisher.publish('development')
        publisher.status('awaiting_external_development_acceptance', development_manifest_sha256=identifier)
        ackpath = work / 'DEVELOPMENT_ACCEPTED.json'
        while not ackpath.exists():
            check()
            time.sleep(2)
        ack = json.loads(ackpath.read_text())
        if (ack.get('kind') != 'stage_science_authorization' or ack.get('binding') != binding
                or ack.get('stage_manifest_sha256') != identifier or not ack.get('acceptance_receipt_record')):
            raise ValueError('Huginn native gate lacks exact external acceptance')
        admission = fit_admission(deadline - time.time(), gate['profiles'][0]['prompt_with_checkpoint_seconds'])
        json_save(publisher.root / 'fit_admission.json', {**admission, 'binding': binding,
                  'native_gate_manifest_sha256': identifier, 'external_acceptance': ack,
                  'utc': datetime.now(timezone.utc).isoformat()})
        output = work / 'scratch/huginn'
        run('fit_n100', [sys.executable, deployment / 'run_huginn.py', *common, *calibration,
                        '--output-dir', output, '--stop-at-utc', config['work_deadline_utc'],
                        '--reserve-seconds', '900', '--initial-prompt-bound-seconds',
                        str(admission['initial_prompt_bound_seconds'])])
        export_fit(output / 'fit_01', publisher.root / 'fit')
        publisher.publish('fit')
        # A completed estimator is offered to the receiver immediately, while
        # the original two-seed readout executes within the work deadline.
        readouts = work / 'scratch/huginn_evaluation'
        run('historical_readouts', [sys.executable, deployment / 'evaluate_huginn.py', *common,
                '--huginn-calibration', deployment / 'huginn_calibration.json',
                '--ouro-manifest', deployment / 'model_manifest.json',
                '--huginn-manifest', deployment / 'huginn_model_manifest.json',
                '--eligibility', deployment / 'huginn_eligibility.json',
                '--ouro-snapshot', work / 'models/ouro', '--huginn-snapshot', work / 'models/huginn',
                '--fit-dir', output / 'fit_01', '--out', readouts,
                '--stop-at-utc', config['work_deadline_utc'], '--reserve-seconds', '0'])
        for relative in spec['readout_paths']:
            copy_once(readouts / relative, publisher.root / 'readouts' / relative)
        publisher.publish('readouts')
        verify()
        publisher.publish()
        publisher.status('computation_finished', outcome='complete')
    except BaseException as error:
        outcome = 'stopped' if isinstance(error, InterruptedError) else 'failed'
        if children_quiescent:
            fit_root = work / 'scratch/huginn/fit_01'
            if (fit_root / 'LATEST.json').is_file() and not (publisher.root / 'fit/evidence.json').exists():
                try:
                    export_checkpoint_evidence(fit_root, publisher.root / 'fit')
                except BaseException as recovery_error:
                    preservation_errors.append({'operation': 'export_checkpoint_evidence',
                        'error_type': type(recovery_error).__name__, 'error': str(recovery_error)})
            for relative in spec['readout_paths']:
                source = work / 'scratch/huginn_evaluation' / relative
                target = publisher.root / 'readouts' / relative
                if source.is_file() and not target.exists():
                    try:
                        copy_once(source, target)
                    except BaseException as recovery_error:
                        preservation_errors.append({'operation': 'copy_existing_readout', 'path': relative,
                            'error_type': type(recovery_error).__name__, 'error': str(recovery_error)})
        json_save(work / 'WORKER_FAILURE.json', {'binding': binding, 'outcome': outcome,
                  'error_type': type(error).__name__, 'error': str(error),
                  'children_quiescent': children_quiescent, 'preservation_errors': preservation_errors}, replace=True)
        publisher.publish(outcome=outcome)
        publisher.status('computation_finished', outcome=outcome, error_type=type(error).__name__, error=str(error),
                         children_quiescent=children_quiescent, preservation_errors=preservation_errors)
        raise

if __name__ == '__main__':
    main()
