#!/usr/bin/env python3
"""Package the historical Huginn closure and separately reviewed retention code."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from artifacts import record, json_save

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[2]
OLD = REPO / 'research/refit_round_2026-09-07'
OURO_SRC = Path('/home/moloch/ouro_project/src')
READOUT_PATHS = ['OWNER.json', 'COMPLETE.json', 'population/eligibility.json', 'population/metadata.json', 'population/SEAL.json'] + [
    f'seeds/{seed}/{name}' for seed in (2026090803, 2026090804)
    for name in ('cache.pt', 'arrays.npz', 'initializations.json', 'metadata.json', 'summaries.json', 'SEAL.json')]


def copy(source, target, *, expected=None, expected_sha256=None):
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Only real regular source files may be packaged')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8*1024*1024)
    observed = record(target)
    if record(source) != observed:
        raise ValueError('Packaged copy differs')
    if expected is not None and observed != expected:
        raise ValueError('Packaged copy differs from its exact reviewed record')
    if expected_sha256 is not None and observed['sha256'] != expected_sha256:
        raise ValueError('Packaged copy differs from the historical source pin')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--implementation-review', type=Path, required=True)
    parser.add_argument('--controller-review', type=Path, required=True)
    args = parser.parse_args()
    checked_reviews = {}
    reviewed_sources = {}
    for path, source_dir in ((args.implementation_review, ROUND / 'evaluation'),
                             (args.controller_review, ROUND / 'controller')):
        raw = path.read_bytes()
        review = json.loads(raw)
        checked_reviews[source_dir.name] = raw
        reviewed_sources[source_dir.name] = review.get('reviewed_sources', {})
        if review.get('status') != 'passed':
            raise ValueError('Huginn review has not passed')
        required = [p for p in source_dir.iterdir() if p.suffix in ('.py', '.sh')]
        if any(review.get('reviewed_sources', {}).get(str(p.resolve())) != record(p) for p in required):
            raise ValueError('Huginn review does not pin every packaged source')
    bundle = ROUND / 'bundle'
    bundle.mkdir(exist_ok=False)
    historical_spec_path = OLD / 'deployment/run_spec.json'
    historical_sha = '8115648b51ba6cd0f4c3b04bdfb36ae6935d8e47f1223cb1e54f7dd79215f149'
    historical_raw = historical_spec_path.read_bytes()
    if hashlib.sha256(historical_raw).hexdigest() != historical_sha:
        raise ValueError('Original scientific run specification changed')
    historical_spec = json.loads(historical_raw)
    roots = {'jlens': (REPO, bundle / 'repo'), 'ouro_src': (OURO_SRC, bundle / 'ouro_project/src')}
    for source in historical_spec['source_files']:
        original, output = roots[source['root']]
        path = original / source['path']
        if record(path)['sha256'] != source['sha256']:
            raise ValueError('Original source changed: ' + str(path))
        copy(path, output / source['path'], expected_sha256=source['sha256'])
    deployment = bundle / 'repo/research/refit_round_2026-09-07/deployment'
    for name in ('run_spec.json', 'environment.json', 'model_manifest.json', 'cloud_worker.py'):
        if not (deployment / name).exists():
            expected = {'run_spec.json': historical_sha,
                        'environment.json': historical_spec['environment']['sha256'],
                        'model_manifest.json': historical_spec['model']['manifest_sha256']}
            copy(OLD / 'deployment' / name, deployment / name, expected_sha256=expected.get(name))
    for fit in historical_spec['fits']:
        copy(OLD / 'audit' / Path(fit['prompts_path']).name,
             deployment.parent / 'audit' / Path(fit['prompts_path']).name, expected_sha256=fit['sha256'])
    for task in ('multihop', 'order-ops'):
        target = bundle / f'repo/data/evaluations/lens-eval-{task}.json'
        if not target.exists():
            copy(REPO / f'data/evaluations/lens-eval-{task}.json', target)
    for directory in ('evaluation', 'controller'):
        for path in sorted((ROUND / directory).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.sh'):
                copy(path, bundle / directory / path.name, expected=reviewed_sources[directory][str(path.resolve())])
    prerequisites = OLD / 'monitoring/attempt_06/ouro_handoff_20260909T110503Z/results'
    for name in ('ouro_evaluation', 'controls_evaluation'):
        for path in sorted((prerequisites / name).rglob('*')):
            if path.is_file():
                copy(path, bundle / 'prerequisites' / name / path.relative_to(prerequisites / name))
    frozen = bundle / 'frozen'
    copy(ROUND / 'VERIFICATION_PLAN.md', frozen / 'VERIFICATION_PLAN.md')
    for directory, name in (('evaluation', 'implementation_review.json'), ('controller', 'controller_review.json')):
        with (frozen / name).open('xb') as handle:
            handle.write(checked_reviews[directory])
        if record(frozen / name) != {'bytes': len(checked_reviews[directory]),
                'sha256': hashlib.sha256(checked_reviews[directory]).hexdigest()}:
            raise ValueError('Packaged review bytes differ from checked review')
    command = [sys.executable, str(deployment / 'run_huginn.py'),
               '--run-spec', str(deployment / 'run_spec.json'),
               '--combined-contract', str(deployment / 'combined_contract.json'),
               '--calibration', str(deployment / 'huginn_calibration.json'),
               '--snapshot', str(bundle / 'model_snapshot_not_loaded'),
               '--ouro-src', str(bundle / 'ouro_project/src'),
               '--ouro-evaluation-dir', str(bundle / 'prerequisites/ouro_evaluation'),
               '--controls-evaluation-dir', str(bundle / 'prerequisites/controls_evaluation'),
               '--output-dir', str(bundle / 'results_not_created'), '--plan']
    process = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    probe = json.loads(process.stdout)
    if (probe.get('status') != 'plan' or probe.get('model_loaded') is not False
            or probe.get('gpu_initialized') is not False
            or probe.get('evaluation_prerequisites', {}).get('status') != 'verified'
            or (bundle / 'results_not_created').exists()):
        raise ValueError('Original contract did not verify the actual packaged inputs')
    json_save(frozen / 'original_contract_check.json', probe)
    records = {p.relative_to(bundle).as_posix(): record(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    spec = {'schema': 'huginn_verification_run.v1', 'run_id': args.run_id,
            'frozen_utc': datetime.now(timezone.utc).isoformat(), 'category': 'newly_rerun_estimator',
            'historical_run_spec_record': record(deployment / 'run_spec.json'),
            'historical_bank_sha256': '7eddc849bca857a1406a901d8e4998b6bf90580b09f6ac6b326690ef378dc596',
            'n_prompts': 100, 'source_layers': list(range(32)), 'target_layer': 33, 'd_model': 5280,
            'readout_paths': READOUT_PATHS, 'evaluation_seeds': [2026090803, 2026090804],
            'readout_items': 148, 'source_records': records,
            'verification_plan_record': record(frozen / 'VERIFICATION_PLAN.md'),
            'admission': {'historical_prompt_bound_seconds': 130.48383641405962,
                          'measured_prompt_multiplier': 1.25, 'readout_reserve_seconds': 900},
            'comparison_policy': 'Exact historical bank hash and elementwise retained evaluation states/ranks/initializations; report discrepancies without another fitted lens.'}
    json_save(frozen / 'run_spec.json', spec)
    base = ['run_spec.json', 'provenance.json', 'logs/native_gate.log'] + [
        'gate/' + name for name in ('primal.pt', 'native_matrices.pt', 'optimized_matrices.pt',
                                   'START.json', 'huginn_r8_START.json', 'huginn_r8.json', 'COMPLETE.json')]
    fit = base + ['fit_admission.json', 'logs/fit_n100.log', 'fit/checkpoint.pt', 'fit/lens.pt', 'fit/evidence.json']
    all_paths = fit + ['logs/historical_readouts.log'] + ['readouts/' + relative for relative in READOUT_PATHS]
    contract = {'schema': 'confirmation_output_contract.v1', 'run_id': args.run_id,
                'run_spec_sha256': record(frozen / 'run_spec.json')['sha256'],
                'files': {p: {'role': 'Huginn actual numerical or sealed provenance evidence'} for p in all_paths},
                'stages': {'development': sorted(base), 'fit': sorted(fit), 'readouts': sorted(all_paths)},
                'required_checks': ['loadability', 'numerical']}
    json_save(frozen / 'output_contract.json', contract)
    config = {'schema': 'confirmation_run_config.v1', 'run_id': args.run_id,
              'run_spec_path': str(frozen / 'run_spec.json'),
              'output_contract_path': str(frozen / 'output_contract.json'),
              'semantic_verifier': {'path': str(bundle / 'evaluation/validate_outputs.py'),
                  'record': record(bundle / 'evaluation/validate_outputs.py'), 'callable': 'validate_outputs'},
              'setup_budget_seconds': 3600, 'compute_budget_seconds': 20880,
              'preservation_reserve_seconds': 4680, 'transfer_timeout_seconds': 4200}
    json_save(ROUND / 'controller/run_config.json', config)
    archive = ROUND / 'bundle.tar.gz'
    with tarfile.open(archive, 'x:gz') as target:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                target.add(path, arcname='bundle/' + path.relative_to(bundle).as_posix(), recursive=False)
    receipt = {'schema': 'huginn_verification_freeze.v1', 'run_id': args.run_id,
               'frozen_utc': spec['frozen_utc'], 'bundle': record(archive),
               'files': {p.relative_to(bundle).as_posix(): record(p) for p in sorted(bundle.rglob('*')) if p.is_file()},
               'run_config': record(ROUND / 'controller/run_config.json')}
    json_save(ROUND / 'FREEZE.json', receipt)
    print(json.dumps({'status': 'frozen', 'receipt': record(ROUND / 'FREEZE.json')}))

if __name__ == '__main__':
    main()
