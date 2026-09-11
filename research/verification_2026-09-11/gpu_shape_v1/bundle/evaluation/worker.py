#!/usr/bin/env python3
"""Scoring-shape check on the deployed GPU type: the confirmation runtime gate (M0), then shape_check.

Measured mismatches are data: once M0 passes, a run that finishes publishes a complete final manifest.
Any exception publishes failed; a stop publishes stopped.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import signal
import sys
import time

from artifacts import Publisher, atomic, json_save, npz_save, record, torch_save


class StopRequested(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work, bundle = Path(config['work']), Path(config['bundle'])
    specpath = bundle / 'frozen/run_spec.json'
    spec = json.loads(specpath.read_text())
    contract = json.loads((bundle / 'frozen/output_contract.json').read_text())
    binding = config['binding']
    if (record(specpath)['sha256'] != binding['run_spec_sha256']
            or record(bundle / 'frozen/output_contract.json')['sha256'] != binding['output_contract_sha256']):
        raise ValueError('Launch binding differs from frozen specification')
    publisher = Publisher(work, binding, contract)
    deadline = datetime.fromisoformat(config['work_deadline_utc'].replace('Z', '+00:00')).timestamp()
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def check_stop():
        if stopped or (work / 'STOP').exists() or time.time() >= deadline:
            raise StopRequested('Early stop/deadline')

    def verify_sources():
        for rel, expected in spec['source_records'].items():
            if record(bundle / rel) != expected:
                raise ValueError(f'Changed source: {rel}')

    def stage_status(phase):
        check_stop()
        publisher.status(phase, updated_utc=datetime.now(timezone.utc).isoformat())

    try:
        # M0: the confirmation worker's source, environment, snapshot, imported-module, precision and input checks.
        verify_sources()
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(bundle / 'repo'), str(bundle / 'ouro_project/src'), str(bundle / 'legacy')]
        import torch
        from ouro_jlens import evaluate as evaluator, evaldata, recurrent
        import evaluate_refits as common
        import run_refits as legacy
        import shape_check
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        env = json.loads((bundle / 'legacy/environment.json').read_text())
        legacy._environment(torch, env)
        snapshot = Path(config['snapshot'])
        model_files = legacy._snapshot(snapshot, json.loads((bundle / 'legacy/model_manifest.json').read_text()))
        imported = []
        for module in ('ouro_jlens.evaluate', 'ouro_jlens.evaldata', 'ouro_jlens.recurrent', 'jlens', 'jlens.hf',
                       'jlens.hooks', 'jlens.lens', 'jlens.vis', 'shape_check'):
            actual = Path(importlib.import_module(module).__file__).resolve()
            relative = actual.relative_to(bundle).as_posix()
            if relative not in spec['source_records'] or record(actual) != spec['source_records'][relative]:
                raise ValueError('Unpinned imported source')
            imported.append({'module': module, 'path': relative, 'record': record(actual)})
        stage_status('load_model')
        model, runtime = common._evaluation_runtime(torch, {'environment': env, 'spec': {'model': spec['model_identity']}},
                                                    model_files, snapshot, recurrent, imported)
        if runtime['precision'] != spec['original_precision']:
            raise ValueError('Readout precision switches differ from the original evaluation')
        for key, entry in spec['bank_records'].items():
            if record(work / entry['worker_path']) != entry['record']:
                raise ValueError(f'Input mismatch: {key}')
        atomic(publisher.root / 'run_spec.json', lambda f, raw=specpath.read_bytes(): f.write(raw))
        json_save(publisher.root / 'provenance.json', {
            'binding': binding, 'source_records': spec['source_records'], 'model_record': spec['model_record'],
            'input_records': spec['bank_records'], 'dtype': 'torch.bfloat16', 'runtime': runtime,
            'geometry': [model.n_ut, model.n_physical, model.n_layers, model.d_model],
            'actual_worker_source': record(Path(__file__)), 'config_record': record(args.config)})

        population = json.loads((bundle / 'frozen/population.json').read_text())
        items = []
        for row in population['rows']:
            ids, dropped = evaldata.readout_context(lambda t: model.encode(t)[0].tolist(), row['prompt'], row['target'])
            forms = {name: evaldata.single_token_ids(model.tokenizer, evaldata.surface_forms(name)) for name in row['intermediates']}
            if ids != row['token_ids'] or forms != row['intermediate_tokens'] or dropped > 1:
                raise ValueError('Worker tokenization differs from frozen population')
            items.append(evaldata.Item(row['name'], 'multihop', row['prompt'], row['target'], row['intermediates'], ids, forms,
                                       dict(zip(row['intermediates'], row['leaked']))))
        inputs = {key: torch.load(work / entry['worker_path'], map_location='cpu', weights_only=True)
                  for key, entry in spec['bank_records'].items()}
        states = inputs['states']['H'].float()
        comparisons = {}
        with common.task_names(evaluator, items, tasks=('multihop',)) as registry, torch.no_grad():
            names = registry['multihop']
            if names.names != population['names']['multihop']:
                raise ValueError('Worker name catalogue differs')
            stage_status('development_shapes')
            comparisons['development'], rows = shape_check.development(model, inputs['development'])
            torch_save(publisher.root / 'development/rows.pt', rows)
            stage_status('sample_rows')
            comparisons['samples'] = shape_check.samples(model, inputs['samples'])
            comparisons['layouts'] = {}
            for arm, arm_rows in (('raw', states), ('fit01', inputs['fit01_rows']['rows'])):
                stage_status(f'layouts_{arm}')
                arrays, comparisons['layouts'][arm] = shape_check.layouts(model, names, arm_rows, evaluator.MAX_NAMES, check_stop)
                npz_save(publisher.root / f'layouts/{arm}.npz',
                         {f'{layout}__{key}': value for layout, group in arrays.items() for key, value in group.items()})
                del arrays
            stage_status('native_outputs')
            torch_save(publisher.root / 'native/exits_from_retained_states.pt',
                       model.unembed(states[:, 191].to(model.input_device)).cpu())
            native_logits, regenerated, comparisons['native'] = shape_check.native(
                model, evaluator, names, items, states, inputs['states']['continuations'], check_stop)
            torch_save(publisher.root / 'native/native_logits.pt', native_logits)
            torch_save(publisher.root / 'native/regenerated_states.pt', {'H': regenerated})
        json_save(publisher.root / 'comparisons.json', comparisons)
        verify_sources()
        publisher.publish()
        publisher.status('computation_finished', outcome='complete')
    except BaseException as error:
        outcome = 'stopped' if isinstance(error, StopRequested) else 'failed'
        try:
            publisher.publish(outcome=outcome)
            publisher.status('computation_finished', outcome=outcome, error_type=type(error).__name__, error=str(error))
        finally:
            raise


if __name__ == '__main__':
    main()
