#!/usr/bin/env python3
"""A4: run the worker's M1-M4 code end to end on CPU with the real local snapshot.

Pipeline validation only: CPU kernels say nothing about GPU bit behaviour, and
M0's CUDA runtime load cannot run here.
Usage: python -B tests/cpu_pipeline.py --work EMPTY_DIRECTORY
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from common import SNAPSHOT, copy_bundle, empty_directory, record, require, save_evidence

MIN_AVAILABLE_GIB = 16
PAYLOADS = ('development/native.pt', 'diagnostic/tensors.pt', 'diagnostic/comparisons.json')


def available_gib():
    line = next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    return int(line.split()[1]) / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    work = empty_directory(parser.parse_args().work)
    memory = available_gib()
    require(memory >= MIN_AVAILABLE_GIB, f'{memory:.1f} GiB available; the contract fallback model is required')
    bundle = work / 'bundle'
    files = copy_bundle(bundle)
    # The bootstrap environment, so the precision dictionary is read under the same variables.
    os.environ.update({'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
                       'HF_HOME': str(work / 'huggingface'), 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TOKENIZERS_PARALLELISM': 'false'})
    sys.path[:0] = [str(bundle / 'evaluation'), str(bundle / 'repo'), str(bundle / 'ouro_project/src'), str(bundle / 'legacy')]
    import torch
    from ouro_jlens import evaluate as evaluator, recurrent
    import run_refits as legacy
    import validate_outputs
    import worker
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    started = time.time()
    spec = json.loads((bundle / 'frozen/run_spec.json').read_text())
    legacy._environment(torch, json.loads((bundle / 'legacy/environment.json').read_text()))
    legacy._snapshot(SNAPSHOT, json.loads((bundle / 'legacy/model_manifest.json').read_text()))
    model = recurrent.load_ouro(path=SNAPSHOT, device='cpu', dtype=torch.bfloat16)
    allitems = evaluator.load_items(model.tokenizer, ('multihop',), encode=lambda t: model.encode(t)[0].tolist())
    items = [item for name in spec['development_item_names'] for item in allitems if item.name == name]
    require([item.name for item in items] == spec['development_item_names'], 'development items changed')
    measured = time.time()
    native, tensors = worker.measure(model, items, check_stop=lambda: None)
    results = work / 'results'
    worker.save_outputs(results, native, tensors)
    elapsed = time.time() - measured
    native = torch.load(results / PAYLOADS[0], map_location='cpu', weights_only=True, mmap=True)
    tensors = torch.load(results / PAYLOADS[1], map_location='cpu', weights_only=True, mmap=True)
    validate_outputs.check_payloads(native, tensors, spec['development_item_names'])
    recorded = json.loads((results / PAYLOADS[2]).read_text())
    require(recorded == validate_outputs.comparisons(native, tensors), 'saved comparison record differs from its recomputation')
    payloads = {name: record(results / name) for name in PAYLOADS}
    save_evidence('A4_cpu_pipeline.json', {
        'schema': 'native_logit_cpu_pipeline.v1', 'status': 'passed',
        'label': 'Pipeline validation only: CPU kernels, real local snapshot. Not GPU evidence; the CUDA runtime load of M0 did not run.',
        'bundle_records': files, 'python': sys.executable, 'torch': torch.__version__,
        'memory_available_gib_before_load': memory, 'environment_check': 'passed', 'snapshot_check': 'passed',
        'precision_equals_frozen_original': legacy._precision(torch) == spec['original_precision'],
        'token_counts': {name: len(entry['token_ids']) for name, entry in tensors['items'].items()},
        'exit_steps': recorded['exit_steps'], 'payloads': payloads,
        'payload_bytes_total': sum(value['bytes'] for value in payloads.values()),
        'comparisons': len(recorded['comparisons']),
        'cpu_unequal_comparisons': [r['name'] for r in recorded['comparisons'] if not r['torch_equal']],
        'measurement_seconds': elapsed, 'total_seconds': time.time() - started, 'results_directory': str(results)})
    print(json.dumps({'status': 'passed', 'payload_bytes_total': sum(v['bytes'] for v in payloads.values())}))


if __name__ == '__main__':
    main()
