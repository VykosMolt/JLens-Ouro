#!/usr/bin/env python3
"""Run the scoring-shape check on this machine's GPU: independent evidence, not the deployed RTX 5090 runtime.

  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTORCH_ALLOC_CONF=expandable_segments:True python -B run_local.py --out DIR [--items N]
"""
import argparse
from datetime import datetime, timezone
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'replay'))
sys.path.insert(0, str(HERE / 'bundle/evaluation'))
import replay_scorer as rs  # noqa: E402
import shape_check  # noqa: E402


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def saved_torch(path, value):
    with path.open('xb') as handle:
        torch.save(value, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--items', type=int, default=160)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'layouts').mkdir()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    run = {'schema': 'shape_check_local.v1', 'started_utc': datetime.now(timezone.utc).isoformat(), 'items': args.items,
           'scripts': {name: rs.sha256(path) for name, path in (('run_local.py', Path(__file__)), ('shape_check.py', Path(shape_check.__file__)))}}
    inputs = json.loads((HERE / 'inputs/INPUTS.json').read_text())
    run['inputs'] = {name: rs.verified(HERE / 'inputs' / name, value) for name, value in inputs['files'].items()}
    manifest = json.loads((rs.BUNDLE / 'legacy/model_manifest.json').read_text())
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(rs.BUNDLE / part) for part in ('repo', 'ouro_project/src', 'legacy')]
    from ouro_jlens import evaluate as evaluator, evaldata, recurrent
    import evaluate_refits as common
    import run_refits
    run['snapshot_files_verified'] = len(run_refits._snapshot(rs.SNAPSHOT, manifest))
    model = recurrent.load_ouro(path=rs.SNAPSHOT, device='cuda:0', dtype=torch.bfloat16)
    spec = json.loads((rs.RESULTS / 'run_spec.json').read_text())
    run['precision'] = run_refits._precision(torch)
    run['precision_matches_original'] = run['precision'] == spec['original_precision']
    properties = torch.cuda.get_device_properties(0)
    run['runtime'] = {'gpu': properties.name, 'capability': [properties.major, properties.minor], 'python': platform.python_version(),
                      'torch': torch.__version__, 'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
                      'attention': str(model.hf_model.config._attn_implementation)}

    population = json.loads((rs.RESULTS / 'population.json').read_text())
    everyone = [evaldata.Item(r['name'], 'multihop', r['prompt'], r['target'], r['intermediates'], r['token_ids'],
                              r['intermediate_tokens'], dict(zip(r['intermediates'], r['leaked']))) for r in population['rows']]
    n = args.items
    states = torch.load(HERE / 'inputs/states.pt', map_location='cpu', weights_only=True)
    H, continuations = states['H'][:n].float(), states['continuations']
    fit01 = torch.load(HERE / 'inputs/fit01_rows.pt', map_location='cpu', weights_only=True)['rows'][:n]
    samples = torch.load(HERE / 'inputs/samples.pt', map_location='cpu', weights_only=True)
    development = torch.load(HERE / 'inputs/development.pt', map_location='cpu', weights_only=True)
    saved_exits = torch.load(rs.RESULTS / 'common/cache.pt', map_location='cpu', weights_only=True)['exit_logits'][:n]
    stop = lambda: None  # noqa: E731
    with common.task_names(evaluator, everyone, tasks=('multihop',)) as registry, torch.no_grad():
        names = registry['multihop']
        exits = model.unembed(H[:, 191].to(model.input_device)).cpu()
        run['exits_[n,2048]_vs_saved_exit3'] = shape_check.compare(exits.float(), saved_exits[:, 3])
        run['layouts'] = {}
        for arm, rows in (('raw', H), ('fit01', fit01)):
            arrays, summary = shape_check.layouts(model, names, rows, evaluator.MAX_NAMES, stop)
            with np.load(rs.RESULTS / f'readouts/{arm}.npz', allow_pickle=False) as saved:
                run['layouts'][arm] = {'logits_vs_executed': summary,
                                       'executed_allrank_equals_saved': bool(np.array_equal(arrays['executed']['allrank'], saved['allrank'][:n])),
                                       'executed_top10_equals_saved': bool(np.array_equal(arrays['executed']['top10'], saved['top10_ids'][:n]))}
            for layout, value in arrays.items():
                np.savez_compressed(args.out / 'layouts' / f'{arm}_{layout}.npz', **value)
            del arrays
        run['samples'] = shape_check.samples(model, samples)
        run['development'], rows = shape_check.development(model, development)
        saved_torch(args.out / 'development_rows.pt', rows)
        native_logits, regenerated, run['native'] = shape_check.native(model, evaluator, names, everyone[:n], H, continuations, stop)
        saved_torch(args.out / 'native_logits.pt', native_logits)
        saved_torch(args.out / 'regenerated_states.pt', {'H': regenerated, 'source': 'regenerated on ' + properties.name})
    run['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(args.out / 'run.json', run)
    print(json.dumps({'status': 'complete', 'out': str(args.out)}))


if __name__ == '__main__':
    main()
