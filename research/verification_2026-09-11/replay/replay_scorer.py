#!/usr/bin/env python3
"""Readout-only replay of the confirmation scorer from retained states and hash-verified banks.

The frozen producer code builds the transport and applies the executed readout. A layout changes
only how the transported rows reach the head (packing) or the head's arithmetic; nothing is fitted.

  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTORCH_ALLOC_CONF=expandable_segments:True python replay_scorer.py --device cuda --out DIR
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

RESEARCH = Path('/home/moloch/jacobian-lens/research')
ROUND = RESEARCH / 'confirmation_2026-09-09'
BUNDLE = ROUND / 'corrections/native_check_v1/bundle'
FINAL = ROUND / ('cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/'
                 'ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef')
RESULTS = FINAL / 'results'
SNAPSHOT = Path('/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/'
                '1ed04250da1a9936042725d302e81c8fa2ab5abd')
REFIT = RESEARCH / 'refit_round_2026-09-07/cloud_leases/attempt_06'
# Copies kept by the 2026-09-11 purge, keyed by run_spec worker path; hashes are checked before use.
BANK_FILES = {
    'banks/fit01.pt': REFIT / ('prefetch_v3_20260909T134030Z_649dff12/staging/ouro/fit_01/final/'
                               'cursor_000100_e090a066bc3b4630891f195e7e7d1d29/lens.pt'),
    'banks/fit02.pt': ROUND / 'artifacts/reconstructed/ouro/fit_02/lens.pt',
    'banks/penultimate.pt': REFIT / ('retrieved/controls/ouro_penultimate/fit_01/final/'
                                     'cursor_000100_6d4d010756ab4a1cbca52509c897410c/lens.pt'),
    'banks/positions.pt': REFIT / ('retrieved/controls/ouro_positions/fit_01/final/'
                                   'cursor_000100_435bacee56d94ff988ff069b59f2c987/lens.pt'),
}
ARMS = ('raw', 'fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')
LAYOUTS = ('executed', 'row1', 'rows190', 'rows191', 'rows192', 'col160', 'fp32', 'fp64_head', 'fp64')
VALUE_LAYOUTS = {'executed', 'fp32', 'fp64_head', 'fp64'}
EXIT_COLUMNS = (47, 95, 143, 191)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, expected):
    actual = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    if actual != {'bytes': expected['bytes'], 'sha256': expected['sha256']}:
        raise ValueError(f'hash mismatch: {path}')
    return {'path': str(path), **actual}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {'equal': False, 'shapes': [list(actual.shape), list(expected.shape)],
                'dtypes': [str(actual.dtype), str(expected.dtype)]}
    differ = actual != expected
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {'equal': not bool(differ.any()), 'differing': int(differ.sum()), 'max_abs': float(delta.max(initial=0))}


class Stub:
    """The model attributes read by the frozen transport and readout code."""
    n_ut, n_physical, n_layers, d_model = 4, 48, 192, 2048

    def __init__(self, device, unembed):
        self.input_device = torch.device(device)
        self.unembed = unembed


class Head:
    """jlens.hf unembed for Ouro: OuroRMSNorm (FP32 inside, cast back) then the bias-free lm_head."""

    def __init__(self, device, eps):
        with safe_open(SNAPSHOT / 'model.safetensors', 'pt') as tensors:
            self.weight = tensors.get_tensor('lm_head.weight').to(device)
            self.gain = tensors.get_tensor('model.norm.weight').to(device)
        if self.weight.dtype != torch.bfloat16 or self.gain.dtype != torch.bfloat16:
            raise ValueError('head weights are not BF16')
        self.eps, self.casts = eps, {}

    def bf16(self, residual):
        states = residual.to(torch.bfloat16).to(self.weight.device).to(torch.float32)
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self.eps)
        return F.linear(self.gain * states.to(torch.bfloat16), self.weight)

    def exact(self, residual, dtype):
        if dtype not in self.casts:
            self.casts[dtype] = (self.weight.to(dtype), self.gain.to(dtype))
        weight, gain = self.casts[dtype]
        states = residual.to(self.weight.device).to(dtype)
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self.eps)
        return F.linear(gain * states, weight)


def packed(head, rows, layout):
    """Logits for one item's [n, d] rows; each row keeps its values and only its batch changes."""
    n = len(rows)
    if layout == 'executed':
        return head.bf16(rows)
    if layout == 'row1':
        return torch.cat([head.bf16(rows[r:r + 1]) for r in range(n)])
    if layout.startswith('rows'):
        size = int(layout[4:])
        if n == size:
            return head.bf16(rows)
        if n < size:  # pad with the item's own leading rows
            return head.bf16(torch.cat([rows, rows[:size - n]]))[:n]
        return torch.cat([head.bf16(rows[:size]), head.bf16(rows[n - size:])[2 * size - n:]])
    if layout == 'fp32':
        return head.exact(rows, torch.float32)
    if layout == 'fp64_head':
        return head.exact(rows, torch.float64)
    raise ValueError(layout)


def rank_rows(names, logits, max_names):
    """The _ranks_of rule (descending argsort, min over alias forms), its top-10, tie bounds and boundary values."""
    n, vocab = logits.shape
    order = logits.argsort(dim=-1, descending=True)
    descending = logits.gather(-1, order)
    position = torch.empty_like(order)
    position.scatter_(1, order, torch.arange(vocab, device=logits.device).expand_as(order))
    ids = names.flat_ids.to(logits.device)
    values = logits[:, ids].contiguous()
    ascending = descending.flip(-1).contiguous()
    greater = vocab - torch.searchsorted(ascending, values, right=True)
    at_least = vocab - torch.searchsorted(ascending, values, right=False)
    stacked = torch.stack([position[:, ids], greater, at_least - 1]).cpu().numpy()
    values = values.double().cpu().numpy()
    out = {key: np.full((max_names, n), -1, np.int32) for key in ('allrank', 'best', 'worst')}
    out['value'] = np.full((max_names, n), np.nan)
    for j, (a, b) in enumerate(zip(names.slices[:-1], names.slices[1:])):
        for k, key in enumerate(('allrank', 'best', 'worst')):
            out[key][j] = stacked[k, :, a:b].min(1)
        out['value'][j] = values[:, a:b].max(1)
    out['top10'] = order[:, :10].to(torch.int32).cpu().numpy()
    out['boundary'] = descending[:, 9:11].double().cpu().numpy()
    return out


def difference(logits, reference):
    """Rows and logits that differ from the executed logits, maximum absolute and summed squared difference."""
    if logits.dtype == reference.dtype:
        differ = logits != reference
        delta = logits.float() - reference.float()
    else:
        delta = logits.double() - reference.double()
        differ = delta != 0
    return [int(differ.any(-1).sum()), int(differ.sum()), float(delta.abs().max()), float(delta.double().pow(2).sum())]


class Layout:
    def __init__(self, n, columns, max_names, values):
        shape = (n, max_names, columns)
        self.arrays = {'allrank': np.full(shape, -1, np.int32), 'best': np.full(shape, -1, np.int32),
                       'worst': np.full(shape, -1, np.int32), 'top10': np.zeros((n, columns, 10), np.int32),
                       'difference': np.full((n, 4), np.nan)}
        if values:
            self.arrays['value'] = np.full(shape, np.nan)
            self.arrays['boundary'] = np.full((n, columns, 2), np.nan)

    def add(self, i, ranks, delta):
        for key in ('allrank', 'best', 'worst', 'value'):
            if key in self.arrays:
                self.arrays[key][i] = ranks[key]
        self.arrays['top10'][i] = ranks['top10']
        if 'boundary' in self.arrays:
            self.arrays['boundary'][i] = ranks['boundary']
        if delta is not None:
            self.arrays['difference'][i] = delta

    def summary(self, columns, vocab):
        d = self.arrays['difference']
        if np.isnan(d).any():
            return None
        return {'rows_differing': int(d[:, 0].sum()), 'rows_total': int(len(d) * columns),
                'logits_differing': int(d[:, 1].sum()), 'max_abs': float(d[:, 2].max()),
                'rms': float(np.sqrt(d[:, 3].sum() / (len(d) * columns * vocab)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', choices=('cuda', 'cpu'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--layouts', default=None, help='default: all on CUDA, fp64 on CPU')
    parser.add_argument('--items', type=int, default=160, help='first N items (smoke tests)')
    parser.add_argument('--states', type=Path, default=None,
                        help='regenerated states {"H": BF16 [160,192,2048]} instead of the retained cache (level-3 check)')
    args = parser.parse_args()
    arms = args.arms.split(',')
    layouts = args.layouts.split(',') if args.layouts else (list(LAYOUTS) if args.device == 'cuda' else ['fp64'])
    if set(arms) - set(ARMS) or set(layouts) - set(LAYOUTS):
        raise ValueError('unknown arm or layout')
    args.out.mkdir(parents=True, exist_ok=False)
    for layout in layouts:
        (args.out / layout).mkdir()
    torch.set_num_threads(8 if args.device == 'cuda' else 24)
    torch.set_num_interop_threads(1)
    run = {'schema': 'scorer_replay.v1', 'started_utc': utc(), 'argv': sys.argv, 'device': args.device,
           'layouts': layouts, 'arms': arms, 'items': args.items,
           'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__))}}

    manifest = json.loads((FINAL / 'MANIFEST.json').read_text())['files']
    inputs = ['common/cache.pt', 'population.json', 'run_spec.json',
              *(f'readouts/{arm}{suffix}' for arm in ARMS for suffix in ('.npz', '_samples.pt'))]
    run['inputs'] = {rel: verified(RESULTS / rel, manifest[rel]) for rel in inputs}
    spec = json.loads((RESULTS / 'run_spec.json').read_text())
    run['banks'] = {entry['worker_path']: verified(BANK_FILES[entry['worker_path']], entry['record'])
                    for entry in spec['bank_records'].values()}
    model_files = {f['path']: f for f in json.loads((BUNDLE / 'legacy/model_manifest.json').read_text())['files']}
    run['model_files'] = {name: verified(SNAPSHOT / name, model_files[name]) for name in ('model.safetensors', 'config.json')}
    for rel, expected in spec['source_records'].items():
        verified(BUNDLE / rel, expected)
    eps = json.loads((SNAPSHOT / 'config.json').read_text())['rms_norm_eps']

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(BUNDLE / part) for part in ('repo', 'ouro_project/src', 'legacy', 'evaluation')]
    import jlens
    from ouro_jlens import evaluate as evaluator, evaldata
    import evaluate_refits as common
    import run_refits
    import readouts as frozen
    import measurement
    imported = {}
    for module in (jlens, sys.modules['jlens.lens'], sys.modules['jlens.vis'], evaluator, evaldata, common, run_refits, frozen, measurement):
        rel = Path(module.__file__).resolve().relative_to(BUNDLE).as_posix()
        imported[module.__name__] = {'path': rel, 'in_source_records': rel in spec['source_records']}
    run['imported'] = imported
    run['precision'] = run_refits._precision(torch)
    run['precision_matches_original'] = run['precision'] == spec['original_precision']
    run['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_git': torch.version.git_version,
                      'cuda': torch.version.cuda, 'numpy': np.__version__}
    if args.device == 'cuda':
        properties = torch.cuda.get_device_properties(0)
        run['runtime']['gpu'] = {'name': properties.name, 'capability': [properties.major, properties.minor],
                                 'total_memory': properties.total_memory}
        if not run['precision_matches_original']:
            raise ValueError('precision settings differ from the original evaluation')

    population = json.loads((RESULTS / 'population.json').read_text())
    everyone = [evaldata.Item(r['name'], 'multihop', r['prompt'], r['target'], r['intermediates'], r['token_ids'],
                              r['intermediate_tokens'], dict(zip(r['intermediates'], r['leaked'])))
                for r in population['rows']]
    items = everyone[:args.items]
    n = len(items)
    cache = torch.load(RESULTS / 'common/cache.pt', map_location='cpu', weights_only=True)
    states, exits = cache['H'][:n], cache['exit_logits'][:n]
    run['states'] = {'source': 'retained common/cache.pt'}
    if args.states:
        run['states'] = {'source': 'regenerated', 'path': str(args.states.resolve()), 'sha256': sha256(args.states)}
        states, exits = torch.load(args.states, map_location='cpu', weights_only=True)['H'][:n].float(), None
    run['arms_detail'] = {}
    with common.task_names(evaluator, everyone, tasks=('multihop',)) as registry, torch.no_grad():
        names = registry['multihop']
        if names.names != population['names']['multihop']:
            raise ValueError('name catalogue differs from the population')
        head = Head(args.device, eps)
        vocab = head.weight.shape[0]
        if exits is None:  # the executed exits layout over the regenerated states
            exits = torch.stack([head.bf16(states[:, v].to(args.device)).float().cpu() for v in EXIT_COLUMNS], dim=1)
        device_exits = exits.to(args.device)
        for arm in arms:
            started = time.time()
            columns = len(measurement.SUPPORT[arm])
            detail = {'columns': columns}
            saved = {k: v[:n] for k, v in np.load(RESULTS / f'readouts/{arm}.npz', allow_pickle=False).items()}
            saved_samples = torch.load(RESULTS / f'readouts/{arm}_samples.pt', map_location='cpu', weights_only=True)
            bank, J = None, None
            if arm != 'raw':
                entry = spec['bank_records'][arm]
                state = torch.load(BANK_FILES[entry['worker_path']], map_location='cpu', weights_only=True, mmap=True)
                bank = state['J'][arm] if arm in ('sampled_sum', 'diagonal') else state['J']
                target = 190 if arm == 'penultimate' else 191
                if sorted(bank) != list(range(target)):
                    raise ValueError('bank sources differ from the frozen target')
                if layouts != ['fp64']:
                    lens = jlens.JacobianLens(bank, n_prompts=100, d_model=2048)
                    plan = common.prepare_readout(Stub(args.device, None), lens, target_layer=target,
                                                  source_layers=list(range(target)), evaluator=evaluator)
                    J = plan.jacobians[:columns]
                    del lens, plan

            if 'executed' in layouts:  # the frozen readout itself, compared with every saved array and sample
                outputs, samples = frozen.readout(evaluator, Stub(args.device, head.bf16), items, states, J,
                                                  device_exits, columns)
                detail['frozen_readout_vs_saved'] = {key: compare(outputs[key], saved[key]) for key in sorted(saved)}
                detail['frozen_samples_vs_saved'] = [
                    {'item_index': a['item_index'], 'virtual_indices': a['virtual_indices'] == b['virtual_indices'],
                     **{key: compare(a[key].numpy(), b[key].numpy()) for key in ('transported', 'logits', 'sorted_ids')}}
                    for a, b in zip(samples, saved_samples)]
                np.savez_compressed(args.out / f'frozen_readout_{arm}.npz', **outputs)
                del outputs, samples

            T = None
            if layouts != ['fp64']:  # the executed transport expression, once per item
                T = torch.empty(n, columns, 2048)
                for i in range(n):
                    h = states[i, :columns].to(args.device)
                    T[i] = (h if J is None else torch.einsum('vde,ve->vd', J, h)).cpu()
                detail['transport_vs_saved_samples'] = [
                    compare(T[s['item_index'], s['virtual_indices']].numpy(), s['transported'].numpy())
                    for s in saved_samples if s['item_index'] < n]
            del J
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            T64 = None
            if 'fp64' in layouts:  # FP16 bank and BF16-valued states in exact FP64 arithmetic
                J64 = None
                if bank is not None:
                    J64 = torch.eye(2048, dtype=torch.float64).repeat(columns, 1, 1)
                    for v in range(min(target, columns)):
                        J64[v] = bank[v].to(torch.float64)
                    J64 = J64.to(args.device)
                T64 = torch.empty(n, columns, 2048, dtype=torch.float64)
                for i in range(n):
                    h = states[i, :columns].to(args.device).to(torch.float64)
                    T64[i] = (h if J64 is None else torch.einsum('vde,ve->vd', J64, h)).cpu()
                del J64
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            del bank

            results = {layout: Layout(n, columns, evaluator.MAX_NAMES, layout in VALUE_LAYOUTS) for layout in layouts}
            frozen_rule_mismatch = 0
            for i in range(n):
                rows = None if T is None else T[i].to(args.device)
                reference = None if rows is None or args.device != 'cuda' else head.bf16(rows)
                for layout in layouts:
                    if layout == 'col160':
                        continue
                    if layout == 'fp64':
                        logits = head.exact(T64[i], torch.float64)
                    elif layout == 'executed' and reference is not None:
                        logits = reference
                    else:
                        logits = packed(head, rows, layout)
                    scored = logits.float() if logits.dtype == torch.bfloat16 else logits
                    ranks = rank_rows(names, scored, evaluator.MAX_NAMES)
                    if layout == 'executed' and not np.array_equal(ranks['allrank'][:len(names.names)], names.ranks(scored)):
                        frozen_rule_mismatch += 1
                    results[layout].add(i, ranks, None if reference is None else difference(logits, reference))
            detail['staged_rank_rule_mismatched_items'] = frozen_rule_mismatch

            if 'col160' in layouts:  # every virtual column batched across items, as the executed exits were
                grid = torch.empty(n, columns, vocab, dtype=torch.bfloat16)
                for v in range(columns):
                    grid[:, v] = head.bf16(T[:, v].to(args.device)).cpu()
                if arm == 'raw':
                    detail['col160_vs_saved_exit_logits'] = {
                        str(v): compare(grid[:, v].float().numpy(), exits[:, u].numpy()) for u, v in enumerate(EXIT_COLUMNS)}
                for i in range(n):
                    logits = grid[i].to(args.device)
                    ranks = rank_rows(names, logits.float(), evaluator.MAX_NAMES)
                    results['col160'].add(i, ranks, difference(logits, head.bf16(T[i].to(args.device))))
                del grid

            detail['layouts'] = {}
            for layout, result in results.items():
                np.savez_compressed(args.out / layout / f'{arm}.npz', **result.arrays)
                detail['layouts'][layout] = {
                    'logits_vs_executed': result.summary(columns, vocab),
                    'allrank_vs_saved': compare(result.arrays['allrank'], saved['allrank']),
                    'top10_vs_saved': compare(result.arrays['top10'], saved['top10_ids'])}
            del T, T64, results
            detail['seconds'] = round(time.time() - started, 1)
            run['arms_detail'][arm] = detail
            write_json(args.out / f'arm_{arm}.json', detail)
            print(json.dumps({'arm': arm, 'seconds': detail['seconds']}), flush=True)
    run['finished_utc'] = utc()
    write_json(args.out / 'run.json', run)


if __name__ == '__main__':
    main()
