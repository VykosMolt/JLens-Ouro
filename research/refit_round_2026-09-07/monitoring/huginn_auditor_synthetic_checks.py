"""Local synthetic mechanism checks; no scientific result inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

spec = importlib.util.spec_from_file_location('independent_huginn_audit', '/tmp/audit_huginn_actual_results.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, operation):
    try:
        operation()
    except (ValueError, AssertionError):
        checks.append(name)
    else:
        raise AssertionError(name + ' did not reject')


plan = json.loads((a.ROUND / 'deployment/huginn_eligibility.json').read_text())
original = [{**{key: row[key] for key in ('name', 'task', 'prompt', 'target', 'intermediates', 'own_index')},
             **row['ouro']} for row in plan['rows']]
slots = a.common_population(plan, original, plan['task_names'], plan['token_forms']['ouro'])
check('frozen metadata population only: 139 eligible items', sum(bool(s) for s in slots) == 139)
bad_plan = copy.deepcopy(plan)
bad_plan['rows'][0]['control_indices'][0].append(0)
rejects('all own names excluded from controls', lambda: a.common_population(bad_plan, original, plan['task_names'], plan['token_forms']['ouro']))
bad_plan = copy.deepcopy(plan)
bad_plan['rows'][0]['huginn']['leaked'][0] = True
rejects('recorded leakage independently recomputed', lambda: a.common_population(bad_plan, original, plan['task_names'], plan['token_forms']['ouro']))

# Two eligible own slots, two matched controls. Each name succeeds in a
# different location, distinguishing mean(any(name)) from max(mean(names)).
tiny = [{'own_index': [0, 1, -1], 'control_indices': [[2, 3], [2, 3]]},
        {'own_index': [4, -1, -1], 'control_indices': [[2, 3]]},
        {'own_index': [5, -1, -1], 'control_indices': [[2, 3]]}]
ranks = np.full((3, 128, 32), 50, np.int32)
ranks[0, 0, 3] = 0; ranks[0, 1, 7] = 0
ranks[0, 2, 3] = 0; ranks[0, 3, 7] = 0
ranks[1, 2, 3] = 0
bank = a.reduce_ranks(ranks, 'huginn', tiny, [[0, 1], [0], []])
check('slot averages preserve item weights', bank['own_layer'][0, 3] == .5 and bank['own_layer'][0, 7] == .5)
check('any event per own/control name before means', bank['own_regions'][0, 2] == 1 and bank['control_regions'][0, 2] == 1
      and bank['control_layer'][0].max() == .5)
check('matched-control denominator and signed excess', bank['control_regions'][1, 2] == .5 and bank['delta_regions'][1, 2] == -.5)
check('ineligible stored items remain zero and masked', not bank['eligible'][2] and not np.any(bank['own_layer'][2])
      and not np.any(bank['control_regions'][2]))
corrupt = ranks.copy(); corrupt[0, 0, 3] = -1
rejects('negative eligible rank cannot become hit or denominator miss', lambda: a.reduce_ranks(corrupt, 'huginn', tiny, [[0, 1], [0], []]))

weights = a.draw_weights(np.array([[0, 1], [0, 0], [1, 1]]), np.array([0, 0, 1]))
check('unequal component sizes use item-weighted ratio means', np.allclose(weights, [[1/3, 1/3, 1/3], [.5, .5, 0], [0, 0, 1]]))
rows = [{'intermediates': ['three']}, {'intermediates': ['03', 'different']},
        {'intermediates': ['third']}, {'intermediates': ['different']}]
groups = a.components(rows, np.arange(4), [[0], [0, 1], [0], [0]])
check('normalized shared concepts connect transitively', np.array_equal(groups, [0, 0, 0, 0]))


def make_readout(model, method, seed=None):
    columns = len(a.SUPPORT[model])
    ranks = np.full((148, 128, columns), -1, np.int32)
    for i, row in enumerate(plan['rows']):
        names = plan['common_name_indices'][row['task']] if model == 'huginn' else range(len(plan['task_names'][row['task']]))
        ranks[i, list(names)] = 50
    # Only item0 (one Brazil slot, no alias controls) carries an effect.
    if model == 'ouro' and method == 'jlens':
        ranks[0, 0, 23] = 0
    if model == 'huginn' and method == 'jlens' and seed == a.SEEDS[0]:
        ranks[0, 0, 3] = 0
    if model == 'huginn' and method == 'coda':
        ranks[0, 0, 7] = 0
    own = np.full((148, 3, columns), -1, np.int32)
    for i, row in enumerate(plan['rows']):
        for slot, index in enumerate(row['own_index']):
            if index >= 0:
                own[i, slot] = ranks[i, index]
    top = np.full((148, columns), 1 if model == 'ouro' or method == 'raw' else 2, np.int64)
    if model == 'ouro' and method == 'jlens':
        top[:] = 2; top[:, 191] = 1
    if model == 'huginn':
        if method == 'coda':
            top[:] = 0; top[:, 0] = 3
        if method == 'raw':
            top[:, 0] = 3
    return {'allrank': ranks, 'rank': own, 'top1': top}


arrays = {'ouro': {method: make_readout('ouro', method) for method in a.METHODS['ouro']}, 'huginn': {}}
arrays['ouro']['exit_top1'] = arrays['ouro']['raw']['top1'][:, a.EXITS['ouro']].copy()
for seed in a.SEEDS:
    arrays['huginn'][seed] = {method: make_readout('huginn', method, seed) for method in a.METHODS['huginn']}
    arrays['huginn'][seed]['native_top1'] = np.zeros(148, np.int64)
for model in a.METHODS:
    for seed in ((None,) if model == 'ouro' else a.SEEDS):
        for method in a.METHODS[model]:
            a.check_arrays(arrays[model][method] if seed is None else arrays[model][seed][method], model, plan)
check('both native geometries, original masks and identical aliases accepted', True)
valid = arrays['huginn'][a.SEEDS[0]]['raw']
corrupt = {key: value.copy() for key, value in valid.items()}; corrupt['allrank'][0, 37, 0] = 0
rejects('unsupported interior catalogue gap rejected', lambda: a.check_arrays(corrupt, 'huginn', plan))
corrupt = {key: value.copy() for key, value in valid.items()}; corrupt['allrank'][0, 2, 0] = 65536
rejects('out-of-vocabulary supported rank rejected', lambda: a.check_arrays(corrupt, 'huginn', plan))
corrupt = {key: value.copy() for key, value in valid.items()}; corrupt['top1'] = corrupt['top1'].astype(np.int32)
rejects('top1 dtype mismatch rejected', lambda: a.check_arrays(corrupt, 'huginn', plan))
corrupt = {key: value.copy() for key, value in valid.items()}; corrupt['allrank'][0, 36, 0] = 0
rejects('identical token-form alias rank inconsistency rejected', lambda: a.check_arrays(corrupt, 'huginn', plan))

saved, result = a.reconstruct(arrays, plan, slots)
cross = np.array([1/64 - 1/191, -1/14, -.5])
check('three scalars use191 learned Ouro sources and average scored seeds',
      np.allclose(saved['cross_model_seed_average_metrics'][0], cross, rtol=0, atol=1e-15)
      and not np.any(saved['cross_model_seed_average_metrics'][1:]))
check('task item means exclude nine masked rows', np.allclose(result['comparison']['tasks']['multihop']['mean'], cross/88)
      and not np.any(result['comparison']['tasks']['order-ops']['mean']))
check('seed average does not union recovery events', saved['huginn_2026090803_jlens_own_metrics'][0, 2] == 1
      and saved['huginn_2026090804_jlens_own_metrics'][0, 2] == 0
      and result['readouts']['huginn']['seed_average']['multihop']['jlens']['own_regions'][2] == .5/88)
check('joint component counts and exact full archive membership', len(saved) == 135
      and result['populations']['multihop']['component_count'] == 62
      and result['populations']['order-ops']['component_count'] == 14)
chosen = saved['bootstrap_multihop_item_indices']
proportions = (chosen == 0).sum(axis=1) / 88
manual_draws = proportions[:, None] * cross
radius = np.quantile(np.abs(manual_draws - cross/88).max(axis=1), .95)
check('six-family simultaneous radius matches direct one-item bootstrap',
      abs(result['comparison']['family']['simultaneous_radius95']['conditional_item'] - radius) < 1e-15)
check('percentile intervals match direct draw counts', np.allclose(
    result['comparison']['tasks']['multihop']['intervals']['conditional_item']['pointwise95'],
    np.quantile(manual_draws, [.025, .975], axis=0).T, rtol=0, atol=1e-15))
block = result['readouts']['huginn']['seed_average']['multihop']['jlens']['physical_block_means']['own_layer']
check('all four physical blocks retain eight-pass means', np.allclose(block, [0, 0, 0, .5/88/8], rtol=0, atol=1e-15))
native = result['native_agreement']
check('native comparisons stay within model vocabulary and trajectory',
      native['ouro']['multihop']['raw']['to_final_native'].tolist() == [1.]*192
      and native['ouro']['multihop']['jlens']['to_final_native'].sum() == 1
      and native['huginn'][str(a.SEEDS[0])]['multihop']['raw']['to_same_source_coda'][0] == 1
      and native['huginn'][str(a.SEEDS[0])]['multihop']['raw']['to_final_native'].sum() == 0)
examples = a.selected_examples(plan['rows'], slots, saved)
check('pre-outcome ten-item random sample', examples['random']['selected_indices'].tolist() == [13, 36, 90, 48, 133, 24, 110, 71, 111, 0])
check('readout failures require misses in both seeds', examples['readout_failures']['tasks']['multihop']['eligible_readout_failure_count'] == 87
      and 0 not in examples['readout_failures']['tasks']['multihop']['eligible_readout_failure_indices']
      and examples['readout_failures']['tasks']['order-ops']['eligible_readout_failure_count'] == 51)
# Positive recovery in every eligible item must retain explicitly empty pools.
all_hit = dict(saved)
for seed in a.SEEDS:
    all_hit[f'huginn_{seed}_jlens_own_metrics'] = np.ones((148, 3))
empty = a.selected_examples(plan['rows'], slots, all_hit)
check('empty failure pools remain explicitly empty', all(
    item['eligible_readout_failure_count'] == 0 and len(item['selected_indices']) == 0 and item['examples'] == []
    for item in empty['readout_failures']['tasks'].values()))
tables = a.expected_tables(result)
check('complete ordered CSV geometry and endpoint labels', {key: len(value) for key, value in tables.items()} ==
      {'cells.csv': 2304, 'metrics.csv': 6, 'native_agreement.csv': 1152}
      and tables['cells.csv'][191]['ouro_identity_reference'] is True
      and tables['cells.csv'][1152]['model'] == 'huginn')
errors = {'maximum_absolute_difference': 0.0}
bad = a.plain(result['comparison']); bad['family']['contrasts'] = 12
rejects('twelve-seed family rejected by output comparison', lambda: a.compare_tree(result['comparison'], bad, 'comparison', errors))
bad = a.plain(result['comparison']); bad['tasks']['multihop']['mean'][0] = float('nan')
rejects('nonfinite report values rejected', lambda: a.compare_tree(result['comparison'], bad, 'comparison', errors))

root = Path(tempfile.mkdtemp(prefix='huginn-audit-proof-'))
receipt = {'schema': 'independent_huginn_auditor_synthetic_proof.v1', 'status': 'passed', 'checks': checks,
           'helper': a.io._record(Path(a.__file__)), 'proof_source': a.io._record(Path(__file__)),
           'frozen_eligibility': a.io._record(a.ROUND / 'deployment/huginn_eligibility.json'),
           'scope': 'Synthetic rank/top1 arrays and frozen pre-outcome population metadata only; no actual or historical scores, no network/GPU/inference; no frozen scoring/aggregation/bootstrap/example functions invoked.'}
(root / 'PROOF.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'checks': len(checks), 'proof': str(root / 'PROOF.json'), 'helper': receipt['helper']}))
