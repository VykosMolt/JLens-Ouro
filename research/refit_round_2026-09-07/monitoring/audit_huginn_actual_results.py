"""Independent CPU reconstruction of the frozen Huginn/Ouro comparison.

Prepare before outcomes; run only on a completed analysis explicitly authorized
by root. Scientific inputs are read-only. The only output is a new /tmp audit
JSON. No network, fitting, inference, model loading or tensor deserialization.
Only the frozen modules' provenance/seal validators are reused; their scoring,
aggregation, bootstrap, example and load_inputs functions are never invoked.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

import numpy as np

sys.dont_write_bytecode = True
ROUND = Path('/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07')
sys.path.insert(0, str(ROUND / 'analysis'))
import analyze_huginn as sealed

io = sealed.io
TASKS = ('multihop', 'order-ops')
SEEDS = (2026090803, 2026090804)
DRAWS, BOOT_SEED, EXAMPLE_SEED, FAILURE_SEED = 10000, 2026090808, 2026090809, 2026090810
METRICS = ('all_learned_mean', 'seven_mean', 'any_of_seven')
MODES = ('conditional_item', 'conditional_component')
FIELDS = tuple(f'{term}_{axis}' for axis in ('layer', 'regions') for term in ('own', 'control', 'delta'))
TERMS = ('own', 'control', 'delta')
METHODS = {'ouro': ('raw', 'jlens'), 'huginn': ('raw', 'jlens', 'coda')}
PAIRS = {'jlens_minus_raw': ('jlens', 'raw'), 'coda_minus_raw': ('coda', 'raw'),
         'jlens_minus_coda': ('jlens', 'coda')}
SUPPORT = {'ouro': list(range(192)), 'huginn': list(range(32))}
LEARNED = {'ouro': list(range(191)), 'huginn': list(range(32))}
GRID = {'ouro': [23, 47, 71, 95, 119, 143, 167], 'huginn': [3, 7, 11, 15, 19, 23, 27]}
EXITS = {'ouro': [47, 95, 143, 191], 'huginn': [3, 7, 11, 15, 19, 23, 27, 31]}
VOCAB = {'ouro': 49152, 'huginn': 65536}
COUNTS = {'items': 148, 'tasks': {'multihop': 93, 'order-ops': 55},
          'eligible_items': {'multihop': 88, 'order-ops': 51},
          'eligible_slots': {'multihop': 98, 'order-ops': 51},
          'control_names': {'multihop': 68, 'order-ops': 20}}
ELIGIBILITY_SHA256 = '153c340591c4c1ff298383a24f2ac1135a157fba08dc9ecb5a82dedc298457af'
OPERATIONS = {'addition', 'subtraction', 'multiplication', 'division', 'mod', 'squared'}
NUMBER_WORDS = {word: str(i) for i, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
NUMBER_WORDS['third'] = '3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def same(left, right, message):
    require(io._canonical(plain(left)) == io._canonical(plain(right)), message)


def close(left, right, message, errors):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, message + ': shape')
    if np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_):
        require(np.array_equal(left, right), message + ': exact values')
        return
    require(np.isfinite(left).all() and np.isfinite(right).all(), message + ': finite values')
    error = float(np.max(np.abs(left - right), initial=0.0))
    errors['maximum_absolute_difference'] = max(errors['maximum_absolute_difference'], error)
    require(error <= 1e-12, message + f': absolute difference {error}')


def compare_tree(expected, actual, label, errors):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), label + ': fields')
        for key, value in expected.items():
            compare_tree(value, actual[key], label + '/' + str(key), errors)
    elif isinstance(expected, np.ndarray) or isinstance(expected, (float, np.floating)):
        close(expected, actual, label, errors)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(expected) == len(actual), label + ': length')
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_tree(left, right, label + '/' + str(index), errors)
    else:
        same(expected, actual, label)


def track(path, records, expected=None):
    path = io._no_links(path)
    record = io._record(path) if expected is None else expected
    io._verify_record(path, record)
    require(str(path) not in records or records[str(path)] == record, 'conflicting record: ' + str(path))
    records[str(path)] = record
    return path


def npz(path):
    require(Path(path).suffix == '.npz', 'only NPZ may enter numerical reader')
    with np.load(io._no_links(path), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'duplicate NPZ members')
        return {key: archive[key].copy() for key in archive.files}


def common_population(plan, original, names, forms):
    """Rebuild joint eligibility from both recorded contexts, keeping aliases."""
    same(plan['task_names'], names, 'original catalogue order')
    same(plan['token_forms']['ouro'], forms, 'Ouro token forms')
    require(len(plan['rows']) == len(original) == 148, 'ordered population size')
    common = {task: [j for j, name in enumerate(names[task])
                     if all(plan['token_forms'][model][task].get(name) for model in METHODS)] for task in TASKS}
    same(common, plan['common_name_indices'], 'joint supported name indices')
    same({task: [name for j, name in enumerate(names[task]) if j not in common[task]] for task in TASKS},
         {'multihop': ['14', 'Vatican'], 'order-ops': []}, 'excluded control names')
    slots = []
    for i, (row, old) in enumerate(zip(plan['rows'], original)):
        for key in ('name', 'task', 'prompt', 'target', 'intermediates', 'own_index'):
            same(row[key], old[key], f'item {i}: ordered {key}')
        for key, value in row['ouro'].items():
            same(value, old[key], f'item {i}: Ouro recorded {key}')
        require(len(row['own_index']) == 3 and len(row['intermediates']) <= 3, 'own slot geometry')
        own = {index for index in row['own_index'] if index >= 0}
        eligibility, controls, reasons = [], [], []
        for s, label in enumerate(row['intermediates']):
            why = []
            for model in METHODS:
                view = row[model]
                tokens = view['intermediate_tokens'][label]
                require(view['n_tokens'] == len(view['token_ids']) and 1 <= view['n_tokens'] <= 512,
                        f'item {i}: recorded context length')
                require(all(type(token) is int and 0 <= token < VOCAB[model] for token in view['token_ids']),
                        f'item {i}: context token vocabulary')
                same(sorted(tokens), sorted(plan['token_forms'][model][row['task']].get(label, [])),
                     f'item {i}: exact token aliases')
                scorable, leaked = bool(tokens), bool(set(tokens) & set(view['token_ids']))
                require(view['scorable'][s] == scorable and view['leaked'][s] == leaked, 'scorable/leakage flags')
                native_eligible = scorable and not leaked and (row['task'] == 'multihop' or label not in OPERATIONS)
                require(view['eligible'][s] == native_eligible, 'per-tokenizer eligibility')
                if not scorable:
                    why.append(model + ':no_single_token_form')
                if leaked:
                    why.append(model + ':form_in_readout_context')
            if row['task'] == 'order-ops' and label in OPERATIONS:
                why.append('arithmetic:operation_slot_excluded')
            pool = [j for j in common[row['task']] if j not in own
                    and ((names[row['task']][j] in OPERATIONS) == (label in OPERATIONS))]
            if not why:
                require(pool and row['own_index'][s] in common[row['task']]
                        and names[row['task']][row['own_index'][s]] == label, 'eligible own/control membership')
            eligibility.append(not why); controls.append(pool); reasons.append(why)
        same(row['eligible'], eligibility, 'joint eligibility')
        same(row['control_indices'], controls, 'exact historical-alias control pools')
        same(row['exclusion_reasons'], reasons, 'joint exclusion reasons')
        slots.append([s for s, yes in enumerate(eligibility) if yes])
    counts = {'items': len(slots), 'tasks': {}, 'eligible_items': {}, 'eligible_slots': {}, 'control_names': {}}
    for task in TASKS:
        indices = [i for i, row in enumerate(plan['rows']) if row['task'] == task]
        counts['tasks'][task] = len(indices)
        counts['eligible_items'][task] = sum(bool(slots[i]) for i in indices)
        counts['eligible_slots'][task] = sum(len(slots[i]) for i in indices)
        counts['control_names'][task] = len(common[task])
    same(counts, COUNTS, 'frozen joint denominators'); same(plan['counts'], counts, 'plan denominators')
    return slots


def check_arrays(arrays, model, plan):
    columns, limit = len(SUPPORT[model]), VOCAB[model]
    ranks, own, top = (arrays[key] for key in ('allrank', 'rank', 'top1'))
    require(ranks.shape == (148, 128, columns) and ranks.dtype == np.int32, model + ': allrank geometry')
    require(own.shape == (148, 3, columns) and own.dtype == np.int32, model + ': own-rank geometry')
    require(top.shape == (148, columns) and top.dtype == np.int64, model + ': top1 geometry')
    require(np.all((top >= 0) & (top < limit)), model + ': top1 vocabulary bounds')
    for i, row in enumerate(plan['rows']):
        supported = np.zeros(128, bool)
        indices = plan['common_name_indices'][row['task']] if model == 'huginn' else range(len(plan['task_names'][row['task']]))
        supported[list(indices)] = True
        require(np.all((ranks[i, supported] >= 0) & (ranks[i, supported] < limit))
                and np.all(ranks[i, ~supported] == -1), model + ': original name-axis support/sentinels')
        for slot, name_index in enumerate(row['own_index']):
            expected = ranks[i, name_index] if name_index >= 0 else np.full(columns, -1, np.int32)
            require(np.array_equal(own[i, slot], expected), model + ': own rank alias/padding')
        alias_rows = {}
        for name_index in indices:
            name = plan['task_names'][row['task']][name_index]
            token_forms = tuple(sorted(plan['token_forms'][model][row['task']][name]))
            if token_forms in alias_rows:
                require(np.array_equal(ranks[i, name_index], ranks[i, alias_rows[token_forms]]),
                        model + ': identical token-form aliases have different ranks')
            else:
                alias_rows[token_forms] = name_index


def reduce_ranks(ranks, model, rows, slots):
    """Score every control's event before controls, slots, items or seeds average."""
    own, control = np.zeros((len(rows), len(SUPPORT[model]))), np.zeros((len(rows), len(SUPPORT[model])))
    own_regions, control_regions = np.zeros((len(rows), 3)), np.zeros((len(rows), 3))
    selections = (SUPPORT[model], LEARNED[model], GRID[model])
    for i, row in enumerate(rows):
        if not slots[i]:
            continue
        a, b, ar, br = [], [], [], []
        for slot in slots[i]:
            own_ranks, other_ranks = ranks[i, row['own_index'][slot]], ranks[i, row['control_indices'][slot]]
            require(np.all(own_ranks >= 0) and np.all(other_ranks >= 0), 'negative rank entered eligible scoring')
            yes, other = own_ranks < 10, other_ranks < 10
            a.append(yes.astype(float)); b.append(other.mean(axis=0))
            ar.append([float(yes[selection].any()) for selection in selections])
            br.append([np.any(other[:, selection], axis=1).mean() for selection in selections])
        own[i], control[i] = np.mean(a, axis=0), np.mean(b, axis=0)
        own_regions[i], control_regions[i] = np.mean(ar, axis=0), np.mean(br, axis=0)
    return {'eligible': np.asarray([bool(slot) for slot in slots]),
            'own_layer': own, 'control_layer': control, 'delta_layer': own - control,
            'own_regions': own_regions, 'control_regions': control_regions, 'delta_regions': own_regions - control_regions}


def scalars(bank, model, term):
    layer = bank[term + '_layer']
    return np.column_stack((layer[:, LEARNED[model]].mean(axis=1), layer[:, GRID[model]].mean(axis=1),
                            bank[term + '_regions'][:, 2]))


def components(rows, indices, slots):
    # Independent connectivity closure also checks the frozen deterministic
    # component numbering used by the exact bootstrap stream.
    labels = []
    for index in indices:
        group = set()
        for s in slots[index]:
            name = rows[index]['intermediates'][s].lower()
            group.add(NUMBER_WORDS.get(name, str(int(name)) if name.isdigit() else name))
        labels.append(group)
    edges = np.asarray([[bool(a & b) for b in labels] for a in labels])
    for k in range(len(indices)):
        edges |= edges[:, k, None] & edges[None, k, :]
    # Root attachment follows first-seen labels in the frozen ordered slots.
    parent, seen = list(range(len(indices))), {}
    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index
    for local, index in enumerate(indices):
        for slot in slots[index]:
            name = rows[index]['intermediates'][slot].lower()
            label = NUMBER_WORDS.get(name, str(int(name)) if name.isdigit() else name)
            if label in seen:
                parent[root(local)] = root(seen[label])
            else:
                seen[label] = local
    roots = [root(i) for i in range(len(indices))]
    mapping = {value: j for j, value in enumerate(sorted(set(roots)))}
    groups = np.asarray([mapping[value] for value in roots], np.int64)
    require(np.array_equal(edges, groups[:, None] == groups[None, :]), 'component partition and numbering differ')
    return groups


def draw_weights(chosen, groups=None):
    size = chosen.shape[1]
    counts = np.zeros((chosen.shape[0], size))
    for k in range(size):
        counts[:, k] = (chosen == k).sum(axis=1)
    expanded = counts if groups is None else counts[:, groups]
    return expanded / expanded.sum(axis=1, keepdims=True)


def bootstrap(rows, slots):
    designs, populations, saved = {}, {}, {}
    for ti, task in enumerate(TASKS):
        indices = np.asarray([i for i, row in enumerate(rows) if row['task'] == task and slots[i]], np.int64)
        groups = components(rows, indices, slots)
        n, g = len(indices), int(groups.max()) + 1
        items = np.random.default_rng(np.random.SeedSequence(BOOT_SEED, spawn_key=(1, ti))).integers(0, n, (DRAWS, n))
        clusters = np.random.default_rng(np.random.SeedSequence(BOOT_SEED, spawn_key=(2, ti))).integers(0, g, (DRAWS, g))
        designs[task] = {'indices': indices, 'conditional_item': draw_weights(items),
                         'conditional_component': draw_weights(clusters, groups)}
        populations[task] = {'item_indices': indices, 'eligible_items': n,
            'eligible_slots': sum(len(slots[i]) for i in indices), 'component_membership': groups,
            'component_sizes': np.bincount(groups), 'component_count': g}
        saved.update({f'bootstrap_{task}_indices': indices, f'bootstrap_{task}_groups': groups,
                      f'bootstrap_{task}_item_indices': items, f'bootstrap_{task}_component_indices': clusters})
    return designs, populations, saved


def reconstruct(arrays, plan, slots):
    rows = plan['rows']
    design, populations, expected = bootstrap(rows, slots)
    banks = {}
    for model, methods in METHODS.items():
        for seed in ((None,) if model == 'ouro' else SEEDS):
            prefix = model if seed is None else f'{model}_{seed}'
            banks[prefix] = {}
            for method in methods:
                data = arrays['ouro'][method] if seed is None else arrays['huginn'][seed][method]
                bank = reduce_ranks(data['allrank'], model, rows, slots)
                banks[prefix][method] = bank
                expected.update({f'{prefix}_{method}_{field}': value for field, value in bank.items()})
                for term in TERMS:
                    expected[f'{prefix}_{method}_{term}_metrics'] = scalars(bank, model, term)
            for pair, (left, right) in PAIRS.items():
                if left not in methods or right not in methods:
                    continue
                expected.update({f'{prefix}_{pair}_{field}': banks[prefix][left][field] - banks[prefix][right][field]
                                 for field in FIELDS})
    paired_o = expected['ouro_jlens_delta_metrics'] - expected['ouro_raw_delta_metrics']
    paired_h = np.stack([expected[f'huginn_{seed}_jlens_delta_metrics'] - expected[f'huginn_{seed}_raw_delta_metrics'] for seed in SEEDS])
    cross = paired_h - paired_o[None]
    average = cross.mean(axis=0)
    seven = np.stack([expected[f'huginn_{seed}_jlens_minus_raw_delta_layer'][:, GRID['huginn']]
                      - expected['ouro_jlens_minus_raw_delta_layer'][:, GRID['ouro']] for seed in SEEDS])
    expected.update(ouro_paired_metrics=paired_o, huginn_paired_metrics=paired_h, cross_model_paired_metrics=cross,
                    cross_model_seed_average_metrics=average, cross_model_paired_seven_cells=seven)
    points = {task: average[design[task]['indices']].mean(axis=0) for task in TASKS}
    draws = {task: {mode: design[task][mode] @ average[design[task]['indices']] for mode in MODES} for task in TASKS}
    radii = {mode: float(np.quantile(np.abs(np.concatenate([draws[task][mode] - points[task] for task in TASKS], axis=1)).max(axis=1), .95)) for mode in MODES}
    comparison = {'metric_ids': list(METRICS), 'conditional_on_fit_id': 1, 'calibration_fits_per_model': 1,
                  'evaluation_base_seeds': list(SEEDS), 'family': {'contrasts': 6, 'simultaneous_radius95': radii}, 'tasks': {}}
    for task in TASKS:
        indices = design[task]['indices']
        intervals = {mode: {'pointwise95': np.quantile(draws[task][mode], [.025, .975], axis=0).T,
                            'simultaneous95': np.column_stack((points[task] - radii[mode], points[task] + radii[mode]))} for mode in MODES}
        comparison['tasks'][task] = {'mean': points[task], 'per_seed': cross[:, indices].mean(axis=1),
            'ouro_paired_effect': paired_o[indices].mean(axis=0), 'huginn_paired_effect_per_seed': paired_h[:, indices].mean(axis=1),
            'seven_cell_difference_per_seed': seven[:, indices].mean(axis=1), 'intervals': intervals}
    return expected, {'populations': populations, 'comparison': comparison,
                      'readouts': curves(banks, design), 'native_agreement': agreements(arrays, design)}


def curves(banks, design):
    result = {}
    for model, methods in METHODS.items():
        result[model] = {}
        for seed in (('fixed',) if model == 'ouro' else (*SEEDS, 'seed_average')):
            if seed == 'seed_average':
                source = {method: {field: np.mean([banks[f'huginn_{s}'][method][field] for s in SEEDS], axis=0)
                                   for field in FIELDS} for method in methods}
            else:
                source = banks['ouro' if model == 'ouro' else f'huginn_{seed}']
            views = {method: {field: source[method][field] for field in FIELDS} for method in methods}
            views.update({pair: {field: source[left][field] - source[right][field] for field in FIELDS}
                          for pair, (left, right) in PAIRS.items() if left in methods and right in methods})
            result[model][str(seed)] = {}
            for task in TASKS:
                task_bank = result[model][str(seed)][task] = {}
                for method, bank in views.items():
                    values = {field: bank[field][design[task]['indices']].mean(axis=0) for field in FIELDS}
                    values.update(source_indices=SUPPORT[model], native_exit_indices=EXITS[model],
                                  known_identity_indices=[191] if model == 'ouro' and method.startswith('jlens') else [])
                    if model == 'huginn':
                        values['physical_block_means'] = {field: np.asarray([values[field][block::4].mean() for block in range(4)])
                                                          for field in ('own_layer', 'control_layer', 'delta_layer')}
                    learned = values['delta_layer'][LEARNED[model]]
                    values['learned_cell_empirical_cdf'] = {'sorted_cell_means': np.sort(learned),
                        'cdf': np.arange(1, len(learned) + 1) / len(learned),
                        'unit': 'correlated source locations, not independent observations'}
                    task_bank[method] = values
    return result


def agreements(arrays, design):
    result = {'ouro': {}, 'huginn': {}}
    for task in TASKS:
        indices = design[task]['indices']
        result['ouro'][task] = {method: {'to_final_native': np.mean(
            arrays['ouro'][method]['top1'][indices] == arrays['ouro']['exit_top1'][indices, 3, None], axis=0)} for method in METHODS['ouro']}
    for seed in SEEDS:
        result['huginn'][str(seed)] = {}
        bank = arrays['huginn'][seed]
        for task in TASKS:
            indices = design[task]['indices']
            result['huginn'][str(seed)][task] = {method: {
                'to_final_native': np.mean(bank[method]['top1'][indices] == bank['native_top1'][indices, None], axis=0),
                'to_same_source_coda': np.mean(bank[method]['top1'][indices] == bank['coda']['top1'][indices], axis=0)} for method in METHODS['huginn']}
    return result


def selected_examples(rows, slots, values):
    eligible = np.asarray([i for i, selected in enumerate(slots) if selected], np.int64)
    selected = np.random.default_rng(EXAMPLE_SEED).choice(eligible, 10, replace=False)
    same(selected, [13, 36, 90, 48, 133, 24, 110, 71, 111, 0], 'frozen pre-outcome random indices')
    def record(index):
        index = int(index)
        readouts = {'ouro': {}, 'huginn': {}}
        for model, methods in METHODS.items():
            for seed in ((None,) if model == 'ouro' else SEEDS):
                prefix = model if seed is None else f'{model}_{seed}'
                target = readouts[model] if seed is None else readouts[model].setdefault(str(seed), {})
                for method in methods:
                    target[method] = {term: values[f'{prefix}_{method}_{term}_metrics'][index] for term in TERMS}
        return {'index': index, **{key: rows[index][key] for key in ('name', 'task', 'prompt', 'target', 'intermediates', 'eligible', 'exclusion_reasons')},
                'readout_metrics': readouts, 'cross_model_paired_metrics_per_seed': values['cross_model_paired_metrics'][:, index]}
    failures = {}
    # A miss requires zero own recovery in BOTH trajectories, before averaging.
    miss = np.logical_and.reduce([values[f'huginn_{seed}_jlens_own_metrics'][:, 2] == 0 for seed in SEEDS])
    for ti, task in enumerate(TASKS):
        pool = np.asarray([i for i in eligible if rows[i]['task'] == task and miss[i]], np.int64)
        chosen = np.random.default_rng(np.random.SeedSequence(FAILURE_SEED, spawn_key=(ti,))).choice(pool, min(5, len(pool)), replace=False)
        failures[task] = {'eligible_readout_failure_count': len(pool), 'eligible_readout_failure_indices': pool,
                          'selected_indices': chosen, 'examples': [record(i) for i in chosen]}
    return {'metric_ids': METRICS, 'method_field_delta': 'own recovery minus matched-control recovery',
        'interpretation': 'descriptive item examples; selection never changes metrics, cells, seeds or denominators',
        'random': {'seed': EXAMPLE_SEED, 'eligible_population_size': len(eligible), 'selected_indices': selected,
                   'selection': 'ten jointly eligible items uniformly without replacement, independent of outcomes',
                   'examples': [record(i) for i in selected]},
        'readout_failures': {'seed': FAILURE_SEED,
            'definition': 'seed-average Huginn J-Lens own any-of-seven recovery equals zero; no eligible own label is recovered on the seven cells in either seed',
            'interpretation': 'conditional descriptive concept-readout misses, not whole-answer incorrectness', 'tasks': failures}}


def expected_tables(result):
    cells, native, metrics = [], [], []
    for model, seeds in result['readouts'].items():
        for seed, tasks in seeds.items():
            for task, methods in tasks.items():
                for method, bank in methods.items():
                    for source in SUPPORT[model]:
                        width = 48 if model == 'ouro' else 4
                        cells.append({'model': model, 'seed': seed, 'task': task, 'method_or_contrast': method,
                            'source_index': source, 'pass_one_based': source // width + 1, 'physical_block_one_based': source % width + 1,
                            'native_pass_end': source in EXITS[model], 'ouro_identity_reference': model == 'ouro' and source == 191,
                            'within_pass_coda_intervention': model == 'huginn' and source not in EXITS[model] and 'coda' in method,
                            'own': float(bank['own_layer'][source]), 'control': float(bank['control_layer'][source]), 'excess': float(bank['delta_layer'][source])})
    for model in METHODS:
        for seed in (('fixed',) if model == 'ouro' else tuple(map(str, SEEDS))):
            task_bank = result['native_agreement'][model] if model == 'ouro' else result['native_agreement'][model][seed]
            for task in TASKS:
                for method in METHODS[model]:
                    bank = task_bank[task][method]
                    for source in SUPPORT[model]:
                        native.append({'model': model, 'seed': seed, 'task': task, 'method': method, 'source_index': source,
                            'native_pass_end': source in EXITS[model], 'to_final_native': float(bank['to_final_native'][source]),
                            'to_same_source_coda': '' if model == 'ouro' else float(bank['to_same_source_coda'][source])})
    for task in TASKS:
        bank = result['comparison']['tasks'][task]
        for j, metric in enumerate(METRICS):
            row = {'task': task, 'metric': metric, 'mean_over_two_fixed_seeds': float(bank['mean'][j]),
                   'seed_2026090803': float(bank['per_seed'][0, j]), 'seed_2026090804': float(bank['per_seed'][1, j]),
                   'ouro_fit01_paired_effect': float(bank['ouro_paired_effect'][j])}
            for mode in MODES:
                for field, label in (('pointwise95', 'pointwise'), ('simultaneous95', 'simultaneous')):
                    for k, side in enumerate(('low', 'high')):
                        row[f'{mode}_{label}_{side}'] = float(bank['intervals'][mode][field][j, k])
            metrics.append(row)
    require((len(cells), len(metrics), len(native)) == (2304, 6, 1152), 'complete table geometry')
    return {'cells.csv': cells, 'metrics.csv': metrics, 'native_agreement.csv': native}


def check_tables(root, expected, errors):
    for name, expected_rows in expected.items():
        with (root / name).open(newline='') as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames == list(expected_rows[0]), name + ': exact header order')
            actual = list(reader)
        require(len(actual) == len(expected_rows), name + ': row count')
        for i, (left, right) in enumerate(zip(expected_rows, actual)):
            for field, value in left.items():
                label = f'{name}/{i}/{field}'
                if type(value) is float:
                    close(value, float(right[field]), label, errors)
                else:
                    same(str(value), right[field], label)


def audit(args):
    records, errors = {}, {'maximum_absolute_difference': 0.0}
    out = io._no_links(args.output)
    require(out.is_absolute() and Path('/tmp') in out.parents and not out.exists(), 'output must be a new absolute /tmp path')
    roots = {name: io._no_links(getattr(args, name)) for name in ('main', 'controls', 'huginn', 'analysis', 'accepted_p2')}
    require(all(root != out and root not in out.parents for root in roots.values()), 'audit output lies in input tree')
    track(Path(__file__), records)
    for name in ('run_spec', 'combined_contract', 'huginn_manifest', 'huginn_calibration', 'eligibility'):
        track(getattr(args, name), records)
    require(records[str(io._no_links(args.eligibility))]['sha256'] == ELIGIBILITY_SHA256, 'frozen eligibility bytes')
    validation = sealed.validate_output(roots['analysis'])
    accepted_validation = sealed.ouro.validate_output(roots['accepted_p2'])
    for label in ('analysis', 'accepted_p2'):
        complete = io._json(track(roots[label] / 'COMPLETE.json', records))
        for name, record in complete['files'].items():
            track(roots[label] / name, records, record)
    owner = io._json(roots['analysis'] / 'OWNER.json')['identity']
    accepted = io._json(roots['accepted_p2'] / 'OWNER.json')['identity']
    require(accepted['kind'] == 'five_fit_ouro_and_conditional_controls', 'accepted input must include completed P1 and P2')
    for key, value in {'bootstrap_draws': DRAWS, 'bootstrap_seed': BOOT_SEED, 'example_seed': EXAMPLE_SEED,
                       'readout_failure_seed': FAILURE_SEED, 'evaluation_base_seeds': list(SEEDS),
                       'fit_ids': {'ouro': 1, 'huginn': 1}}.items():
        same(owner[key], value, 'analysis identity ' + key)
    for name, record in owner['source_records'].items():
        track(Path(name), records, record)
    main = io._contract(args.run_spec, args.ouro_src, [1, 2, 3, 4, 5])
    sealed.evaluation._evaluation_sources(main)
    hmain, combined, manifest, calibration = sealed.huginn._contract(args)
    require(owner['run_spec_sha256'] == accepted['run_spec_sha256'] == main['spec_sha256'] == hmain['spec_sha256'], 'run specification bindings')
    expected_sources = {str(path): {'bytes': path.stat().st_size, 'sha256': next(row['sha256'] for row in main['sources']
        if (row['root'], row['path']) == key)} for key, path in main['source_paths'].items()}
    source_arguments = [getattr(args, name) for name in ('run_spec', 'combined_contract', 'huginn_manifest', 'huginn_calibration', 'eligibility')]
    authored = [ROUND / 'analysis' / name for name in ('analyze_huginn.py', 'HUGINN_COMPARISON_CONTRACT.md', 'analyze_refits.py', 'CONTRACT.md')]
    for path in [*source_arguments, *authored]:
        path = io._no_links(path)
        record = io._record(path)
        require(str(path) not in expected_sources or expected_sources[str(path)] == record, 'source/argument digest collision')
        expected_sources[str(path)] = record
    same(owner['source_records'], expected_sources, 'complete frozen source and argument record membership')
    checked_main = sealed.controls.validate_completed_main(roots['main'], main, require_all=True)
    checked_controls = sealed.controls.validate_completed_controls(roots['controls'], main, combined, checked_main)
    # Generic seals only: no evaluate_huginn.validate_output (it invokes a scorer
    # and deserializes tensors). Rank/top1/initialization semantics below are ours.
    checked_huginn = sealed.evaluation.validate_output(roots['huginn'], schema='huginn_r8_evaluation.v1')
    artifacts = {}
    for label, checked in (('main', checked_main), ('controls', checked_controls), ('huginn', checked_huginn)):
        artifact = {'root': str(roots[label]), 'files': sealed.ouro.artifact_records(roots[label], checked)}
        same(owner['evaluation_artifacts'][label], artifact, 'analysis consumed artifact ' + label)
        if label != 'huginn':
            same(accepted['evaluation_artifacts'][label]['files'], artifact['files'], 'accepted P1/P2 bytes ' + label)
        for name, record in artifact['files'].items():
            track(roots[label] / name, records, record)
        artifacts[label] = artifact
    require(set(owner['evaluation_artifacts']) == set(artifacts), 'input family membership')
    plan = io._json(args.eligibility)
    same(io._json(roots['huginn'] / 'population/eligibility.json'), plan, 'stored evaluation population')
    same(io._json(roots['analysis'] / 'population.json'), plan, 'retained analysis population')
    sealed.bind_huginn(checked_huginn, roots['huginn'], args, main, combined, manifest, calibration, plan,
                       sealed._prior_records(artifacts))
    original, names, forms = (io._json(roots['main'] / 'common' / name) for name in ('items.json', 'task_names.json', 'token_forms.json'))
    slots = common_population(plan, original, names, forms)
    arrays = {'ouro': {}, 'huginn': {}}
    for method, section, prefix in (('raw', 'common', 'logitlens_'), ('jlens', 'fits/fit_01', 'jlens_exit3_')):
        archive = npz(roots['main'] / section / 'arrays.npz')
        data = {key: archive[prefix + key] for key in ('allrank', 'rank', 'top1')}
        check_arrays(data, 'ouro', plan)
        arrays['ouro'][method] = data
        if method == 'raw':
            exits = archive['exit_top1']
            require(exits.shape == (148, 4) and np.array_equal(exits, data['top1'][:, EXITS['ouro']]), 'Ouro native-exit pairing')
            arrays['ouro']['exit_top1'] = exits
    for key in ('allrank', 'rank', 'top1'):
        require(np.array_equal(arrays['ouro']['raw'][key][..., 191], arrays['ouro']['jlens'][key][..., 191]), 'Ouro identity endpoint ' + key)
    for seed in SEEDS:
        section = roots['huginn'] / 'seeds' / str(seed)
        archive = npz(section / 'arrays.npz')
        require(set(archive) == {'native_top1'} | {f'{method}_{key}' for method in METHODS['huginn'] for key in ('allrank', 'rank', 'top1')}, 'Huginn exact array fields')
        bank = {'native_top1': archive['native_top1']}
        require(bank['native_top1'].shape == (148,) and bank['native_top1'].dtype == np.int64
                and np.all((bank['native_top1'] >= 0) & (bank['native_top1'] < VOCAB['huginn'])), 'Huginn native token array')
        for method in METHODS['huginn']:
            bank[method] = {key: archive[f'{method}_{key}'] for key in ('allrank', 'rank', 'top1')}
            check_arrays(bank[method], 'huginn', plan)
        require(np.array_equal(bank['coda']['top1'][:, 31], bank['native_top1']), 'paired final coda/native token IDs')
        initializations = io._json(section / 'initializations.json')
        require(len(initializations) == 148, 'initialization item count')
        for i, (row, actual) in enumerate(zip(plan['rows'], initializations)):
            recipe = {'version': 1, 'base_seed': seed, 'namespace': 'evaluation', 'input_ids': row['huginn']['token_ids']}
            state_seed = int.from_bytes(bytes.fromhex(io._digest(recipe))[:8], 'big') % 2 ** 63
            for key, value in {'index': i, 'name': row['name'], 'task': row['task'], 'base_seed': seed,
                               'namespace': 'evaluation', 'prompt_state_seed': state_seed,
                               'input_ids_sha256': io._digest(row['huginn']['token_ids']), 'seed_recipe': plan['seed_rule']}.items():
                same(actual[key], value, 'coupled seed provenance ' + key)
        arrays['huginn'][seed] = bank
    expected, result = reconstruct(arrays, plan, slots)
    saved = npz(roots['analysis'] / 'paired_items.npz')
    require(set(saved) == set(expected), 'complete retained archive membership')
    for key, value in expected.items():
        close(value, saved[key], 'retained array ' + key, errors)
    report = io._json(roots['analysis'] / 'report.json')
    for key, value in {'schema': 'huginn_ouro_comparison.v1', 'counts': COUNTS, 'learned_sources': LEARNED,
        'displayed_sources': SUPPORT, 'seven_sources': GRID, 'region_order': ['all_displayed', 'all_learned', 'seven'],
        'native_exit_indices': EXITS, 'known_identity': {'ouro': [191], 'huginn': []}, 'examples': 'examples.json',
        'prediction_verbatim': 'If J-Lens recovers content outside the native output basis, its advantage over raw logit lens should be larger in Huginn than in Ouro.'}.items():
        same(report[key], value, 'report fixed field ' + key)
    for key, value in {'draws': DRAWS, 'seed': BOOT_SEED, 'modes': list(MODES), 'fit_resampling': False,
                       'seed_resampling': False, 'seed_streams': {'items': [1, 'task_index'], 'components': [2, 'task_index']}}.items():
        same(report['uncertainty'][key], value, 'uncertainty scope ' + key)
    for key, value in result.items():
        compare_tree(value, report[key], 'report/' + key, errors)
    examples = selected_examples(plan['rows'], slots, expected)
    compare_tree(examples, io._json(roots['analysis'] / 'examples.json'), 'examples', errors)
    tables = expected_tables(result)
    check_tables(roots['analysis'], tables, errors)
    for path, record in records.items():
        io._verify_record(Path(path), record)
    receipt = {'schema': 'independent_huginn_actual_audit.v1', 'status': 'passed', 'analysis_validation': validation,
        'accepted_p2_validation': accepted_validation, 'counts': COUNTS, 'archive_fields': len(expected),
        'table_rows': {name: len(rows) for name, rows in tables.items()}, 'numerical_agreement': errors,
        'bootstrap_draws': DRAWS, 'bootstrap_seed': BOOT_SEED, 'simultaneous_family_size': 6,
        'populations': result['populations'], 'comparison': result['comparison'],
        'random_indices': examples['random']['selected_indices'],
        'readout_failure_samples': {task: {key: value for key, value in entry.items() if key != 'examples'}
                                    for task, entry in examples['readout_failures']['tasks'].items()},
        'consumed_records': records, 'limitations': [
            'Independent rank/top1 analysis; semantic owner/seal validation reused from frozen modules.',
            'Tensor cache bytes are hashed but not deserialized; cache tensor/native-logit checks remain with the completed frozen analyzer.',
            'Saved minimum ranks support frozen aliases; full logits and tokenizer retokenization are not reconstructed.',
            'Conditional on one fitted bank per model and two fixed Huginn trajectories; no fit or seed resampling.',
            'No inference, GPU, network, source edits or scientific interpretation receipt.']}
    io._new_json(out, plain(receipt))
    return {'status': 'passed', 'output': str(out), 'identity_sha256': validation['identity_sha256'],
            'archive_fields': len(expected), 'maximum_absolute_difference': errors['maximum_absolute_difference']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'controls', 'huginn', 'analysis', 'accepted-p2'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ouro-src', type=Path, default=Path('/home/moloch/ouro_project/src'))
    for name, filename in (('run-spec', 'run_spec.json'), ('combined-contract', 'combined_contract.json'),
                           ('huginn-manifest', 'huginn_model_manifest.json'), ('huginn-calibration', 'huginn_calibration.json'),
                           ('eligibility', 'huginn_eligibility.json')):
        parser.add_argument('--' + name, type=Path, default=ROUND / 'deployment' / filename)
    parser.add_argument('--output', type=Path, default=Path('/tmp') / ('huginn_actual_audit_' + uuid.uuid4().hex + '.json'))
    print(json.dumps(audit(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
