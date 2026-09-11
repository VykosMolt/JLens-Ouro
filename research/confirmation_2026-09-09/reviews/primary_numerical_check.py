#!/usr/bin/env python3
"""Independent saved-rank reconstruction; imports no producer/reducer/model code.

Usage: --results ACCEPTED_RESULTS --analysis ANALYSIS_DIRECTORY --out NEW_JSON
       --self-test --out NEW_JSON
All comparisons use absolute tolerance 1e-12 and zero relative tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import uuid

import numpy as np

sys.dont_write_bytecode = True
ROUND = Path(__file__).resolve().parents[1]
DRAWS, SEED, TOLERANCE = 20000, 2026090901, 1e-12
BAND = tuple(range(169, 181))
ARMS = ('raw', 'fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')
LENGTHS = {'raw': 192, 'fit01': 192, 'fit02': 192, 'penultimate': 190, 'sampled_sum': 191, 'diagonal': 191}
COMPONENTS = ['primary_excess_difference', 'fit01_intended', 'fit01_control', 'raw_intended',
              'raw_control', 'intended_difference', 'control_difference']
FROZEN = {
    'bundle/frozen/run_spec.json': {'bytes': 18643, 'sha256': '8a4a3e6725d811d0b2584561a277d5071f097c551f40c46be9f6773a7d61181e'},
    'bundle/frozen/population.json': {'bytes': 373077, 'sha256': 'd7b4efa7e650114d40a521a05e0ca64caa10259f62d50aa94ad1fdc5d210c087'},
    'bundle/frozen/benchmark.json': {'bytes': 967118, 'sha256': '735816c5d540f15ed47cf88ba924f04d5e6ad77eb0cd062b20554543011c6e99'},
    'analysis/analyze.py': {'bytes': 5870, 'sha256': 'e4c3812d91d16d588d2beddab34e683e7732fc1d4ea4b1ee1ecb8b5b6b79e47f'},
    'evaluation/measurement.py': {'bytes': 4277, 'sha256': '06c615b29285e0e4e10196ae8f15b015d5c8a362f5e7704adb38ce6e275799fe'},
    'FREEZE.json': {'bytes': 7279, 'sha256': '8148c2ab0bb62e962ab33c74940b5c84cf711f4a1d41e18f79d55d208b5a6ab7'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Linked input/output path: ' + str(part))
    return path


def signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def byte_record(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read_bytes(path, records):
    path = regular(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 256 * 1024**2, 'Input is not a bounded regular file')
    with path.open('rb') as handle:
        require(signature(os.fstat(handle.fileno())) == signature(before), 'Input changed before open')
        raw = handle.read()
        require(signature(os.fstat(handle.fileno())) == signature(before), 'Input changed during read')
    require(signature(path.stat()) == signature(before) and len(raw) == before.st_size, 'Input pathname changed')
    pin = byte_record(raw)
    if str(path) in records:
        require(records[str(path)] == pin, 'Input changed between reads')
    records[str(path)] = pin
    return raw


def read_json(path, records):
    return json.loads(read_bytes(path, records))


def read_npz(path, records, keys):
    with np.load(io.BytesIO(read_bytes(path, records)), allow_pickle=False) as archive:
        require(set(keys) <= set(archive.files), 'Missing saved arrays: ' + str(path))
        return {key: archive[key].copy() for key in keys}


def recheck_inputs(records):
    for path, expected in list(records.items()):
        require(byte_record(read_bytes(path, {})) == expected, 'Input changed before verification publication: ' + path)


def publish(path, document):
    path = regular(path)
    require(path.parent.is_dir() and not path.exists(), 'Output must be a fresh JSON path in an existing directory')
    raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as writer:
            writer.write(raw); writer.flush(); os.fsync(writer.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def array_record(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype), **byte_record(value.tobytes(order='C'))}


class Comparisons:
    def __init__(self):
        self.rows = []

    def exact(self, label, actual, expected):
        okay = actual == expected
        self.rows.append({'check': label, 'status': 'passed' if okay else 'failed', 'comparison': 'exact'})
        require(okay, 'Exact comparison failed: ' + label)

    def numeric(self, label, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected, dtype=np.float64)
        require(actual.shape == expected.shape and actual.dtype.kind in 'fiu'
                and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid numerical comparison: ' + label)
        difference = float(np.max(np.abs(actual.astype(np.float64) - expected), initial=0))
        okay = difference <= TOLERANCE
        self.rows.append({'check': label, 'status': 'passed' if okay else 'failed',
                          'shape': list(expected.shape), 'values': int(expected.size),
                          'max_absolute_difference': difference, 'absolute_tolerance': TOLERANCE,
                          'relative_tolerance': 0})
        require(okay, 'Numerical comparison failed: ' + label + '; max difference ' + str(difference))


def population_weights(population, *, frozen=False):
    """Name-axis weights are constructed without any layer/scoring reducer."""
    require(population.get('schema') == 'confirmation_population.v1', 'Wrong population schema')
    require(set(population['names']) == {'multihop'}, 'Unexpected task catalogue')
    names, rows = population['names']['multihop'], population['rows']
    require(names and len(names) <= 128 and len(set(names)) == len(names)
            and all(isinstance(name, str) and name for name in names), 'Invalid or duplicated canonical names')
    require(rows and len({row['name'] for row in rows}) == len(rows), 'Invalid or duplicated item names')
    forms = population['token_forms']['multihop']
    require(set(forms) == set(names) and not population['cross_concept_token_collisions'], 'Alias catalogue differs')
    token_owner = {}
    for name in names:
        require(isinstance(forms[name], list) and forms[name] and len(set(forms[name])) == len(forms[name]), 'Invalid token aliases')
        for token in forms[name]:
            require(type(token) is int and 0 <= token < 49152 and token not in token_owner, 'Cross-concept alias contamination')
            token_owner[token] = name
    own, control = np.zeros((len(rows), len(names))), np.zeros((len(rows), len(names)))
    eligible = np.zeros(len(rows), dtype=bool)
    control_counts, eligible_counts = [], []
    for index, row in enumerate(rows):
        require(row['task'] == 'multihop' and isinstance(row['dependency_group_id'], str)
                and row['dependency_group_id'], 'Invalid item task/dependency group')
        labels = row['intermediates']
        require(0 < len(labels) <= 3 and len(set(labels)) == len(labels)
                and all(label in names for label in labels), 'Invalid intended labels')
        intended = [names.index(label) for label in labels]
        require(row['own_index'] == intended + [-1] * (3-len(labels)), 'Own-name axis/padding changed')
        require(len(row['eligible']) == len(row['scorable']) == len(row['leaked']) == len(labels)
                and len(row['control_indices']) == len(labels), 'Eligibility geometry changed')
        selected = []
        for slot, label in enumerate(labels):
            flags = (row['eligible'][slot], row['scorable'][slot], row['leaked'][slot])
            require(all(type(flag) is bool for flag in flags), 'Eligibility is not boolean')
            require(sorted(row['intermediate_tokens'][label]) == sorted(forms[label]), 'Intended token alias set changed')
            leaked = bool(set(forms[label]) & set(row['token_ids']))
            require(row['leaked'][slot] == leaked and row['scorable'][slot] == bool(forms[label])
                    and row['eligible'][slot] == (row['scorable'][slot] and not leaked), 'Eligibility/leakage does not reconstruct')
            if row['eligible'][slot]:
                selected.append(slot)
        eligible_counts.append(len(selected)); eligible[index] = bool(selected)
        expected_controls = [j for j in range(len(names)) if j not in intended]
        require(expected_controls, 'No non-own controls')
        for slot in selected:
            require(row['control_indices'][slot] == expected_controls, 'Control names include own labels or changed weights')
            control_counts.append(len(expected_controls))
            own[index, intended[slot]] += 1 / len(selected)
            control[index, expected_controls] += 1 / (len(selected) * len(expected_controls))
    if frozen:
        require(len(rows) == 160 and len(names) == 80 and eligible.all()
                and eligible_counts == [1] * 160 and control_counts == [79] * 160,
                'Frozen 160-item / 80-name / 79-control eligibility differs')
    return {'own': own, 'control': control, 'eligible': eligible, 'control_counts': control_counts,
            'eligible_label_counts': eligible_counts, 'names': names}


def rank_hits(arrays, population, columns):
    rows, names = population['rows'], population['names']['multihop']
    ranks, top = arrays['allrank'], arrays['top10_ids']
    require(ranks.dtype == np.int32 and ranks.shape == (len(rows), 128, columns), 'Saved allrank geometry/dtype differs')
    require(np.all((ranks[:, :len(names)] >= 0) & (ranks[:, :len(names)] < 49152))
            and np.all(ranks[:, len(names):] == -1), 'Saved name ranks or padding differ')
    require(top.dtype == np.int32 and top.shape == (len(rows), columns, 10)
            and np.all((top >= 0) & (top < 49152))
            and np.all(np.diff(np.sort(top, axis=-1), axis=-1) > 0), 'Saved top-10 tokens differ')
    require(arrays['rank'].dtype == np.int32 and arrays['rank'].shape == (len(rows), 3, columns), 'Saved own rank geometry differs')
    for item, row in enumerate(rows):
        for slot, name_index in enumerate(row['own_index']):
            expected = ranks[item, name_index] if name_index >= 0 else np.full(columns, -1)
            require(np.array_equal(arrays['rank'][item, slot], expected), 'Own ranks do not match canonical name axis')
    hits = ranks[:, :len(names)] < 10
    for index, name in enumerate(names):
        alias_hit = np.isin(top, population['token_forms']['multihop'][name]).any(axis=-1)
        require(np.array_equal(alias_hit, hits[:, index]), 'Alias hit@10 differs from saved rank<10: ' + name)
    return hits


def weighted_layers(hits, weights):
    # An item-by-name weight matrix multiplies each layer's binary name hits.
    return {field: np.einsum('ij,ijk->ik', weights[field], hits, dtype=np.float64, optimize=False)
            for field in ('own', 'control')}


def contrast(layers, left, right, indices):
    columns = np.asarray(indices)
    return ((layers[left]['own'][:, columns].mean(axis=1) - layers[left]['control'][:, columns].mean(axis=1))
            - (layers[right]['own'][:, columns].mean(axis=1) - layers[right]['control'][:, columns].mean(axis=1)))


def any_contrast(hits, weights, left, right, indices):
    difference = hits[left][:, :, indices].any(axis=2).astype(np.float64) - hits[right][:, :, indices].any(axis=2)
    return np.sum((weights['own'] - weights['control']) * difference, axis=1)


def secondary_definitions():
    specs = []
    def add(name, left, right, indices, metric='fixed_mean'):
        specs.append({'id': name, 'left': left, 'right': right, 'region': list(indices), 'metric': metric})
    for loop in (1, 2, 3):
        for metric in ('fixed_mean', 'any_layer'):
            add('early_loop' + str(loop) + '_' + metric, 'fit01', 'raw', range((loop-1)*48, loop*48), metric)
    add('fit01_final_third', 'fit01', 'raw', range(176, 192))
    add('fit01_layer32', 'fit01', 'raw', [175])
    add('fit02_local', 'fit02', 'raw', BAND)
    add('fit01_minus_fit02_local', 'fit01', 'fit02', BAND)
    for arm, stop in (('penultimate', 190), ('sampled_sum', 191), ('diagonal', 191)):
        add(arm + '_local', arm, 'raw', BAND)
        add(arm + '_final_third', arm, 'raw', range(176, stop))
    for left, right, stop in (('fit01', 'penultimate', 190), ('sampled_sum', 'diagonal', 191)):
        add(left + '_minus_' + right + '_local', left, right, BAND)
        add(left + '_minus_' + right + '_final_third', left, right, range(176, stop))
    return specs


def group_layout(groups):
    labels, lookup, codes = [], {}, []
    for label in groups:
        if label not in lookup:
            lookup[label] = len(labels); labels.append(label)
        codes.append(lookup[label])
    require(len(labels) >= 2, 'At least two resampling groups are required')
    return labels, np.asarray(codes, dtype=np.int64)


def dependency_bootstrap(values, groups, seed, *, draws=DRAWS):
    """Use sampled-group multiplicities, then divide total scores by total items."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    labels, codes = group_layout(groups)
    require(len(values) == len(codes) and np.isfinite(values).all(), 'Invalid bootstrap values/groups')
    size = len(labels)
    totals = np.zeros((size, values.shape[1]))
    np.add.at(totals, codes, values)
    counts_per_group = np.bincount(codes, minlength=size)
    selected = np.random.default_rng(seed).integers(size, size=(draws, size))
    multiplicity = np.zeros((draws, size), dtype=np.int64)
    np.add.at(multiplicity, (np.repeat(np.arange(draws), size), selected.ravel()), 1)
    numerator = multiplicity @ totals
    denominator = multiplicity @ counts_per_group
    return numerator / denominator[:, None], labels, counts_per_group


def percentile95(values):
    return np.quantile(values, [.025, .975], axis=0, method='linear').T


def reconstruct(population, hits, weights):
    layers = {arm: weighted_layers(value, weights) for arm, value in hits.items()}
    eligible = weights['eligible']
    groups = [row['dependency_group_id'] for row, yes in zip(population['rows'], eligible) if yes]
    components = np.stack([layers[arm][field][:, BAND].mean(axis=1)[eligible]
                           for arm, field in [('fit01', 'own'), ('fit01', 'control'), ('raw', 'own'), ('raw', 'control')]], axis=1)
    intended = components[:, 0] - components[:, 2]
    controls = components[:, 1] - components[:, 3]
    primary = intended - controls
    allvalues = np.column_stack([primary, components, intended, controls])
    boots, labels, group_sizes = dependency_bootstrap(allvalues, groups, SEED)
    item_boots, _, _ = dependency_bootstrap(primary, list(range(len(primary))), SEED+1)
    specifications = secondary_definitions()
    secondary = np.stack([(contrast(layers, spec['left'], spec['right'], spec['region']) if spec['metric'] == 'fixed_mean'
                           else any_contrast(hits, weights, spec['left'], spec['right'], spec['region']))[eligible]
                          for spec in specifications], axis=1)
    secondary_boots, _, _ = dependency_bootstrap(secondary, groups, SEED+2)
    estimates = secondary.mean(axis=0)
    deviations = secondary_boots.std(axis=0, ddof=1)
    standardized = np.abs(secondary_boots - estimates) / np.where(deviations == 0, 1, deviations)
    critical = float(np.quantile(standardized.max(axis=1), .95, method='linear'))
    heterogeneity = []
    for group, count in zip(labels, group_sizes):
        selected = np.asarray([value == group for value in groups])
        heterogeneity.append({'group': group, 'items': int(count), 'primary_mean': float(primary[selected].mean()),
                              'without_group': float(primary[~selected].mean()),
                              'intended_difference': float(intended[selected].mean()),
                              'control_difference': float(controls[selected].mean())})
    return {'layers': layers, 'groups': groups, 'group_labels': labels, 'group_sizes': group_sizes,
            'allvalues': allvalues, 'primary': primary, 'components': components, 'family_bootstrap': boots,
            'primary_item_bootstrap': item_boots, 'secondary': secondary, 'secondary_bootstrap': secondary_boots,
            'secondary_specs': specifications, 'secondary_estimates': estimates, 'secondary_std': deviations,
            'secondary_critical': critical, 'heterogeneity': heterogeneity}


def compare_analysis(actual, saved, rebuilt, weights, checks):
    checks.exact('analysis.schema', actual['schema'], 'confirmation_analysis.v1')
    checks.exact('analysis.eligible_items', actual['eligible_items'], int(weights['eligible'].sum()))
    checks.exact('analysis.groups', actual['groups'], len(rebuilt['group_labels']))
    checks.exact('analysis.component_order', actual['component_order'], COMPONENTS)
    checks.exact('analysis.bootstrap', actual['bootstrap'], {
        'replicates': DRAWS, 'seed': SEED,
        'resampling': 'whole dependency groups with replacement; sum divided by sampled item count',
        'secondary_policy': '20 contrasts; centered bootstrap max-t, one shared 95% interval family'})
    for key in ('primary', 'components', 'secondary', 'family_bootstrap', 'secondary_bootstrap'):
        require(saved[key].dtype == np.float64, 'Saved statistic dtype differs: ' + key)
        checks.numeric('saved.' + key, saved[key], rebuilt[key])
    checks.numeric('analysis.estimates', actual['estimates'], rebuilt['allvalues'].mean(axis=0))
    checks.numeric('analysis.family_percentile_95_intervals', actual['family_percentile_95_intervals'], percentile95(rebuilt['family_bootstrap']))
    checks.numeric('analysis.primary_item_resampling_95_interval', actual['primary_item_resampling_95_interval'], percentile95(rebuilt['primary_item_bootstrap'])[0])
    checks.exact('analysis.heterogeneity_group_order', [row['group'] for row in actual['group_heterogeneity']], rebuilt['group_labels'])
    checks.exact('analysis.heterogeneity_group_sizes', [row['items'] for row in actual['group_heterogeneity']], rebuilt['group_sizes'].tolist())
    for old, new in zip(actual['group_heterogeneity'], rebuilt['heterogeneity']):
        for field in ('primary_mean', 'without_group', 'intended_difference', 'control_difference'):
            checks.numeric('heterogeneity.' + str(new['group']) + '.' + field, old[field], new[field])
    left_out = [row['without_group'] for row in rebuilt['heterogeneity']]
    checks.numeric('analysis.leave_one_group_range', actual['leave_one_group_range'], [min(left_out), max(left_out)])
    checks.exact('analysis.secondary_count', len(actual['secondary_family']), 20)
    checks.numeric('analysis.secondary_max_t_95_quantile', actual['secondary_max_t_95_quantile'], rebuilt['secondary_critical'])
    for index, spec in enumerate(rebuilt['secondary_specs']):
        old = actual['secondary_family'][index]
        checks.exact('secondary.' + spec['id'] + '.definition', {key: old[key] for key in spec}, spec)
        estimate, radius = rebuilt['secondary_estimates'][index], rebuilt['secondary_critical'] * rebuilt['secondary_std'][index]
        checks.numeric('secondary.' + spec['id'] + '.estimate', old['estimate'], estimate)
        checks.numeric('secondary.' + spec['id'] + '.simultaneous_interval', old['simultaneous_95_interval'], [estimate-radius, estimate+radius])
    checks.exact('analysis.curve_arms', sorted(actual['descriptive_curves']), sorted(ARMS))
    for arm in ARMS:
        values = rebuilt['layers'][arm]
        for field in ('own', 'control', 'excess'):
            curve = values[field] if field != 'excess' else values['own'] - values['control']
            checks.numeric('curve.' + arm + '.' + field, actual['descriptive_curves'][arm][field], curve[weights['eligible']].mean(axis=0))


def verify(results, analysis, records, checks):
    for relative, expected in FROZEN.items():
        raw = read_bytes(ROUND / relative, records)
        checks.exact('frozen.' + relative, byte_record(raw), expected)
    spec = read_json(results / 'run_spec.json', records)
    population = read_json(results / 'population.json', records)
    benchmark = read_json(results / 'benchmark.json', records)
    for name in ('run_spec', 'population', 'benchmark'):
        checks.exact('results.' + name + '.frozen_record', records[str(regular(results / (name + '.json')))], FROZEN['bundle/frozen/' + name + '.json'])
    checks.exact('frozen.primary_band', spec['primary']['virtual_indices'], list(BAND))
    checks.exact('frozen.physical_band', benchmark['design']['primary_readout']['physical_layers'], list(range(26, 38)))
    checks.exact('frozen.uncertainty', spec['uncertainty'], {'draws': DRAWS, 'groups': 'dependency_group_id', 'primary_interval': 'percentile95', 'secondary_family': 'centered bootstrap max-t95, 20 contrasts', 'seed': SEED})
    checks.exact('frozen.secondary_definitions', spec['secondary_specs'], secondary_definitions())
    weights = population_weights(population, frozen=True)
    checks.exact('population.item_name_order', [row['name'] for row in population['rows']], [row['name'] for row in benchmark['items']])
    for row, annotation in zip(population['rows'], benchmark['items']):
        for key in ('task', 'prompt', 'target', 'intermediates', 'dependency_group_id'):
            require(row[key] == annotation[key], 'Population annotation differs: ' + row['name'] + '/' + key)
    hits = {}
    for arm in ARMS:
        checks.exact('frozen.support.' + arm, spec['arm_support'][arm] if arm in spec['arm_support'] else list(range(192)), list(range(LENGTHS[arm])))
        arrays = read_npz(results / 'readouts' / (arm + '.npz'), records, ('allrank', 'rank', 'top10_ids'))
        hits[arm] = rank_hits(arrays, population, LENGTHS[arm])
    actual = read_json(analysis / 'analysis.json', records)
    saved = read_npz(analysis / 'item_statistics_and_resamples.npz', records,
                     ('primary', 'components', 'secondary', 'family_bootstrap', 'secondary_bootstrap'))
    rebuilt = reconstruct(population, hits, weights)
    checks.exact('frozen.group_count', len(rebuilt['group_labels']), spec['dependency_groups'])
    compare_analysis(actual, saved, rebuilt, weights, checks)
    early = [index for index, spec in enumerate(rebuilt['secondary_specs']) if spec['id'] in
             ('early_loop1_fixed_mean', 'early_loop2_fixed_mean', 'early_loop3_fixed_mean')]
    return {'run_id': spec['run_id'], 'item_count': len(population['rows']), 'eligible_item_count': int(weights['eligible'].sum()),
            'name_count': len(weights['names']), 'controls_per_eligible_label': 79,
            'group_order': rebuilt['group_labels'], 'group_sizes': rebuilt['group_sizes'].tolist(),
            'bootstrap_replicates': DRAWS, 'seeds': {'dependency_primary': SEED, 'item_sensitivity': SEED+1, 'secondary_family': SEED+2},
            'primary_band': {'zero_based_virtual': list(BAND), 'loop_one_based': 4, 'physical_layers_one_based': list(range(26, 38))},
            'component_order': COMPONENTS, 'estimates': rebuilt['allvalues'].mean(axis=0).tolist(),
            'family_percentile_95_intervals': percentile95(rebuilt['family_bootstrap']).tolist(),
            'primary_item_resampling_95_interval': percentile95(rebuilt['primary_item_bootstrap'])[0].tolist(),
            'group_heterogeneity': rebuilt['heterogeneity'],
            'early_fixed_loop_1_2_3_estimates': rebuilt['secondary'][:, early].mean(axis=0).tolist(),
            'per_item': [{'name': row['name'], 'dependency_group_id': row['dependency_group_id'],
                          'components': rebuilt['allvalues'][index].tolist(),
                          'early_fixed_loop_1_2_3': rebuilt['secondary'][index, early].tolist()}
                         for index, row in enumerate([row for row, yes in zip(population['rows'], weights['eligible']) if yes])],
            'reconstructed_array_records': {key: array_record(rebuilt[key]) for key in
                ('primary', 'components', 'allvalues', 'family_bootstrap', 'primary_item_bootstrap', 'secondary', 'secondary_bootstrap')},
            'saved_array_records': {key: array_record(value) for key, value in saved.items()},
            'scope': 'All frozen primary/component, item-sensitivity, group, 20-secondary-family and descriptive-curve numerical outputs; no result selection or model execution.'}


def self_test():
    cases = []
    def rejected(operation, label):
        try:
            operation()
        except ValueError:
            cases.append(label); return
        raise AssertionError('Invalid synthetic fixture was accepted: ' + label)
    names = ['alpha', 'beta', 'gamma', 'delta']
    forms = {'alpha': [1, 2], 'beta': [3], 'gamma': [4], 'delta': [5]}
    rows = []
    for index, labels in enumerate([['alpha', 'beta'], ['alpha'], ['alpha'], ['alpha']]):
        own = [names.index(label) for label in labels]
        rows.append({'name': 'synthetic-' + str(index), 'task': 'multihop', 'intermediates': labels,
                     'dependency_group_id': ['large', 'large', 'singleton', 'large'][index],
                     'own_index': own + [-1] * (3-len(own)), 'eligible': [True]*len(labels),
                     'scorable': [True]*len(labels), 'leaked': [False]*len(labels),
                     'control_indices': [[j for j in range(4) if j not in own] for _ in labels],
                     'intermediate_tokens': {label: forms[label] for label in labels}, 'token_ids': [1000]})
    population = {'schema': 'confirmation_population.v1', 'rows': rows, 'names': {'multihop': names},
                  'token_forms': {'multihop': forms}, 'cross_concept_token_collisions': []}
    rows[0]['intermediate_tokens']['alpha'] = [2, 1]  # Alias order is immaterial to the minimum rank.
    weights = population_weights(population)
    require(np.array_equal(weights['own'][0], [.5, .5, 0, 0])
            and np.array_equal(weights['control'][0], [0, 0, .5, .5]), 'Multiple-label hand weights differ')
    ranks = np.full((4, 128, 192), -1, np.int32); ranks[:, :4] = 100
    ranks[0, 0, :6] = 9; ranks[0, 1, :6] = 10; ranks[0, 2, :6] = 0
    ranks[1:, 0, :12] = 0
    hits = ranks[:, :4] < 10
    measured = weighted_layers(hits, weights)
    require(abs(measured['own'][0, :12].mean() - .25) < TOLERANCE
            and abs(measured['control'][0, :12].mean() - .25) < TOLERANCE
            and abs(measured['own'][:, :12].mean(axis=1).mean() - .8125) < TOLERANCE,
            'Hand calculation must average labels, then layers, then equal items')
    pooled_labels = (.5 + 0 + 1 + 1 + 1) / 5
    require(abs(pooled_labels - .8125) > .1, 'Fixture fails to distinguish pooled labels')
    cases.append('hand_calculated_multiple_eligible_labels_equal_layers_and_equal_items_not_pooled_labels')
    contaminated = json.loads(json.dumps(population)); contaminated['rows'][0]['control_indices'][0] = [0, 2]
    rejected(lambda: population_weights(contaminated), 'own_name_contamination_rejected')
    aliases = json.loads(json.dumps(population)); aliases['token_forms']['multihop']['beta'] = [2]
    rejected(lambda: population_weights(aliases), 'cross_concept_alias_contamination_rejected')
    ineligible = json.loads(json.dumps(population)); ineligible['rows'][0]['token_ids'] = [3]
    ineligible['rows'][0]['leaked'][1] = True; ineligible['rows'][0]['eligible'][1] = False
    changed_weights = population_weights(ineligible)
    require(np.array_equal(changed_weights['own'][0], [1, 0, 0, 0])
            and np.array_equal(changed_weights['control'][0], [0, 0, .5, .5]),
            'Ineligible own label contaminated controls or label denominator')
    cases.append('ineligible_labels_excluded_from_label_mean_but_all_own_names_excluded_from_controls')
    # The best alias is at rank 9; another alias at rank 12 is irrelevant.
    alias_ranks = np.full((4, 128, 192), -1, np.int32); alias_ranks[:, :4] = 100
    alias_ranks[:, 0] = min(12, 9); alias_ranks[:, 1] = 10
    top = np.broadcast_to(np.asarray([101,102,103,104,105,106,107,108,109,2], np.int32), (4,192,10)).copy()
    own_ranks = np.full((4,3,192), -1, np.int32)
    for item, row in enumerate(rows):
        for slot, j in enumerate(row['own_index']):
            if j >= 0: own_ranks[item,slot] = alias_ranks[item,j]
    arrays = {'allrank': alias_ranks, 'rank': own_ranks, 'top10_ids': top}
    recovered = rank_hits(arrays, population, 192)
    require(recovered[:, 0].all() and not recovered[:, 1:].any(), 'Alias min rank / strict rank<10 boundary differs')
    cases.append('minimum_alias_rank_9_hits_rank_10_misses_saved_top10_crosscheck')
    bad = {key: value.copy() for key, value in arrays.items()}; bad['top10_ids'][:, :, -1] = 110
    rejected(lambda: rank_hits(bad, population, 192), 'saved_alias_rank_and_top10_disagreement_rejected')
    values = np.asarray([[0., 1.], [2., -1.], [10., 4.], [4., 0.]])
    groups = ['large', 'large', 'singleton', 'large']
    boots, order, sizes = dependency_bootstrap(values, groups, SEED)
    require(order == ['large', 'singleton'] and sizes.tolist() == [3, 1], 'First-occurrence group order/size differs')
    rng = np.random.default_rng(SEED); draw = rng.integers(2, size=(DRAWS, 2))
    expected = np.empty_like(boots)
    members = [[0, 1, 3], [2]]
    for index, chosen in enumerate(draw):
        selected_rows = members[int(chosen[0])] + members[int(chosen[1])]
        expected[index] = values[selected_rows].mean(axis=0)
    require(np.max(np.abs(boots-expected)) < TOLERANCE, 'Whole-group ratio differs from direct repeated-item enumeration')
    require(np.allclose(values.mean(axis=0), [4, 1])
            and not np.allclose(values.mean(axis=0), [6, 2]), 'Unequal-group fixture cannot distinguish equal-item/equal-group means')
    batched = np.random.default_rng(SEED)
    require(np.array_equal(draw, np.concatenate([batched.integers(2, size=(500,2)) for _ in range(40)])), '500-draw batching changed RNG stream')
    cases.append('all_20000_dependency_draws_match_direct_repeated_item_oracle_unequal_groups_and_500_batch_stream')
    item_boots, _, _ = dependency_bootstrap(values[:,0], list(range(4)), SEED+1)
    item_draws = np.random.default_rng(SEED+1).integers(4, size=(DRAWS,4))
    require(np.max(np.abs(item_boots[:,0]-values[item_draws,0].mean(axis=1))) < TOLERANCE, 'Item sensitivity seed/weights differ')
    cases.append('all_20000_item_sensitivity_draws_match_direct_equal_item_oracle_seed_plus_one')
    require(list(BAND) == [3*48 + layer-1 for layer in range(26,38)] and len(BAND) == 12,
            'One-based physical layer mapping differs')
    cases.append('fixed_band_virtual_169_to_180_is_loop4_physical26_to37')
    # Full reconstruction plus independent hand primary and leave-one-out means.
    allhits = {arm: np.zeros((4,4,LENGTHS[arm]), dtype=bool) for arm in ARMS}
    allhits['fit01'][0,0,BAND] = True
    allhits['fit01'][1,0,BAND] = True
    allhits['fit01'][2,2,BAND] = True
    rebuilt = reconstruct(population, allhits, weights)
    primary = np.asarray([.5, 1, -1/3, 0])
    require(np.max(np.abs(rebuilt['primary']-primary)) < TOLERANCE
            and abs(rebuilt['allvalues'][:,0].mean()-7/24) < TOLERANCE,
            'Hand primary seven-component construction differs')
    require(abs(rebuilt['heterogeneity'][0]['without_group']+1/3) < TOLERANCE
            and abs(rebuilt['heterogeneity'][1]['without_group']-.5) < TOLERANCE,
            'Hand leave-one-group-out reconstruction differs')
    require(rebuilt['secondary'].shape == (4,20) and rebuilt['secondary_bootstrap'].shape == (DRAWS,20),
            'Frozen secondary family geometry differs')
    cases.append('hand_primary_components_and_leave_one_group_out_plus_complete_secondary_geometry')
    checks = Comparisons()
    rejected(lambda: checks.numeric('deliberate_drift', np.asarray([1.]), np.asarray([1.+1e-6])), 'explicit_1e_minus_12_tolerance_rejects_numerical_drift')
    return {'cases': cases, 'case_count': len(cases), 'temporary_or_synthetic_data_only': True,
            'bootstrap_replicates_per_oracle': DRAWS, 'primary_hand_estimate': 7/24,
            'producer_imports': [], 'no_model_or_provider_execution': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        require(args.results is None and args.analysis is None, 'Self-test cannot consume result/analysis paths')
    else:
        require(args.results is not None and args.analysis is not None, '--results and --analysis are required')
    require(not regular(args.out).exists(), 'Verification output already exists')
    records, checks = {}, Comparisons()
    code_record = byte_record(read_bytes(__file__, records))
    result = {'schema': 'confirmation_independent_numerical_check.v1', 'status': 'failed',
              'mode': 'synthetic_self_test' if args.self_test else 'saved_result_reconstruction',
              'utc': datetime.now(timezone.utc).isoformat(), 'code_record': code_record,
              'python': sys.version, 'numpy': np.__version__, 'absolute_tolerance': TOLERANCE, 'relative_tolerance': 0,
              'input_records': records, 'checks': checks.rows}
    try:
        if args.self_test:
            result['self_test'] = self_test()
        else:
            result['results_root'], result['analysis_root'] = str(regular(args.results)), str(regular(args.analysis))
            result['reconstruction'] = verify(regular(args.results), regular(args.analysis), records, checks)
        recheck_inputs(records)
        result['status'] = 'passed'
        result['max_absolute_difference'] = max((row.get('max_absolute_difference', 0) for row in checks.rows), default=0)
        result['comparison_count'] = len(checks.rows)
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    publish(args.out, result)
    print(json.dumps({'status': result['status'], 'proof': str(regular(args.out)),
                      'max_absolute_difference': result.get('max_absolute_difference'),
                      'case_count': result.get('self_test', {}).get('case_count'),
                      'error': result.get('error')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
