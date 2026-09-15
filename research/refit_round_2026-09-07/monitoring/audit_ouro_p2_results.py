"""Independent local audit of an actual, verified Ouro P1/P2 handoff.

No fitting, inference, network, cloud mutation, tensor-cache deserialization,
new item selection, alternative scientific estimand, or interpretation receipt.
Run only after root authorizes the completed actual handoff and full analysis.
Only a new audit JSON under /tmp may be written; all scientific artifacts stay
read-only. P2 remains conditional on the preselected fit01 control experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROUND = Path('/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07')
sys.path.insert(0, str(ROUND / 'analysis'))
sys.path.insert(0, str(ROUND / 'monitoring'))
import analyze_refits as frozen
import analyze_main_refits as p1_only
import collect_ouro_handoff06 as handoff

io = frozen.io
TASKS = ('multihop', 'order-ops')
DRAWS = 10000
SEED = 2026090805
EXAMPLE_SEED = 2026090806
FAILURE_SEED = 2026090807
ACCEPTED_P1_ID = '7b127cfa493775305db15bcd5ab2c566bdf05c7154467444b5ca02b4f22db406'
FIELDS = ('own_layer', 'control_layer', 'delta_layer', 'own_regions', 'control_regions', 'delta_regions')
OPERATIONS = {'addition', 'subtraction', 'multiplication', 'division', 'mod', 'squared'}
NUMBER_WORDS = {word: str(i) for i, word in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
NUMBER_WORDS['third'] = '3'
BANDS = {'early': (0, 16), 'middle': (16, 32), 'final_third': (32, 48),
         'historical_local': (25, 37), 'full_loop': (0, 48)}
SUPPORT = {'target': list(range(190)), 'position': list(range(191))}
LATE = {'target': list(range(176, 190)), 'position': list(range(176, 191))}
PAIRS = {
    'target_main_minus_penultimate': ('target_main', 'target_penultimate'),
    'target_main_minus_raw': ('target_main', 'target_raw'),
    'target_penultimate_minus_raw': ('target_penultimate', 'target_raw'),
    'position_sampled_sum_minus_diagonal': ('position_sampled_sum', 'position_diagonal'),
    'position_main_minus_sampled_sum': ('position_main', 'position_sampled_sum'),
    'position_main_minus_raw': ('position_main', 'position_raw'),
    'position_sampled_sum_minus_raw': ('position_sampled_sum', 'position_raw'),
    'position_diagonal_minus_raw': ('position_diagonal', 'position_raw'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right, message):
    require(io._canonical(left) == io._canonical(right), message)


def close(left, right, message, errors, *, tolerance=1e-12):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, message + ': shape')
    if np.issubdtype(left.dtype, np.bool_) or np.issubdtype(left.dtype, np.integer):
        require(np.array_equal(left, right), message + ': exact values')
        return
    require(np.isfinite(left).all() and np.isfinite(right).all(), message + ': nonfinite')
    error = float(np.max(np.abs(left - right), initial=0.0))
    errors['maximum_absolute_difference'] = max(errors['maximum_absolute_difference'], error)
    require(error <= tolerance, message + f': absolute difference {error}')


def npz(path):
    # Explicitly refuse pickle-backed NumPy members; cache.pt is never passed here.
    require(Path(path).suffix == '.npz', 'numeric reader only accepts NPZ paths')
    with np.load(io._no_links(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def track(path, records, expected=None):
    path = io._no_links(path)
    record = io._record(path) if expected is None else expected
    io._verify_record(path, record)
    previous = records.get(str(path))
    require(previous is None or previous == record, 'conflicting consumed record: ' + str(path))
    records[str(path)] = record
    return path


def normalize(label):
    label = label.lower()
    return NUMBER_WORDS.get(label, str(int(label)) if label.isdigit() else label)


def population(rows, names, forms):
    """Reconstruct masks/pools and token-form leakage without the frozen scorer."""
    require(len(rows) == 148 and set(names) == set(TASKS), 'unexpected population geometry')
    same(io._digest([{key: row[key] for key in frozen.evaluation.POPULATION_FIELDS} for row in rows]),
         frozen.evaluation.POPULATION_SHA256, 'frozen population hash')
    same(io._digest(names), frozen.evaluation.TASK_NAMES_SHA256, 'frozen catalogue hash')
    same(io._digest(forms), frozen.evaluation.TOKEN_FORMS_SHA256, 'frozen token forms hash')
    slots = []
    for i, row in enumerate(rows):
        own = {index for index in row['own_index'] if index >= 0}
        eligible, selected = [], []
        for s, label in enumerate(row['intermediates']):
            token_ids = row['intermediate_tokens'][label]
            scorable = bool(token_ids)
            leaked = bool(set(token_ids) & set(row['token_ids']))
            require(row['scorable'][s] == scorable and row['leaked'][s] == leaked,
                    f'item {i}: scorable/leakage convention')
            if scorable:
                require(set(token_ids) == set(forms[row['task']][label]), f'item {i}: own token aliases')
            expected = scorable and not leaked and (row['task'] == 'multihop' or label not in OPERATIONS)
            eligible.append(expected)
            pool = [j for j, other in enumerate(names[row['task']])
                    if j not in own and ((other in OPERATIONS) == (label in OPERATIONS))] if scorable else []
            same(row['control_indices'][s], pool, f'item {i}: historical matched controls')
            if expected:
                require(pool and names[row['task']][row['own_index'][s]] == label, f'item {i}: own/control labels')
                selected.append(s)
        same(row['eligible'], eligible, f'item {i}: eligibility')
        answer = row['target'].strip().lower()
        continuation = row['continuation'].strip().strip('"').lower()
        correct = continuation.startswith(answer) and (len(continuation) == len(answer)
                                                       or not continuation[len(answer)].isalnum())
        require(type(row['correct']) is bool and row['correct'] == correct, f'item {i}: historical matcher')
        slots.append(selected)
    for task, nitems, nslots in [('multihop', 90, 100), ('order-ops', 51, 51)]:
        indices = [i for i, row in enumerate(rows) if row['task'] == task and slots[i]]
        require(len(indices) == nitems and sum(len(slots[i]) for i in indices) == nslots,
                task + ': fixed eligible denominators')
    return slots


def readout(path, prefix, columns, rows, names):
    archive = npz(path)
    arrays = {key[len(prefix):]: value for key, value in archive.items() if key.startswith(prefix)}
    require({'allrank', 'rank', 'top1'} <= set(arrays), 'readout lacks rank banks')
    ranks, own = arrays['allrank'], arrays['rank']
    require(ranks.shape == (148, 128, columns) and ranks.dtype == np.int32, 'allrank support/dtype')
    require(own.shape == (148, 3, columns) and own.dtype == np.int32, 'own-rank support/dtype')
    for key, value in arrays.items():
        require(value.ndim >= 2 and value.shape[0] == 148 and value.shape[-1] == columns,
                key + ': unsupported virtual columns')
        require(not np.issubdtype(value.dtype, np.floating) or np.isfinite(value).all(), key + ': finite')
    for i, row in enumerate(rows):
        count = len(names[row['task']])
        require(np.all(ranks[i, :count] >= 0) and np.all(ranks[i, count:] == -1),
                f'item {i}: real rank/sentinel name padding')
        for s, name_index in enumerate(row['own_index']):
            expected = ranks[i, name_index] if name_index >= 0 else np.full(columns, -1, np.int32)
            require(np.array_equal(own[i, s], expected), f'item {i}: own rank/sentinel slot')
    if prefix == 'logitlens_':
        require(np.array_equal(archive['exit_top1'], arrays['top1'][:, [47, 95, 143, 191]]),
                'raw native exits differ')
    return arrays


def score(ranks, support, late, rows, slots):
    """Independent item-first score; controls take their own any-layer event."""
    columns = len(support)
    require(ranks.shape == (148, 128, columns), 'score bank support shape')
    late_columns = [support.index(index) for index in late]
    own = np.zeros((148, columns)); control = np.zeros_like(own)
    own_regions = np.zeros((148, 2)); control_regions = np.zeros_like(own_regions)
    for i, row in enumerate(rows):
        if not slots[i]:
            continue
        own_hits, control_hits, own_any, control_any = [], [], [], []
        for s in slots[i]:
            pool = row['control_indices'][s]
            own_rank = ranks[i, row['own_index'][s]]
            control_rank = ranks[i, pool]
            require(np.all(own_rank >= 0) and np.all(control_rank >= 0), 'unsupported rank entered scoring')
            true = (own_rank >= 0) & (own_rank < 10)
            other = (control_rank >= 0) & (control_rank < 10)
            own_hits.append(true.astype(float))
            control_hits.append(other.sum(axis=0) / len(pool))
            own_any.append([float(true.any()), float(true[late_columns].any())])
            control_any.append([sum(row_hits.any() for row_hits in other) / len(pool),
                                sum(row_hits[late_columns].any() for row_hits in other) / len(pool)])
        own[i], control[i] = np.mean(own_hits, axis=0), np.mean(control_hits, axis=0)
        own_regions[i], control_regions[i] = np.mean(own_any, axis=0), np.mean(control_any, axis=0)
    return {'eligible': np.asarray([bool(selected) for selected in slots]),
            'own_layer': own, 'control_layer': control, 'delta_layer': own - control,
            'own_regions': own_regions, 'control_regions': control_regions,
            'delta_regions': own_regions - control_regions}


def metric_definitions(comparison):
    support, late = SUPPORT[comparison], LATE[comparison]
    definitions = []
    for loop in range(4):
        for band, (lo, hi) in BANDS.items():
            selected = [index for index in range(loop * 48 + lo, loop * 48 + hi) if index in support]
            require(selected, 'empty common-support band')
            definitions.append({'id': f'loop{loop + 1}_{band}_mean', 'kind': 'fixed_mean', 'loop': loop + 1,
                                'band': band, 'virtual_indices': selected, 'historical_selected': band == 'historical_local'})
    definitions.extend([{'id': 'all_common_any_layer', 'kind': 'any_layer', 'virtual_indices': support},
                        {'id': 'loop4_late_any_layer', 'kind': 'any_layer', 'virtual_indices': late}])
    require(len(definitions) == 22, 'P2 must have 22 metrics, without depth slopes')
    return definitions


def metric_values(layer, regions, comparison):
    support = SUPPORT[comparison]
    values = []
    for definition in metric_definitions(comparison):
        if definition['kind'] == 'fixed_mean':
            columns = [support.index(index) for index in definition['virtual_indices']]
            values.append(layer[:, columns].mean(axis=1))
    return np.column_stack([*values, regions[:, 0], regions[:, 1]])


def counts(draws, size):
    require(draws.ndim == 2 and draws.shape[0] == DRAWS and np.issubdtype(draws.dtype, np.integer),
            'bootstrap membership geometry')
    require(np.all((draws >= 0) & (draws < size)), 'bootstrap index bounds')
    result = np.zeros((DRAWS, size))
    np.add.at(result, (np.repeat(np.arange(DRAWS), draws.shape[1]), draws.ravel()), 1.0)
    return result


def bootstrap_memberships(saved, rows, slots):
    fit_indices = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,))).integers(0, 5, size=(DRAWS, 5))
    require(np.array_equal(saved['bootstrap_fit_indices'], fit_indices), 'unchanged P1 fit stream')
    result, descriptions = {}, {}
    for task_index, task in enumerate(TASKS):
        indices = np.asarray([i for i, row in enumerate(rows) if row['task'] == task and slots[i]], dtype=np.int64)
        require(np.array_equal(saved[f'bootstrap_{task}_indices'], indices), task + ': item order')
        groups = saved[f'bootstrap_{task}_groups']
        require(groups.shape == indices.shape and np.issubdtype(groups.dtype, np.integer), task + ': group shape')
        require(np.array_equal(np.unique(groups), np.arange(int(groups.max()) + 1)), task + ': contiguous groups')
        concepts = [{normalize(rows[i]['intermediates'][s]) for s in slots[i]} for i in indices]
        connected = np.asarray([[bool(a & b) for b in concepts] for a in concepts])
        for k in range(len(indices)):
            connected |= connected[:, k, None] & connected[None, k, :]
        require(np.array_equal(connected, groups[:, None] == groups[None, :]), task + ': shared-concept graph')
        ncomponents = int(groups.max()) + 1
        item_draws, component_draws = saved[f'bootstrap_{task}_item_indices'], saved[f'bootstrap_{task}_component_indices']
        expected_items = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(1, task_index))).integers(
            0, len(indices), size=(DRAWS, len(indices)))
        expected_components = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(2, task_index))).integers(
            0, ncomponents, size=(DRAWS, ncomponents))
        require(np.array_equal(item_draws, expected_items), task + ': fixed item stream')
        require(np.array_equal(component_draws, expected_components), task + ': fixed component stream')
        item_weights = counts(item_draws, len(indices)) / len(indices)
        # Expand each sampled component to all its items; normalize by sampled
        # item count, rather than giving large and small components equal means.
        component_weights = counts(component_draws, ncomponents)[:, groups]
        component_weights /= component_weights.sum(axis=1, keepdims=True)
        result[task] = {'indices': indices, 'conditional_item': item_weights, 'conditional_component': component_weights}
        sizes = np.bincount(groups)
        descriptions[task] = {'eligible_items': len(indices), 'eligible_slots': sum(len(slots[i]) for i in indices),
                              'components': ncomponents, 'component_sizes': sizes.tolist(),
                              'largest_component_fraction': float(sizes.max() / sizes.sum())}
    return result, descriptions


def verify_summary(summary, banks, contrasts, rows, slots, errors):
    for field, expected in [('conditional_on_fit_id', 1), ('n_independent_control_fits', 1),
                            ('target_support', SUPPORT['target']), ('position_support', SUPPORT['position']),
                            ('target_late_band', LATE['target']), ('position_late_band', LATE['position']),
                            ('regions', ['all_common', 'loop4_late'])]:
        same(summary[field], expected, 'producer summary: ' + field)
    require(set(summary['scores']) == set(TASKS) and set(summary['contrasts']) == set(TASKS), 'summary tasks')
    for task in TASKS:
        for stratum in ('all', 'correct', 'incorrect'):
            indices = [i for i, row in enumerate(rows) if row['task'] == task and slots[i]
                       and (stratum == 'all' or row['correct'] == (stratum == 'correct'))]
            for group, values in [('scores', banks), ('contrasts', contrasts)]:
                observed = summary[group][task][stratum]
                require(set(observed) == set(values), 'producer summary arm/contrast membership')
                for name, bank in values.items():
                    require(observed[name]['eligible_items'] == len(indices), 'producer summary denominator')
                    for field in FIELDS:
                        if indices:
                            close(bank[field][indices].mean(axis=0), observed[name][field],
                                  f'producer summary {task}/{stratum}/{name}/{field}', errors)
                        else:
                            require(observed[name][field] is None, 'empty producer stratum must be null')


def frozen_examples(document, rows, slots, metric_banks, contrast_metric_banks):
    eligible = np.asarray([i for i, selected in enumerate(slots) if selected])
    selected = np.random.default_rng(EXAMPLE_SEED).choice(eligible, min(10, len(eligible)), replace=False)
    require(selected.tolist() == [11, 66, 25, 139, 72, 3, 1, 108, 15, 119], 'frozen random item indices')
    same(document['random']['selected_indices'], selected.tolist(), 'report random selection')
    groups = {'random': selected.tolist()}
    for task_index, task in enumerate(TASKS):
        pool = np.asarray([i for i in eligible if rows[i]['task'] == task and not rows[i]['correct']])
        chosen = np.random.default_rng(np.random.SeedSequence(FAILURE_SEED, spawn_key=(task_index,))).choice(
            pool, min(5, len(pool)), replace=False)
        saved = document['failures']['tasks'][task]
        same(saved['eligible_failure_indices'], pool.tolist(), task + ': fixed failure pool')
        same(saved['selected_indices'], chosen.tolist(), task + ': fixed failure selection')
        groups['failures_' + task] = chosen.tolist()
    records = {}
    for group, indices in groups.items():
        records[group] = []
        for i in indices:
            row = rows[i]
            record = {'index': i, **{key: row[key] for key in ('name', 'task', 'prompt', 'target', 'continuation', 'correct', 'eligible')},
                      'P2_contrasts': {}, 'P2_arm_own_control_excess': {}}
            for name, values in contrast_metric_banks.items():
                comparison = name.split('_', 1)[0]
                record['P2_contrasts'][name] = {definition['id']: float(values[i, j])
                    for j, definition in enumerate(metric_definitions(comparison))}
            for arm, by_field in metric_banks.items():
                comparison = arm.split('_', 1)[0]
                record['P2_arm_own_control_excess'][arm] = {definition['id']: {
                    field: float(values[i, j]) for field, values in by_field.items()}
                    for j, definition in enumerate(metric_definitions(comparison))}
            records[group].append(record)
    return {'selection': 'Exactly the accepted P1 frozen random/failure samples; no P2-effect-based selection.',
            'failure_rule': 'Frozen short-continuation prefix matcher; no fresh factual-error adjudication.',
            'records': records}


def audit(args):
    consumed, errors = {}, {'maximum_absolute_difference': 0.0}
    proof_root = io._no_links(args.handoff)
    analysis = io._no_links(args.analysis)
    accepted = io._no_links(args.accepted_p1)
    out = io._no_links(args.output)
    require(out.is_absolute() and Path('/tmp') in out.parents, 'audit output must be a new absolute /tmp path')
    require(not out.exists(), 'refusing to overwrite any audit output')
    for root in (proof_root, analysis, accepted):
        require(root != out and root not in out.parents, 'audit output cannot be in an input tree')
    source_record = io._record(Path(__file__))
    full_validation = frozen.validate_output(analysis)
    accepted_validation = p1_only.validate_output(accepted)
    require(accepted_validation['identity_sha256'] == ACCEPTED_P1_ID, 'unexpected accepted P1 identity')
    for root in (analysis, accepted):
        complete = io._json(track(root / 'COMPLETE.json', consumed))
        for name, record in complete['files'].items():
            track(root / name, consumed, record)
    owner = io._json(analysis / 'OWNER.json')['identity']
    accepted_owner = io._json(accepted / 'OWNER.json')['identity']
    require(owner['kind'] == 'five_fit_ouro_and_conditional_controls', 'not a completed combined P1/P2 analysis')
    same(owner['fit_ids'], [1, 2, 3, 4, 5], 'combined fit order')
    authored_paths = [str(Path(frozen.__file__)), str(ROUND / 'analysis/CONTRACT.md')]
    same(owner['analysis_sources'], {name: accepted_owner['analysis_sources'][name] for name in authored_paths},
         'full analysis changed the accepted frozen statistical source or contract')
    require(owner['bootstrap_draws'] == DRAWS and owner['bootstrap_seed'] == SEED, 'combined uncertainty identity')
    for source_group in ('analysis_sources', 'frozen_source_records'):
        for name, record in owner[source_group].items():
            track(Path(name), consumed, record)
    receipt = io._json(track(proof_root / 'COPY_VERIFIED.json', consumed))
    require(receipt['status'] == 'passed' and receipt['lease_name'] == handoff.LEASE_NAME
            and receipt['pod_id'] == handoff.POD_ID, 'unexpected verified handoff identity')
    results = proof_root / 'results'
    contract_records = {name: io._record(frozen.DEPLOYMENT / name) for name in handoff.CONTRACTS}
    for name, record in contract_records.items():
        track(frozen.DEPLOYMENT / name, consumed, record)
    snapshots = [io._json(track(proof_root / name, consumed)) for name in ('REMOTE_BEFORE.json', 'REMOTE_AFTER.json')]
    for snapshot in snapshots:
        handoff.verify_snapshot(snapshot, contract_records)
        same(snapshot['files'], receipt['files'], 'handoff before/after/receipt file binding')
        for key in ('status_before', 'status_after'):
            same(snapshot[key]['required_bindings'], receipt['bindings'], 'handoff required-binding stability')
    same(handoff.local_inventory(results), receipt['files'], 'handoff exact local inventory')
    for name, record in receipt['files'].items():
        track(results / name, consumed, record)
    main_root, controls_root = results / 'ouro_evaluation', results / 'controls_evaluation'
    contract = io._contract(frozen.DEPLOYMENT / 'run_spec.json', args.ouro_src, [1, 2, 3, 4, 5])
    frozen.evaluation._evaluation_sources(contract)
    require(owner['run_spec_sha256'] == accepted_owner['run_spec_sha256'] == contract['spec_sha256'],
            'full/accepted/evaluation run specification binding')
    combined = io._json(results / 'combined_contract.json')
    main_checked = frozen.controls.validate_completed_main(main_root, contract, require_all=True)
    controls_checked = frozen.controls.validate_completed_controls(controls_root, contract, combined, main_checked)
    require(set(owner['evaluation_artifacts']) == {'main', 'controls'}, 'combined artifact families')
    for family, root, checked in [('main', main_root, main_checked), ('controls', controls_root, controls_checked)]:
        require(Path(owner['evaluation_artifacts'][family]['root']) == root, 'analysis consumed a different handoff root')
        expected_records = frozen.artifact_records(root, checked)
        same(owner['evaluation_artifacts'][family]['files'], expected_records, 'analysis input artifact records')
        for name, record in expected_records.items():
            same(record, receipt['files'][root.name + '/' + name], 'analysis/transfer record mismatch')
    same(owner['evaluation_artifacts']['main']['files'], accepted_owner['evaluation_artifacts']['main']['files'],
         'main evaluation changed after accepted P1')
    report, previous_report = io._json(analysis / 'report.json'), io._json(accepted / 'report.json')
    for field in ('P1', 'uncertainty', 'populations'):
        same(report[field], previous_report[field], 'accepted P1 exact equality: ' + field)
    for name in ('main_cells.csv', 'main_metrics.csv', 'per_fit_metrics.csv', 'examples.json'):
        require((analysis / name).read_bytes() == (accepted / name).read_bytes(), 'accepted P1 bytes: ' + name)
    saved, previous_saved = npz(analysis / 'paired_items.npz'), npz(accepted / 'paired_items.npz')
    selected_keys = {name for name in saved if name.startswith(('main_', 'bootstrap_'))}
    require(selected_keys == set(previous_saved), 'accepted P1 archive membership')
    for name in selected_keys:
        require(np.array_equal(saved[name], previous_saved[name]), 'accepted P1 array: ' + name)
    rows, names, forms = (io._json(main_root / 'common' / name) for name in ('items.json', 'task_names.json', 'token_forms.json'))
    for name, value in [('items.json', rows), ('task_names.json', names), ('token_forms.json', forms)]:
        same(io._json(controls_root / 'comparisons' / name), value, 'P2 paired population: ' + name)
    slots = population(rows, names, forms)
    raw = readout(main_root / 'common/arrays.npz', 'logitlens_', 192, rows, names)
    main = readout(main_root / 'fits/fit_01/arrays.npz', 'jlens_exit3_', 192, rows, names)
    for key in ('allrank', 'rank', 'top1'):
        require(np.array_equal(main[key][..., 191], raw[key][..., 191]), 'main raw identity191 mismatch')
    arms = {'penultimate': readout(controls_root / 'penultimate/arrays.npz', '', 190, rows, names),
            'sampled_sum': readout(controls_root / 'positions/sampled_sum/arrays.npz', '', 191, rows, names),
            'diagonal': readout(controls_root / 'positions/diagonal/arrays.npz', '', 191, rows, names)}
    retained_ranks = npz(controls_root / 'comparisons/paired_ranks.npz')
    expected_retained = {}
    for comparison, support in SUPPORT.items():
        for arm, arrays in [('main', main), ('raw', raw)]:
            for name, value in arrays.items():
                expected_retained[f'{comparison}_{arm}_{name}'] = value[..., support]
    require(set(retained_ranks) == set(expected_retained), 'retained rank-bank membership')
    for name, value in expected_retained.items():
        require(np.array_equal(value, retained_ranks[name]), 'retained paired ranks: ' + name)
    banks = {}
    for comparison, support in SUPPORT.items():
        readouts = {'main': main['allrank'][..., support], 'raw': raw['allrank'][..., support]}
        readouts.update({'penultimate': arms['penultimate']['allrank']} if comparison == 'target' else
                        {'sampled_sum': arms['sampled_sum']['allrank'], 'diagonal': arms['diagonal']['allrank']})
        for arm, ranks in readouts.items():
            banks[comparison + '_' + arm] = score(ranks, support, LATE[comparison], rows, slots)
    contrasts = {}
    expected_scores = {arm + '_' + field: value for arm, bank in banks.items() for field, value in bank.items()}
    for name, (left, right) in PAIRS.items():
        require(np.array_equal(banks[left]['eligible'], banks[right]['eligible']), 'contrast eligibility mismatch')
        contrasts[name] = {field: banks[left][field] - banks[right][field] for field in FIELDS}
        for field, value in contrasts[name].items():
            expected_scores[f'contrast_{name}_{field}'] = value
    producer_scores = npz(controls_root / 'comparisons/scores.npz')
    require(set(producer_scores) == set(expected_scores), 'producer score-array membership')
    require(set(saved) == selected_keys | {'control_' + name for name in expected_scores} | {'control_paired_metrics'},
            'combined paired archive exact membership')
    for name, value in expected_scores.items():
        close(value, producer_scores[name], 'producer score: ' + name, errors)
        close(value, saved['control_' + name], 'analysis score: ' + name, errors)
    producer_summary = io._json(controls_root / 'comparisons/summary.json')
    same(io._json(analysis / 'control_comparisons_summary.json'), producer_summary, 'retained comparison summary')
    verify_summary(producer_summary, banks, contrasts, rows, slots, errors)
    metric_banks = {arm: {field: metric_values(bank[field + '_layer'], bank[field + '_regions'], arm.split('_', 1)[0])
                         for field in ('own', 'control', 'delta')} for arm, bank in banks.items()}
    contrast_metric_banks, definitions = {}, []
    for name, (left, right) in PAIRS.items():
        comparison = name.split('_', 1)[0]
        values = metric_values(contrasts[name]['delta_layer'], contrasts[name]['delta_regions'], comparison)
        close(values, metric_banks[left]['delta'] - metric_banks[right]['delta'], 'paired metric decomposition: ' + name, errors)
        close(values, (metric_banks[left]['own'] - metric_banks[right]['own'])
              - (metric_banks[left]['control'] - metric_banks[right]['control']), 'own/control decomposition: ' + name, errors)
        contrast_metric_banks[name] = values
        definitions.extend({**definition, 'contrast': name, 'id': name + ':' + definition['id']}
                           for definition in metric_definitions(comparison))
    values = np.concatenate(list(contrast_metric_banks.values()), axis=1)
    require(values.shape == (148, 176), 'P2 eight-by-22 metric geometry')
    close(values, saved['control_paired_metrics'], 'retained P2 paired metrics', errors)
    p2 = report['P2']
    require(p2['conditional_on_fit_id'] == 1 and p2['n_independent_control_fits'] == 1 and p2['fit_sd'] is None,
            'P2 cannot claim fit variance or independent control replications')
    same(p2['metric_definitions'], definitions, 'P2 exact metric definitions, locations and historical labels')
    require(p2['family']['contrasts'] == 352 and set(p2['tasks']) == set(TASKS), 'P2 one complete 352-family')
    modes = ('conditional_item', 'conditional_component')
    require(set(p2['family']['simultaneous_radius95']) == set(modes), 'P2 conditional uncertainty modes')
    design, populations = bootstrap_memberships(saved, rows, slots)
    points, draws, intervals = {}, {}, {}
    for task in TASKS:
        bank = values[design[task]['indices']]
        points[task] = bank.mean(axis=0)
        close(points[task], p2['tasks'][task]['mean'], task + ': P2 means', errors)
        require(set(p2['tasks'][task]['intervals']) == set(modes), 'P2 must not use crossed fit draws')
        draws[task] = {mode: design[task][mode] @ bank for mode in modes}
        intervals[task] = {}
    radii = {}
    for mode in modes:
        error_bank = np.concatenate([draws[task][mode] - points[task] for task in TASKS], axis=1)
        radii[mode] = float(np.quantile(np.abs(error_bank).max(axis=1), .95))
        close(radii[mode], p2['family']['simultaneous_radius95'][mode], '352-family radius: ' + mode, errors)
        for task in TASKS:
            intervals[task][mode] = {'pointwise95': np.quantile(draws[task][mode], [.025, .975], axis=0).T,
                                     'simultaneous95': np.column_stack([points[task] - radii[mode], points[task] + radii[mode]])}
            require(set(p2['tasks'][task]['intervals'][mode]) == set(intervals[task][mode]), 'P2 interval fields')
            for kind, value in intervals[task][mode].items():
                close(value, p2['tasks'][task]['intervals'][mode][kind], f'{task}/{mode}/{kind}', errors)
    with (analysis / 'control_metrics.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 352, 'P2 CSV must have 352 rows')
    for task_index, task in enumerate(TASKS):
        for j, definition in enumerate(definitions):
            row = csv_rows[task_index * 176 + j]
            require(row['task'] == task and row['conditional_fit_id'] == '1' and row['contrast'] == definition['contrast']
                    and row['metric'] == definition['id'] and row['kind'] == definition['kind']
                    and row['virtual_indices'] == ' '.join(map(str, definition['virtual_indices'])), 'P2 CSV identity/support')
            close(float(row['mean']), points[task][j], 'P2 CSV mean', errors)
            for mode in modes:
                for kind, label in [('pointwise95', 'pointwise'), ('simultaneous95', 'simultaneous')]:
                    close([float(row[f'{mode}_{label}_low']), float(row[f'{mode}_{label}_high'])],
                          intervals[task][mode][kind][j], 'P2 CSV interval', errors)
    examples = frozen_examples(io._json(analysis / 'examples.json'), rows, slots, metric_banks, contrast_metric_banks)
    aggregate_decomposition = {task: {arm: {field: value[design[task]['indices']].mean(axis=0).tolist()
        for field, value in bank.items()} for arm, bank in metric_banks.items()} for task in TASKS}
    for path, record in consumed.items():
        io._verify_record(Path(path), record)
    io._verify_record(Path(__file__), source_record)
    require('torch' not in sys.modules, 'audit unexpectedly imported torch')
    result = {'status': 'passed', 'scope': 'Independent actual P2 result audit; conditional fit01 only. No new estimand, inference, causal conclusion, or Huginn interpretation receipt.',
              'analysis_validation': full_validation, 'accepted_P1_identity_sha256': ACCEPTED_P1_ID,
              'handoff': str(proof_root), 'analysis': str(analysis), 'audit_script': source_record,
              'input_records_rechecked': len(consumed), 'accepted_P1_exact_equality': True,
              'score_and_summary_verification': errors, 'populations': populations,
              'target_support': SUPPORT['target'], 'position_support': SUPPORT['position'],
              'target_late_band': LATE['target'], 'position_late_band': LATE['position'],
              'uncertainty': {'draws': DRAWS, 'seed': SEED, 'family_size': 352, 'modes': list(modes),
                              'simultaneous_radius95': radii, 'fit_variance_estimated': False},
              'metric_definitions': definitions, 'P2': p2, 'arm_metric_decomposition': aggregate_decomposition,
              'frozen_item_P2_effects': examples,
              'limits': ['One preselected fit01 control experiment cannot establish control fit variance.',
                         'Target contrast uses 0–189 and excludes target identity 190/main identity 191; position contrast uses 0–190 and excludes identity 191.',
                         'Historical final-third intersection starts at 176, not 177; historical local 26–37 remains labelled selected.',
                         'Item/component uncertainty is conditional on the fitted controls; curated prompts do not establish an IID population.',
                         'Component draws retain all component items and use ratio means; they capture shared eligible concepts only.',
                         'Any-layer opportunities differ with support; no unsupported-column or cross-support comparison is silently introduced.',
                         'Examples use frozen random/failure selections and cannot support effect-selected or causal-use conclusions.'],
              'opaque_cache_deserialized': False, 'scientific_artifacts_modified': False, 'cloud_actions': 0}
    io._new_json(out, result)
    return {'status': 'passed', 'audit_output': str(out), 'audit_output_record': io._record(out),
            'analysis_identity_sha256': full_validation['identity_sha256'], 'maximum_absolute_difference': errors['maximum_absolute_difference'],
            'P2_family_size': 352, 'accepted_P1_exact_equality': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True, help='Verified ouro_handoff_* directory containing COPY_VERIFIED.json and results/')
    parser.add_argument('--analysis', type=Path, required=True, help='Completed full frozen P1/P2 analysis directory')
    parser.add_argument('--accepted-p1', type=Path, default=ROUND / 'analysis/main_run01')
    parser.add_argument('--ouro-src', type=Path, default=Path('/home/moloch/ouro_project/src'))
    parser.add_argument('--output', type=Path, default=Path('/tmp') / ('ouro_p2_actual_audit_' + uuid.uuid4().hex + '.json'))
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == '__main__':
    main()
