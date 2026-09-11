"""Independent N=100 arithmetic and continuation audit, 7 September 2026.

Does not import the study's metric/report helpers. This reproduces the original
any-layer statistic; it does not select or examine favorable individual layers.
Historical artifacts are read-only. Run with Python and NumPy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path('/home/moloch/ouro_project')
SOURCE = PROJECT / 'artifacts/jlens/retrieved/jlens-b300-20260905-0359/eval/n100_exit3'
ALL_EXITS = PROJECT / 'artifacts/jlens/eval/b300_local_allexits_strict'
OUT = Path(__file__).resolve().parent
OPERATIONS = {'addition', 'subtraction', 'multiplication', 'division', 'mod', 'squared'}
WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
         'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
         'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
         'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
         'twenty': 20}


def historical_pass(text: str, target: str) -> bool:
    # Independent literal implementation of the published criterion.
    answer = text.strip().strip('"').lower()
    target = target.strip().lower()
    if answer[:len(target)] != target:
        return False
    return len(answer) == len(target) or not answer[len(target)].isalnum()


def strict_numeric_pass(text: str, target: str) -> bool:
    # Retain the original rule except that a decimal/fraction/exponent attached
    # to an integer target must not be accepted merely because of punctuation.
    if not historical_pass(text, target):
        return False
    if not re.fullmatch(r'[+-]?\d+', target.strip()):
        return True
    answer = text.strip().strip('"').lower()
    tail = answer[len(target.strip()):]
    return not bool(re.match(r'(?:[.,/]\d|[eE][+-]?\d)', tail))


def numeric_equivalence_pass(text: str, target: str) -> bool:
    # Narrow sensitivity, NOT complete semantic adjudication: additionally
    # accept an exact initial integer for a number-word target.
    if strict_numeric_pass(text, target):
        return True
    word = target.strip().lower()
    if word not in WORDS:
        return False
    return strict_numeric_pass(text, str(WORDS[word]))


NOTES = {
    'div-sub-left': 'False positive: 3.8 is not the integer target 3.',
    'nested4-add-mult-add-div': 'False positive: 3.75 is not the integer target 3.',
    'word-parens': 'False negative for numerical meaning: initial 20 equals target twenty; retained separately.',
    'dual-stars-visible-opposite': 'daytime fails the exact day boundary criterion but is a plausible semantic equivalent; no relabeling.',
    'roman-rings-olympic': 'Response contains V after written as, rather than at the required initial position; truncated after four tokens; no relabeling.',
    'inv-antarctica-opposite': 'the North Pole is related to north but not an exact direction response; no semantic relabeling.',
    'super-populous-capital': 'Dataset assumes China is the most populous country; task-reference fidelity is distinct from current factual correctness.',
    'firstletter-populous-country': 'Dataset assumes most-populous-country initial C; target validity is a separate question.',
    'nested-add-mult-sub': 'Output 9 is the labeled intermediate but not the target 5; preserve as a counterexample to reasoning-step interpretation.',
    'atomic-26-symbol': 'Iron example retained; main N=100 ranks independently checked below.',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(1 << 20):
            h.update(block)
        return h.hexdigest()


def bootstrap_means(rows: np.ndarray) -> np.ndarray:
    # Same seeded, paired item percentile bootstrap as the published table.
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(rows), size=(20_000, len(rows)))
    return np.quantile(rows[draws].mean(axis=1), [0.025, 0.975], axis=0)


def main() -> None:
    items = json.loads((SOURCE / 'items.json').read_text())
    names = json.loads((SOURCE / 'task_names.json').read_text())
    arrays = dict(np.load(SOURCE / 'arrays.npz', allow_pickle=False))
    local = dict(np.load(ALL_EXITS / 'arrays.npz', allow_pickle=False))
    assert len(items) == len({item['name'] for item in items}) == 148
    assert len({item['prompt'] for item in items}) == 148
    schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}
    annotated = []
    for i, item in enumerate(items):
        keep = [k for k, (s, l, nm) in enumerate(zip(item['scorable'], item['leaked'], item['intermediates']))
                if s and not l and (item['task'] == 'multihop' or nm not in OPERATIONS)]
        row = {'index': i, **item, 'eligible_slots': keep,
               'historical_pass': historical_pass(item['continuation'], item['target']),
               'strict_numeric_pass': strict_numeric_pass(item['continuation'], item['target']),
               'numeric_equivalence_pass': numeric_equivalence_pass(item['continuation'], item['target']),
               'audit_note': NOTES.get(item['name'], 'No further criterion error identified in the retained four-token continuation.')}
        assert row['historical_pass'] == item['correct']
        annotated.append(row)
        for key in ('jlens_exit3', 'logitlens'):
            for slot, own in enumerate(item['own_index']):
                if own >= 0:
                    assert np.array_equal(arrays[key + '_rank'][i, slot], arrays[key + '_allrank'][i, own])
        for key in ('jlens_exit3_allrank', 'logitlens_allrank'):
            assert (arrays[key][i, :len(names[item['task']])] >= 0).all()
            assert (arrays[key][i, len(names[item['task']]):] == -1).all()

    results = {}
    for task in ('multihop', 'order-ops'):
        eligible = [row for row in annotated if row['task'] == task and row['eligible_slots']]
        method_scores = {}
        for method in ('jlens_exit3', 'logitlens'):
            ranks = arrays[method + '_allrank'].reshape(148, 128, 4, 48)
            hit_rows, control_rows = [], []
            for item in eligible:
                own_set = set(item['own_index']) - {-1}
                slot_hit, slot_ctrl = [], []
                for slot in item['eligible_slots']:
                    j = item['own_index'][slot]
                    controls = [c for c, name in enumerate(names[task])
                                if c not in own_set and ((name in OPERATIONS) == (names[task][j] in OPERATIONS))]
                    assert controls
                    # Critical ordering: each true/control name gets its own
                    # any-layer event, THEN controls and within-item slots mean.
                    slot_hit.append((ranks[item['index'], j] < 10).any(axis=-1))
                    slot_ctrl.append((ranks[item['index'], controls] < 10).any(axis=-1).mean(axis=0))
                hit_rows.append(np.mean(slot_hit, axis=0))
                control_rows.append(np.mean(slot_ctrl, axis=0))
            hit, control = np.stack(hit_rows), np.stack(control_rows)
            method_scores[method] = {'hit': hit, 'control': control, 'excess': hit - control}
        delta = method_scores['jlens_exit3']['excess'] - method_scores['logitlens']['excess']
        results[task] = {
            'n_items': len(eligible),
            'n_slots': sum(len(row['eligible_slots']) for row in eligible),
            'n_passing_items_historical': sum(row['historical_pass'] for row in eligible),
            'n_passing_items_strict_numeric': sum(row['strict_numeric_pass'] for row in eligible),
            'n_passing_items_numeric_equivalence': sum(row['numeric_equivalence_pass'] for row in eligible),
            'methods': {method: {key: values.mean(axis=0).tolist() for key, values in metrics.items()}
                        for method, metrics in method_scores.items()},
            'j_minus_logit': delta.mean(axis=0).tolist(),
            'j_minus_logit_ci95': bootstrap_means(delta).T.tolist(),
        }

    ground_truth_checks = {}
    for source_label, data in (('pod_n100', arrays), ('local_all_exits', local)):
        endpoints = data['logitlens_top1'][:, [47, 95, 143, 191]]
        assert np.array_equal(endpoints, data['exit_top1'])
        entry = {'logit_endpoint_equals_actual_exit': True,
                 'final_j_endpoint_equals_final_exit': bool(np.array_equal(data['jlens_exit3_top1'][:, 191], data['exit_top1'][:, 3])),
                 'final_j_to_local_equals_to_final': bool(np.array_equal(data['jlens_exit3_kl_to_local'], data['jlens_exit3_kl_to_final'])),
                 'logit_to_local_equals_to_final': bool(np.array_equal(data['logitlens_kl_to_local'], data['logitlens_kl_to_final'])),
                 'endpoint_argmax_ranks_final_j': dict(Counter(map(str, data['jlens_exit3_rank_of_final_top1'][:, 191]))),
                 'endpoint_argmax_ranks_logit': dict(Counter(map(str, data['logitlens_rank_of_final_top1'][:, 191])))}
        for k in range(3):
            prefix = f'jlens_exit{k}'
            if prefix + '_top1' in data:
                entry[f'local{k}_endpoint_equals_exit'] = bool(np.array_equal(data[prefix + '_top1'][:, k*48+47], data['exit_top1'][:, k]))
                entry[f'local{k}_identity_tail_equals_logit'] = bool(np.array_equal(data[prefix + '_top1'][:, k*48+47:], data['logitlens_top1'][:, k*48+47:]))
        ground_truth_checks[source_label] = entry
    artifacts = {}
    for source in (SOURCE, ALL_EXITS):
        p = json.loads((source / 'provenance.json').read_text())
        artifacts[str(source)] = {
            'output_records_match': {key: digest(source / rec['path']) == rec['sha256']
                                     for key, rec in p['outputs'].items()},
            'current_source_records_match': {rec['path']: digest(PROJECT / rec['path']) == rec['sha256']
                                            for rec in p['source_files']},
            'lens_fit_counts': [rec['identity']['n_fitted'] for rec in p['lens_inputs']],
            'created_at': p['created_at'],
        }
    duplicates = defaultdict(list)
    for row in annotated:
        if row['eligible_slots']:
            labels = [row['intermediates'][k] for k in row['eligible_slots']]
            canonical = tuple(sorted(set(WORDS.get(label, int(label) if label.isdigit() else label) for label in labels), key=str))
            duplicates[(row['task'], str(canonical))].append(row['name'])
    report = {
        'source': str(SOURCE), 'schema': schema,
        'population': {'all_items': len(items), 'raw_task_counts': dict(Counter(row['task'] for row in annotated)),
                       'historical_pass': sum(row['historical_pass'] for row in annotated),
                       'strict_numeric_pass': sum(row['strict_numeric_pass'] for row in annotated),
                       'numeric_equivalence_pass': sum(row['numeric_equivalence_pass'] for row in annotated),
                       'eligible_items': sum(bool(row['eligible_slots']) for row in annotated),
                       'eligible_historical_pass': sum(row['historical_pass'] and bool(row['eligible_slots']) for row in annotated)},
        'headline': results,
        'ground_truth_checks': ground_truth_checks,
        'artifact_checks': artifacts,
        'pod_vs_local_differences': {
            'array_entries': {key: {'different': int((arrays[key] != local[key]).sum()), 'total': int(arrays[key].size)}
                              for key in ('exit_top1', 'jlens_exit3_top1', 'logitlens_top1', 'jlens_exit3_allrank', 'logitlens_allrank')},
            'item_fields': {key: sum(a[key] != b[key] for a, b in zip(items, json.loads((ALL_EXITS / 'items.json').read_text())))
                            for key in items[0]},
        },
        'shared_label_item_groups': {str(key): value for key, value in duplicates.items() if len(value) > 1},
        'random_sample_seed_20260907': [annotated[j]['name'] for j in np.random.default_rng(20260907).choice(148, size=8, replace=False)],
    }
    iron = next(row for row in annotated if row['name'] == 'atomic-26-symbol')
    report['iron_n100_loop1_best_rank_1_based'] = {
        method: int(arrays[method + '_rank'][iron['index'], 0, :48].min()) + 1
        for method in ('jlens_exit3', 'logitlens')}
    (OUT / 'independent_reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'continuation_audit.json').write_text(json.dumps(annotated, indent=2) + '\n')
    with (OUT / 'continuation_audit.csv').open('w', newline='') as f:
        columns = ['index', 'name', 'task', 'target', 'continuation', 'historical_pass', 'strict_numeric_pass', 'numeric_equivalence_pass', 'audit_note']
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(annotated)
    print(json.dumps({key: report[key] for key in ('population', 'headline', 'ground_truth_checks', 'artifact_checks', 'iron_n100_loop1_best_rank_1_based')}, indent=2))


if __name__ == '__main__':
    main()
