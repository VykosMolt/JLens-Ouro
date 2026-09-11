#!/usr/bin/env python3
"""Apply the blinded dependency/relation audit to an assembled annotation document."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / 'dependency_audit.json'
REVIEW = HERE.parent / 'reviews' / 'final_dependency_review.json'
EXPECTED_AUDIT_SHA256 = '5640316b41c142b1054c4a53175e1cef732a4580dd3352c2aa9f00580ba2a8d7'
EXPECTED_CONTENT_SHA256 = 'fcd0ddabfcdf87bb336b548a47632fa61edbda54274d9a1c551b18541e8c9d5f'
APPROVED_PROMPT_REVISION = {
    'item': 'confirmation-062',
    'before': 'Fact: The river whose delta occupies the southern tip of Vietnam has its headwaters on the ',
    'after': 'Fact: The river whose delta occupies much of southern Vietnam has its headwaters on the ',
    'assessment': 'The Mekong remains the intermediate and Tibetan Plateau the answer. The revised downstream-location clue preserves the river-headwaters template and all dependency components.',
}


def serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': sha256(raw)}


def reviewed_content(document):
    return {
        'items': [{key: row[key] for key in ('name', 'task', 'prompt', 'target', 'intermediates', 'concept_family_id')}
                  for row in document['items']],
        'concepts': [{key: row[key] for key in ('concept_family_id', 'intermediate')}
                     for row in document['concepts']],
    }


def normalized(value):
    return ' '.join(value.casefold().split())


def connected_components(names, edges):
    adjacency = {name: set() for name in names}
    for edge in edges:
        a, b = edge['left'], edge['right']
        if a not in adjacency or b not in adjacency or a == b:
            raise ValueError('Invalid dependency edge endpoints')
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(names)
    result = []
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            name = pending.pop()
            if name not in component:
                component.add(name)
                pending.extend(adjacency[name] - component)
        remaining -= component
        result.append(sorted(component))
    return sorted(result, key=lambda values: values[0])


def finalize(document):
    audit_raw = AUDIT.read_bytes()
    if sha256(audit_raw) != EXPECTED_AUDIT_SHA256:
        raise ValueError('Dependency audit changed; independent review is required')
    audit = json.loads(audit_raw)
    if sha256(serialized(reviewed_content(document))) != EXPECTED_CONTENT_SHA256:
        raise ValueError('Assembled prompt/target/intermediate content differs from the reviewed 160-item content')
    if document.get('blind_to_new_method_outcomes') is not True:
        raise ValueError('Assembled document does not affirm outcome-blinded construction')
    if len(document['items']) != 160 or len(document['concepts']) != 80:
        raise ValueError('Expected 160 items and 80 concepts before eligibility filtering')
    result = deepcopy(document)
    rows = result['items']
    by_name = {row['name']: row for row in rows}
    if len(by_name) != 160:
        raise ValueError('Duplicate item names')
    taxonomy = {row['item']: row for row in audit['relation_taxonomy']['items']}
    if set(taxonomy) != set(by_name):
        raise ValueError('Audit taxonomy does not cover assembled items exactly')
    edges = deepcopy(audit['edges'])
    indexed_edges = {(edge['left'], edge['right']): edge for edge in edges}

    def add_reason(a, b, reason):
        left, right = sorted((a, b))
        pair = (left, right)
        if pair not in indexed_edges:
            indexed_edges[pair] = {'left': left, 'right': right,
                                   'left_number': int(left.rsplit('-', 1)[1]),
                                   'right_number': int(right.rsplit('-', 1)[1]), 'reasons': []}
            edges.append(indexed_edges[pair])
        if reason not in indexed_edges[pair]['reasons']:
            indexed_edges[pair]['reasons'].append(reason)

    concept_items = defaultdict(list)
    for row in rows:
        concept_items[row['concept_family_id']].append(row['name'])
    concepts = {row['concept_family_id']: row for row in result['concepts']}
    if set(concepts) != set(concept_items) or len(concepts) != 80:
        raise ValueError('Concept catalogue and item assignments disagree')
    # Alias metadata is not added to scoring forms. Code homographs are linked
    # conservatively without asserting that the underlying concepts are identical.
    alias_collisions = []
    for a, b in combinations(result['concepts'], 2):
        left_aliases = {normalized(a['intermediate']), *(normalized(s) for s in a.get('semantic_aliases', []))}
        right_aliases = {normalized(b['intermediate']), *(normalized(s) for s in b.get('semantic_aliases', []))}
        for item in concept_items[a['concept_family_id']]:
            left_aliases.update(normalized(s) for s in by_name[item].get('intermediate_aliases', []))
        for item in concept_items[b['concept_family_id']]:
            right_aliases.update(normalized(s) for s in by_name[item].get('intermediate_aliases', []))
        overlap = sorted(left_aliases & right_aliases)
        if overlap:
            collision = {'left_concept': a['concept_family_id'], 'right_concept': b['concept_family_id'],
                         'left_intermediate': a['intermediate'], 'right_intermediate': b['intermediate'],
                         'overlap': overlap, 'meaning': 'Shared annotated surface alias/code; this does not establish semantic identity.'}
            alias_collisions.append(collision)
            for left in concept_items[a['concept_family_id']]:
                for right in concept_items[b['concept_family_id']]:
                    add_reason(left, right, {'kind': 'annotated_alias_or_code_collision', 'overlap': overlap})
    # Recheck answers including any assembled alias metadata, not just old strings.
    for a, b in combinations(rows, 2):
        left = {normalized(a['target']), *(normalized(s) for s in a.get('target_aliases', []))}
        right = {normalized(b['target']), *(normalized(s) for s in b.get('target_aliases', []))}
        overlap = sorted(left & right)
        if overlap:
            add_reason(a['name'], b['name'], {'kind': 'assembled_answer_or_alias_collision', 'overlap': overlap})
    edges.sort(key=lambda edge: (edge['left'], edge['right']))
    components = connected_components(by_name, edges)
    expected = {frozenset(row['items']) for row in audit['components']}
    if {frozenset(component) for component in components} != expected:
        raise ValueError('Assembled alias/answer metadata changes audited components; independent re-review required')
    group_for = {name: f'dg{index:03}' for index, component in enumerate(components, 1) for name in component}
    for row in rows:
        reviewed = taxonomy[row['name']]
        row['dependency_group_id'] = group_for[row['name']]
        row['relation_type'] = reviewed['requested_relation']
        row['relation_type_status'] = reviewed['relation_exposure_status']
        row['relation_review'] = {
            'status': 'independently_reviewed_against_revised_prompt',
            'requested_relation': reviewed['requested_relation'],
            'exposure_status': reviewed['relation_exposure_status'],
            'discovery_examples': reviewed['discovery_examples'],
            'classification_reason': reviewed['classification_reason'],
            'concrete_template_families': reviewed['concrete_template_families'],
        }
    for concept in result['concepts']:
        names = concept_items[concept['concept_family_id']]
        groups = {group_for[name] for name in names}
        if len(groups) != 1:
            raise ValueError('One concept was split across dependency groups')
        concept['dependency_group_id'] = groups.pop()
        concept['relation_types'] = sorted({by_name[name]['relation_type'] for name in names})
        concept['relation_type_statuses'] = sorted({by_name[name]['relation_type_status'] for name in names})
    sizes = sorted((len(component) for component in components), reverse=True)
    sum_squares = sum(size * size for size in sizes)
    effective = len(rows) ** 2 / sum_squares
    precision = {
        'population_stage': '160 annotated candidates before token eligibility filtering',
        'group_count': len(components), 'group_sizes_descending': sizes,
        'sum_squared_group_sizes': sum_squares, 'effective_group_count': effective,
        'effective_group_count_formula': 'N^2 / sum_g(n_g^2)',
        'scenarios': [{'assumed_SD': sd, 'approximate_95pct_halfwidth': 1.96 * sd / math.sqrt(effective)}
                      for sd in (0.30, 0.45, 0.60)],
        'eligibility_rule': 'Recompute induced dependency components and precision after eligibility exclusions before freeze.',
    }
    balance = result['design'].setdefault('relation_balance', {})
    balance.update({'status': 'independently_reviewed_revised_prompt_taxonomy',
                    'counts': dict(Counter(row['relation_type_status'] for row in rows)),
                    'granularity': audit['relation_taxonomy']['granularity'],
                    'caution': audit['relation_taxonomy']['caution']})
    dependency = result.setdefault('dependency_design', {})
    dependency.update({
        'status': 'independently_reviewed_connected_components',
        'independence_unit': 'dependency_group_id',
        'group_rule': audit['grouping_policy']['unit'],
        'grouping_policy': audit['grouping_policy'],
        'expected_groups': len(components), 'expected_group_sizes': sizes,
        'precision': precision,
    })
    result['dependency_graph'] = {
        'audit_sha256': EXPECTED_AUDIT_SHA256, 'edges': edges,
        'components': [{'dependency_group_id': f'dg{index:03}', 'size': len(component), 'items': component}
                       for index, component in enumerate(components, 1)],
        'annotated_alias_or_code_collisions': alias_collisions,
        'approved_prompt_revision': APPROVED_PROMPT_REVISION,
    }
    result['dependency_integration'] = {
        'status': 'completed', 'reviewed_content_sha256': EXPECTED_CONTENT_SHA256,
        'new_confirmation_results_exposed': False,
        'scope': 'Dependency groups and requested-relation metadata only; factual sources, eligibility and overall freeze approval are separate.',
    }
    # Verify preservation mechanically, permitting only this integrator's owned metadata.
    preserved = deepcopy(result)
    for destination, original in zip(preserved['items'], document['items']):
        for key in ('dependency_group_id', 'relation_type', 'relation_type_status', 'relation_review'):
            if key in original:
                destination[key] = original[key]
            else:
                destination.pop(key, None)
    for destination, original in zip(preserved['concepts'], document['concepts']):
        for key in ('dependency_group_id', 'relation_types', 'relation_type_statuses'):
            if key in original:
                destination[key] = original[key]
            else:
                destination.pop(key, None)
    if 'relation_balance' in document['design']:
        preserved['design']['relation_balance'] = document['design']['relation_balance']
    else:
        preserved['design'].pop('relation_balance', None)
    for key in ('dependency_design', 'dependency_graph', 'dependency_integration'):
        if key in document:
            preserved[key] = document[key]
        else:
            preserved.pop(key, None)
    if preserved != document:
        raise ValueError('Integration changed metadata outside its owned fields')
    return result, {'precision': precision, 'relation_counts': balance['counts'],
                    'alias_code_collisions': alias_collisions, 'edge_count': len(edges),
                    'non_owned_metadata_preserved': True, 'component_membership_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve():
        raise ValueError('Input and output must differ; assembled input is preserved')
    if args.out.exists():
        raise FileExistsError(args.out)
    input_raw = args.input.read_bytes()
    result, proof = finalize(json.loads(input_raw))
    input_pin = {'bytes': len(input_raw), 'sha256': sha256(input_raw)}
    if record(args.input) != input_pin:
        raise ValueError('Assembled input changed during integration')
    output_text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with args.out.open('x') as handle:
        handle.write(output_text)
    review = {
        'schema': 'confirmation_final_dependency_review.v1',
        'status': 'passed', 'new_confirmation_results_exposed': False,
        'scope': 'Final assembled dependency and relation integration; not factual-source or tokenizer-eligibility approval.',
        'reviewed_sources': {str(args.input.resolve()): input_pin, str(AUDIT): record(AUDIT),
                             str(Path(__file__).resolve()): record(Path(__file__).resolve())},
        'final_benchmark': {'path': str(args.out.resolve()), **record(args.out)},
        'reviewed_content_sha256': EXPECTED_CONTENT_SHA256,
        'approved_prompt_revision': APPROVED_PROMPT_REVISION,
        'checks': proof,
        'limits': ['Planning precision is not an observed interval or a finite-sample coverage guarantee.',
                   'Factual/novelty/control/eligibility gates remain separate.',
                   'After eligibility exclusions, recompute induced components rather than retaining stale group counts.'],
    }
    REVIEW.write_text(json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'out': str(args.out), 'review': str(REVIEW), **proof}))


if __name__ == '__main__':
    main()
