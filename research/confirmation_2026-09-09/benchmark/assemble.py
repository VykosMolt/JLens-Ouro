#!/usr/bin/env python3
"""Combine row-bound blinded source audits; never infer approval from an old row.

This is local benchmark preparation. The scientific freeze still requires an
independent review bound to the final benchmark and tokenizer population.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def record(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    args = parser.parse_args()
    candidate = HERE / 'root_revised_candidates.json'
    data = json.loads(candidate.read_text())
    audit_paths = [HERE.parent / 'reviews/benchmark_sources_partial.json',
                   HERE / 'source_audit.json', HERE / 'science_sources.json',
                   HERE / 'country_sources.json', HERE / 'landform_sources.json']
    inputs, matches, rejected, alias_supplements = {}, {}, [], []
    current = {row['name']: row for row in data['items']}
    for path in audit_paths:
        if not path.exists():
            continue
        audit = json.loads(path.read_text())
        if audit.get('new_confirmation_results_exposed') is not False:
            raise ValueError(f'Unblinded or unspecified source audit: {path}')
        inputs[str(path)] = record(path)
        if audit.get('semantic_alias_evidence_supplement'):
            alias_supplements.append({'audit_path': str(path), 'audit_record': inputs[str(path)],
                'annotations': copy.deepcopy(audit['semantic_alias_evidence_supplement']),
                'source_register': copy.deepcopy(audit['sources'])})
        for row in audit.get('items', audit.get('rows', [])):
            name = row['name']
            if name not in current:
                raise ValueError(f'Unknown audited row {name}')
            actual = current[name]
            if (row['prompt'], row['target']) != (actual['prompt'], actual['target']):
                rejected.append({'name': name, 'audit': str(path), 'reason': 'stale prompt or target'})
                continue
            status = row.get('status')
            if status not in ('both_hops_supported', 'supported_by_inspected_official_page_or_indexed_source_text'):
                continue
            ids1 = row.get('hop1_source_ids', [row.get('hop1_source')])
            ids2 = row.get('hop2_source_ids', [row.get('hop2_source')])
            if not ids1 or not ids2 or any(key not in audit['sources'] for key in [*ids1, *ids2]):
                raise ValueError(f'Unsupported asserted source IDs for {name}')
            evidence = {hop: [copy.deepcopy(audit['sources'][key]) for key in ids]
                        for hop, ids in [('hop1', ids1), ('hop2', ids2)]}
            matches.setdefault(name, []).append({'audit_path': str(path), 'audit_record': inputs[str(path)],
                'row_annotation': copy.deepcopy(row), 'evidence': evidence})
    missing = sorted(set(current) - set(matches))
    coverage = {'schema': 'confirmation_source_coverage.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'new_confirmation_results_exposed': False, 'candidate_record': record(candidate),
                'audits': inputs, 'supported_items': len(matches), 'missing_items': missing,
                'stale_audit_rows': rejected}
    save(args.coverage, coverage)
    if missing:
        print(json.dumps({'status': 'incomplete', 'supported_items': len(matches), 'missing_items': missing}))
        return
    # Remove inherited claims that describe abandoned prompts. Exact new source
    # captures remain attributed; a copied source is not a second independent read.
    for key in ('source_register', 'source_evidence_map'):
        data.pop(key, None)
    for row in data['items']:
        for key in ('hop1_claim', 'hop2_claim', 'source_fact_id', 'source_evidence',
                    'source_candidates_from_previous_draft'):
            row.pop(key, None)
        row['source_status'] = 'both_hops_supported_before_model_results'
        row['source_evidence'] = matches[row['name']]
        row['annotation_check'] = {'identifying_clue': 'The exact recorded prompt identifies the annotated intermediate.',
            'requested_attribute': 'The target is the separately requested fact about that intermediate.',
            'blinded_source_evidence': 'Each attached audit is bound to the exact prompt and target, with distinct hop references.',
            'final_independent_acceptance': 'pending exact benchmark and tokenizer-population review'}
    for concept in data['concepts']:
        concept.pop('source', None)
    data['status'] = 'source_assembled_pending_final_dependency_and_annotation_review'
    data['source_audit_records'] = inputs
    data['semantic_alias_evidence_supplements'] = alias_supplements
    data['source_assembly_record'] = record(args.coverage)
    data['source_assembly_utc'] = coverage['created_utc']
    save(args.out, data)
    print(json.dumps({'status': 'assembled_pending_final_review', 'items': len(current), 'out_record': record(args.out)}))

if __name__ == '__main__':
    main()
