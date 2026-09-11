#!/usr/bin/env python3
"""Tokenize annotations before any confirmation forward/readout is exposed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROUND = Path(__file__).resolve().parents[1]
ROOT = ROUND.parents[1]
OLD = ROUND.parent / 'refit_round_2026-09-07'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), '/home/moloch/ouro_project/src', str(OLD / 'deployment')]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def prepare(document, snapshot):
    import transformers
    from ouro_jlens import evaldata, evaluate
    import evaluate_refits as common
    tok = transformers.AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
    def encode(text):
        return [tok.bos_token_id, *tok(text, truncation=True, max_length=511).input_ids]
    items, alignments = [], []
    for row in document['items']:
        labels = row['intermediates']
        ids, dropped = evaldata.readout_context(encode, row['prompt'], row['target'])
        if not ids or dropped > 1 or len(tok(row['prompt']).input_ids) > 511:
            raise ValueError(f'Unestablished or truncated readout boundary: {row["name"]}')
        tokens = {name: evaldata.single_token_ids(tok, evaldata.surface_forms(name)) for name in labels}
        item = evaldata.Item(row['name'], 'multihop', row['prompt'], row['target'], labels, ids, tokens)
        item.leaked = {name: any(t in ids for t in forms) for name, forms in tokens.items()}
        items.append(item)
        alignments.append({'name': item.name, 'dropped_prompt_tokens': dropped, 'untruncated': True})
    model = SimpleNamespace(tokenizer=tok)
    with common.task_names(evaluate, items, tasks=('multihop',)) as registry:
        if len(registry['multihop'].names) > 128:
            raise ValueError('Historical 128-name axis exceeded')
        rows = common._item_rows(model, items, registry, evaluate)
        names, forms, population = common._population(rows, registry, historical=False)
    population['score_policy'] = ('zero-based full-vocabulary rank<10; min over original token forms; '
                                  'controls averaged within label, labels within item; primary averages fixed layers')
    for row, annotation in zip(rows, document['items']):
        row.update({k: annotation[k] for k in ('concept_family_id', 'prompt_family_id', 'dependency_group_id')})
    # A control name must never be an alias of its own concept. Reject the
    # candidate population before freeze; do not alter historical weights.
    collisions = []
    catalogue = list(forms['multihop'])
    for i, name in enumerate(catalogue):
        for other in catalogue[i + 1:]:
            overlap = sorted(set(forms['multihop'][name]) & set(forms['multihop'][other]))
            if overlap:
                collisions.append({'left': name, 'right': other, 'token_ids': overlap})
    result = {'schema': 'confirmation_population.v1', 'annotation_sha256': digest(document), 'rows': rows,
              'names': names, 'token_forms': forms, 'population': population, 'boundary_checks': alignments,
              'cross_concept_token_collisions': collisions,
              'score_policy': 'fixed-layer zero-based full-vocabulary rank < 10, minimum over original surface forms; equally weighted controls per eligible label, labels per item, eligible items; primary fixed-layer mean'}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = prepare(json.loads(args.benchmark.read_text()), args.snapshot)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'items': len(result['rows']), 'eligible': sum(any(r['eligible']) for r in result['rows']), 'names': len(result['names']['multihop']), 'collisions': result['cross_concept_token_collisions']}))

if __name__ == '__main__':
    main()
