#!/usr/bin/env python3
"""Write RUN_SPECIFICATION.json for the accepted confirmation run from its frozen and accepted records."""
import hashlib
import json
from pathlib import Path

RESEARCH = Path('/home/moloch/ouro_project/jacobian-lens/research')
ROUND = RESEARCH / 'confirmation_2026-09-09'
FINAL = ROUND / ('cloud_leases/attempt_06/handoff/accepted/ouro_confirmation_20260911_fixed160_native1/'
                 'ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef')
RESULTS = FINAL / 'results'
BUNDLE = ROUND / 'corrections/native_check_v1/bundle'
OUT = Path(__file__).resolve().parents[1] / 'RUN_SPECIFICATION.json'
RETAINED = json.loads((Path(__file__).resolve().parents[1] / 'replay/retained_bank_paths.json').read_text())


def record(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def virtual(loop, physical):
    """One-based loop and physical block to zero-based virtual index."""
    return 48 * (loop - 1) + (physical - 1)


spec = json.loads((RESULTS / 'run_spec.json').read_text())
provenance = json.loads((RESULTS / 'provenance.json').read_text())
population = json.loads((RESULTS / 'population.json').read_text())
manifest = json.loads((BUNDLE / 'legacy/model_manifest.json').read_text())
SNAPSHOT = Path('/home/moloch/ouro_project/artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B/snapshots/'
                '1ed04250da1a9936042725d302e81c8fa2ab5abd')
config_record = next(f for f in manifest['files'] if f['path'] == 'config.json')
assert record(SNAPSHOT / 'config.json')['sha256'] == config_record['sha256']
config = json.loads((SNAPSHOT / 'config.json').read_text())
rows = population['rows']
names = population['names']['multihop']

band = spec['primary']['virtual_indices']
assert band == list(range(virtual(4, 26), virtual(4, 37) + 1)) == list(range(169, 181))
assert all(sum(r['eligible']) == 1 and all(len(c) == 79 for c, e in zip(r['control_indices'], r['eligible']) if e) for r in rows)
assert all(c == [k for k in range(len(names)) if k not in r['own_index']]
           for r in rows for c, e in zip(r['control_indices'], r['eligible']) if e)

banks = {}
for arm, entry in spec['bank_records'].items():
    target = 190 if arm == 'penultimate' else 191
    columns = len(spec['arm_support'][arm])
    banks[arm] = {'worker_path': entry['worker_path'], 'record': entry['record'], 'retained_copy': RETAINED[entry['worker_path']],
                  'state_key': f"J['{arm}']" if arm in ('sampled_sum', 'diagonal') else 'J',
                  'source_virtual_layers': [0, target - 1], 'target_virtual_layer': target, 'scored_columns': columns,
                  'identity_row': 191 if columns == 192 else None}

value = {
    'schema': 'run_specification.v1',
    'run_id': spec['run_id'],
    'accepted_final_manifest': {'digest': FINAL.name, 'manifest': record(FINAL / 'MANIFEST.json'), 'receipt': record(FINAL / 'RECEIPT.json')},
    'lease_binding': {k: provenance['binding'][k] for k in ('lease_name', 'pod_id', 'run_spec_sha256', 'output_contract_sha256')},
    'freeze_chain': {'original': record(ROUND / 'FREEZE.json'), 'precision_v1': record(ROUND / 'corrections/precision_v1/FREEZE.json'),
                     'native_check_v1': record(ROUND / 'corrections/native_check_v1/FREEZE.json'),
                     'executed_run_spec': record(RESULTS / 'run_spec.json')},
    'model': {**spec['model_identity'], 'weights_dtype': provenance['dtype'], 'geometry_loops_blocks_virtual_width': provenance['geometry'],
              'vocab_size': config['vocab_size'], 'rms_norm_eps': config['rms_norm_eps'], 'tie_word_embeddings': config['tie_word_embeddings'],
              'attention_implementation': provenance['runtime']['attention_implementation'],
              'files': {f['path']: {'bytes': f['bytes'], 'sha256': f['sha256']} for f in manifest['files']}},
    'tokenizer': {f['path']: f['sha256'] for f in manifest['files'] if 'token' in f['path'] or f['path'] in ('vocab.json', 'merges.txt')},
    'estimator_banks': banks,
    'raw_arm': {'transport': None, 'scored_columns': 192},
    'population': {'record': spec['population_record'], 'benchmark_record': spec['benchmark_record'], 'plan_record': spec['plan_record'],
                   'annotation_review_record': spec['annotation_review_record'], 'items': len(rows), 'names': len(names),
                   'concept_families': len({r['concept_family_id'] for r in rows}), 'prompt_families': len({r['prompt_family_id'] for r in rows}),
                   'dependency_groups': len({r['dependency_group_id'] for r in rows}), 'eligible_slots_per_item': 1,
                   'controls_per_slot': 79, 'control_rule': 'all other names in the catalogue',
                   'alias_forms_per_name': {str(k): sum(1 for r in rows for v in r['intermediate_tokens'].values() if len(v) == k) for k in (1, 2, 3, 4)},
                   'leaked_items': sum(any(r['leaked']) for r in rows), 'readout_position': 'last token of the common prefix of encode(prompt) and encode(prompt + target)',
                   'development_items_excluded': spec['development_item_names']},
    'score': {'rank': 'full-vocabulary rank (0 = top) from a descending argsort of FP32-cast BF16 logits, minimum over alias forms',
              'hit': 'rank < 10', 'intended': 'hit of the item\'s own name', 'control': 'mean hit over the 79 control names',
              'excess': 'intended - control', 'item_weight': 'equal', 'primary': spec['primary'], 'uncertainty': spec['uncertainty'],
              'secondary_family': [s['id'] for s in spec['secondary_specs']] if 'secondary_specs' in spec else None},
    'scoring_path': {'transport': 'torch.einsum("vde,ve->vd", J, h) in FP32 on CUDA, one call per item over all scored columns',
                     'unembedding': 'residual cast to BF16; OuroRMSNorm computes in FP32 and casts back; BF16 lm_head; one [columns, 2048] call per item',
                     'rows_per_call': {arm: len(spec['arm_support'][arm]) for arm in ('raw', 'fit01', 'fit02', 'penultimate', 'sampled_sum', 'diagonal')},
                     'exit_logits': 'one [160, 2048] call per loop at virtual 47, 95, 143, 191; used for KL and top-1 diagnostics only'},
    'indexing': {'formula': 'virtual = 48 * (loop - 1) + (physical - 1), loop and physical one-based, virtual zero-based',
                 'primary_band': {'virtual': [band[0], band[-1]], 'loop': 4, 'physical': [26, 37]},
                 'layer32_contrast': {'virtual': virtual(4, 32), 'loop': 4, 'physical': 32},
                 'final_third': {'virtual': [virtual(4, 33), virtual(4, 48)], 'loop': 4, 'physical': [33, 48]},
                 'early_loops': {str(loop): [virtual(loop, 1), virtual(loop, 48)] for loop in (1, 2, 3)},
                 'penultimate_target': {'virtual': 190, 'loop': 4, 'physical': 47},
                 'final_target': {'virtual': 191, 'loop': 4, 'physical': 48, 'note': 'the last block output, before the final norm'},
                 'source_comment': 'measurement.py: "Reports number physical layers 1..48; source tensors use 0..47."'},
    'environment': {'gpu': {k: provenance['runtime']['gpu'][k] for k in ('name', 'capability', 'total_memory')},
                    **{k: provenance['runtime']['environment'][k] for k in ('python_version', 'torch_git', 'cuda_runtime', 'cudnn_version',
                                                                           'cpu_threads', 'interop_threads', 'platform', 'packages')},
                    'precision': provenance['runtime']['precision']},
}
assert value['indexing']['layer32_contrast']['virtual'] == 175 and value['indexing']['final_third']['virtual'] == [176, 191]
assert value['indexing']['early_loops'] == {'1': [0, 47], '2': [48, 95], '3': [96, 143]}
with OUT.open('x') as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(OUT)
