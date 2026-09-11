#!/usr/bin/env python3
"""A3: the pinned verifier, executed by the unchanged controller, on synthetic outputs.

Synthetic random tensors exercise validation logic only; they are not measurements.
Usage: python -B tests/verifier_tests.py --work EMPTY_DIRECTORY
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from common import BUNDLE, DIAGNOSTIC, copy_bundle, empty_directory, record, require, save_evidence, tree

ITEM = 'carnival-ocean'
COMPARED = 'Comparison record differs from its recomputation'
INVALID = 'Invalid tensor shape, dtype or values: '


def synthetic(root, spec, binding, torch, worker, json_save, verifier):
    """Random tensors with the exact output schema, including unequal comparisons."""
    generator = torch.Generator().manual_seed(20260910)
    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator).to(dtype)
    names = spec['development_item_names']
    states, logits = randn(3, 192, 2048, dtype=torch.float32), randn(3, 49152, dtype=torch.float32)
    native = {'virtual_states': states, 'physical_states': states.clone(), 'wrapper_states': states.clone(),
              'native_logits': logits, 'unembedded_logits': logits.clone(), 'item_names': names}
    native['unembedded_logits'][0, :5] += 1
    tensors = {'item_names': names, 'items': {}}
    for name, s in zip(names, (5, 4, 3)):
        def run():
            norm_output, head_output = randn(4, 1, s, 2048), randn(1, s, 49152)
            return {'norm_input': randn(4, 1, s, 2048), 'norm_output': norm_output, 'lm_head_input': norm_output[3].clone(),
                    'lm_head_output': head_output, 'logits': head_output.clone(), 'virtual191': randn(2048, dtype=torch.float32)}
        first = run()
        tensors['items'][name] = {
            'token_ids': list(range(s)), 'runs': [first, run()],
            'norm': {label: torch.stack([first['norm_output'][3, 0, -1]] * 2) for label in verifier.NORM_LABELS},
            'lm_head': {label: randn(2, 49152) for label in verifier.NORM_LABELS},
            'lm_head_norm': {label: randn(2, 49152) for label in verifier.COMPOSITE_LABELS},
            'unembed_fp32_virtual191': randn(2, 49152)}
    root.mkdir()
    shutil.copyfile(BUNDLE / 'frozen/run_spec.json', root / 'run_spec.json')
    json_save(root / 'provenance.json', {
        'binding': binding, 'source_records': spec['source_records'], 'model_record': spec['model_record'],
        'dtype': 'torch.bfloat16', 'geometry': [4, 48, 192, 2048],
        'runtime': {'precision': spec['original_precision'], 'gpu': {'name': verifier.GPU}},
        'actual_worker_source': spec['source_records']['evaluation/worker.py'], 'config_record': {'bytes': 0, 'sha256': '0' * 64}})
    worker.save_outputs(root, native, tensors)
    return {'schema': 'confirmation_artifact_manifest.v1', 'binding': binding, 'kind': 'final', 'stage_id': None,
            'outcome': 'complete', 'files': tree(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    work = empty_directory(parser.parse_args().work)
    files = copy_bundle(work / 'bundle')
    sys.path[:0] = [str(work / 'bundle/controller'), str(work / 'bundle/evaluation')]
    import torch
    import artifact_handoff as handoff
    from artifacts import json_save, torch_save
    import validate_outputs as verifier
    import worker
    config = json.loads((DIAGNOSTIC / 'run_config.json').read_text())
    spec = json.loads((BUNDLE / 'frozen/run_spec.json').read_text())
    state = {'name': 'jlens-confirm-' + '0' * 32, 'pod_id': 'fixture-pod', 'account_id': 'fixture-account',
             'run_id': spec['run_id'], 'run_spec_sha256': record(config['run_spec_path'])['sha256'],
             'output_contract_path': config['output_contract_path'], 'output_contract_record': record(config['output_contract_path']),
             'output_contract_sha256': record(config['output_contract_path'])['sha256'],
             'controller_sources': {name: record(work / 'bundle/controller' / name) for name in handoff.SOURCE_NAMES},
             'semantic_verifier': config['semantic_verifier']}
    contract = handoff.load_contract(state)
    conforming = work / 'conforming'
    manifest = synthetic(conforming, spec, handoff.binding(state), torch, worker, json_save, verifier)
    handoff.validate_manifest(state, manifest, contract)
    result = handoff.semantic_validate(conforming, state, manifest, contract)
    recorded = json.loads((conforming / 'diagnostic/comparisons.json').read_text())['comparisons']
    unequal = sum(not r['torch_equal'] for r in recorded)
    require(0 < unequal < len(recorded), 'the conforming set must contain equal and unequal comparisons')

    def edit_torch(root, name, change):
        value = torch.load(root / name, map_location='cpu', weights_only=True)
        change(value)
        (root / name).unlink()
        torch_save(root / name, value)
    def edit_json(root, name, change):
        value = json.loads((root / name).read_text())
        change(value)
        (root / name).unlink()
        json_save(root / name, value)
    def item(value):
        return value['items'][ITEM]
    def first_equal(records):
        return next(r for r in records['comparisons'] if r['torch_equal'])
    def assign(target, key, value):
        target[key] = value

    # (case, mutation of results root and manifest, expected rejection, refresh changed manifest records)
    cases = [
        ('missing_contract_file', lambda r, m: ((r / 'diagnostic/comparisons.json').unlink(), m['files'].pop('diagnostic/comparisons.json')),
         'Manifest files differ from the contract', True),
        ('missing_payload_file', lambda r, m: (r / 'provenance.json').unlink(), 'Payload files differ from the manifest', True),
        ('extra_payload_file', lambda r, m: (r / 'extra.json').write_text('{}'), 'Payload files differ from the manifest', True),
        ('extra_manifest_file', lambda r, m: ((r / 'extra.json').write_text('{}'), assign(m['files'], 'extra.json', None)),
         'Manifest files differ from the contract', True),
        ('payload_bytes_differ', lambda r, m: (r / 'provenance.json').write_text('{}'), 'Payload bytes differ from the manifest', False),
        ('capture_wrong_shape', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: assign(item(v)['runs'][0], 'norm_input', item(v)['runs'][0]['norm_input'][:3])),
         INVALID + ITEM + '/norm_input', True),
        ('recomputation_wrong_shape', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: assign(item(v)['lm_head'], '[2048]', item(v)['lm_head']['[2048]'][:, :2048])),
         INVALID + ITEM + '/lm_head[2048]', True),
        ('capture_wrong_dtype', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: assign(item(v)['runs'][1], 'logits', item(v)['runs'][1]['logits'].float())),
         INVALID + ITEM + '/logits', True),
        ('m1_wrong_dtype', lambda r, m: edit_torch(r, 'development/native.pt', lambda v: assign(v, 'native_logits', v['native_logits'].bfloat16())),
         INVALID + 'native_logits', True),
        ('capture_nonfinite', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: item(v)['runs'][0]['norm_output'].view(-1)[0].fill_(float('inf'))),
         INVALID + ITEM + '/norm_output', True),
        ('m1_nonfinite', lambda r, m: edit_torch(r, 'development/native.pt', lambda v: v['wrapper_states'].view(-1)[0].fill_(float('nan'))),
         INVALID + 'wrapper_states', True),
        ('missing_recomputation', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: item(v)['norm'].pop('[160,2048]/last')),
         'Recomputation schema differs: ' + ITEM, True),
        ('binding_provenance', lambda r, m: edit_json(r, 'provenance.json', lambda v: assign(v['binding'], 'pod_id', 'another-pod')),
         'Provenance binding differs', True),
        ('binding_run_spec', lambda r, m: assign(m['binding'], 'run_spec_sha256', '0' * 64), 'Run specification differs from the binding', True),
        ('source_mismatch', lambda r, m: edit_json(r, 'provenance.json', lambda v: assign(v, 'actual_worker_source', {'bytes': 0, 'sha256': '0' * 64})),
         'Executed sources differ', True),
        ('precision_mismatch', lambda r, m: edit_json(r, 'provenance.json', lambda v: assign(v['runtime']['precision']['matmul'], 'allow_tf32', True)),
         'Precision differs from the frozen original', True),
        ('gpu_mismatch', lambda r, m: edit_json(r, 'provenance.json', lambda v: assign(v['runtime']['gpu'], 'name', 'NVIDIA GeForce RTX 4090')),
         'GPU type differs', True),
        ('failed_manifest', lambda r, m: assign(m, 'outcome', 'failed'), 'Only a complete final manifest is accepted', True),
        ('stopped_manifest', lambda r, m: assign(m, 'outcome', 'stopped'), 'Only a complete final manifest is accepted', True),
        ('stage_manifest', lambda r, m: (assign(m, 'kind', 'stage'), assign(m, 'stage_id', 'development')),
         'Only a complete final manifest is accepted', True),
        ('comparison_equality_flipped', lambda r, m: edit_json(r, 'diagnostic/comparisons.json', lambda v: assign(first_equal(v), 'torch_equal', False)), COMPARED, True),
        ('comparison_count_changed', lambda r, m: edit_json(r, 'diagnostic/comparisons.json', lambda v: assign(first_equal(v), 'differing_elements', 1)), COMPARED, True),
        ('comparison_record_removed', lambda r, m: edit_json(r, 'diagnostic/comparisons.json', lambda v: v['comparisons'].pop()), COMPARED, True),
        ('exit_step_changed', lambda r, m: edit_json(r, 'diagnostic/comparisons.json', lambda v: assign(v['exit_steps'][ITEM], 'exit_step', 0)), COMPARED, True),
        ('tensor_changed_after_record', lambda r, m: edit_torch(r, 'diagnostic/tensors.pt', lambda v: item(v)['lm_head']['[2048]'][0, 7].add_(1)), COMPARED, True),
    ]
    rejected = {}
    for name, mutate, expected, refresh in cases:
        root, changed = work / name, copy.deepcopy(manifest)
        shutil.copytree(conforming, root)
        mutate(root, changed)
        if refresh:
            changed['files'] = {p: record(root / p) if (root / p).is_file() else value for p, value in changed['files'].items()}
        try:
            handoff.semantic_validate(root, state, changed, contract)
        except ValueError as error:
            require(str(error) == expected, f'{name} rejected for another reason: {error}')
            rejected[name] = str(error)
            continue
        raise AssertionError('verifier accepted ' + name)
    save_evidence('A3_verifier_tests.json', {
        'schema': 'native_logit_verifier_tests.v1', 'status': 'passed',
        'label': 'Synthetic random tensors test validation logic only; they are not measurements.',
        'bundle_records': files, 'verifier_record': config['semantic_verifier']['record'],
        'executed_by': 'artifact_handoff.semantic_validate (unchanged controller copy) on the pinned verifier path',
        'accepted': {'result': result, 'comparisons': len(recorded), 'unequal_comparisons_accepted': unequal},
        'rejected': rejected})
    print(json.dumps({'status': 'passed', 'accepted': 1, 'rejected': len(rejected)}))


if __name__ == '__main__':
    main()
